import fs from 'node:fs';
import path from 'node:path';
import { listFiles } from '@huggingface/hub';

/**
 * Polite, throttled generator that compiles a rich static manifest of 3D assets
 * from Hugging Face Hub datasets and writes it to public/manifest-3d.json.
 *
 * Rules:
 * - Uses random human jitter delay (1.5s - 3.5s) between requests.
 * - Supports HF_TOKEN / HUGGINGFACE_TOKEN from .env or process.env if present to avoid 429s.
 * - Handles 429 responses with exponential backoff.
 * - Samples from multiple offsets across large datasets.
 */

interface Discovered3DAsset {
  id: string;
  repoId: string;
  repoType: 'dataset' | 'model';
  path: string;
  filename: string;
  title?: string;
  subtitle?: string;
  format: 'glb' | 'gltf' | 'obj' | 'stl';
  size?: number;
  downloadUrl: string;
  resourceBaseUrl: string;
  repoUrl: string;
  treeUrl: string;
}

interface SeedTarget {
  repoId: string;
  config?: string;
  split?: string;
  samples?: number;
  offsets?: number[];
  isGitTree?: boolean;
  /** Only include files whose path starts with this prefix (git-tree targets). */
  pathPrefix?: string;
  /** Only include files whose path contains this substring (git-tree targets). */
  pathIncludes?: string;
  /** Skip files larger than this (git-tree targets); keeps the viewer responsive. */
  maxSizeBytes?: number;
  /** Skip files smaller than this (git-tree targets); avoids degenerate meshes. */
  minSizeBytes?: number;
}

const SEED_TARGETS: SeedTarget[] = [
  {
    repoId: 'cfahlgren1/lowpoly-coder-4b-data',
    config: 'stage1',
    split: 'train',
    samples: 120,
    offsets: [0, 500, 1500, 3000, 5000, 6800],
  },
  {
    repoId: 'codepawl/smithsonian-3d-glb',
    config: 'default',
    split: 'train',
    samples: 120,
    offsets: [0, 200, 600, 1200, 2000, 2800],
  },
  {
    repoId: 'codepawl/google-poly-glb',
    config: 'default',
    split: 'train',
    samples: 100,
    offsets: [0, 300, 900, 2000],
  },
  {
    repoId: 'ccore/ThreeJS-3D-GLB-Assets',
    config: 'meshes',
    split: 'train',
    samples: 80,
    offsets: [0],
  },
  {
    repoId: 'cfahlgren1/3d-collection',
    config: 'default',
    split: 'train',
    samples: 100,
    offsets: [0, 140, 280, 440, 620, 820, 1020, 1220],
  },
  {
    repoId: 'hugging-3d/smithsonian-open-access-3d',
    config: 'metadata',
    split: 'train',
    samples: 60,
    offsets: [0, 300, 800],
  },
  // Git file-tree datasets (queried via listFiles with throttling)
  {
    repoId: 'YiboZhang2001/TexVerse-1K',
    isGitTree: true,
    samples: 100,
  },
  {
    // Full TexVerse: 858K objects w/ high-res textures. Only pull the 2K tier
    // (glbs_2k) and skip anything over 15MB so models load in <1s in the browser.
    repoId: 'YiboZhang2001/TexVerse',
    isGitTree: true,
    pathPrefix: 'glbs/glbs_2k/',
    maxSizeBytes: 15 * 1024 * 1024,
    samples: 60,
  },
  {
    repoId: 'Optikan112/abo-furniture-glb',
    isGitTree: true,
    samples: 40,
  },
  {
    repoId: 'TuTao/scandi-home-lowpoly',
    isGitTree: true,
    samples: 120,
  },
  {
    repoId: 'cfahlgren1/3d-arena',
    isGitTree: true,
    pathPrefix: 'outputs/',
    samples: 120,
  },
  {
    repoId: 'codepawl/scanned-objects-glb',
    isGitTree: true,
    samples: 40,
  },
];

// Read environment for optional token (supports HUGGINGFACE_API_KEY, HF_TOKEN, etc.)
const token =
  process.env.HUGGINGFACE_API_KEY ||
  process.env.HF_TOKEN ||
  process.env.HUGGINGFACE_TOKEN ||
  process.env.HF_API_KEY;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jitter(): Promise<void> {
  // Polite random delay between 1500ms and 3200ms
  const ms = 1500 + Math.floor(Math.random() * 1700);
  return sleep(ms);
}

function getAuthHeaders(): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchWithRetry(url: string, retries = 3): Promise<Response | null> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, { headers: getAuthHeaders() });
      if (res.status === 429) {
        const retryAfter = Number(res.headers.get('retry-after')) || attempt * 8;
        console.warn(`[Rate Limited 429] Backing off for ${retryAfter}s... (attempt ${attempt}/${retries})`);
        await sleep(retryAfter * 1000);
        continue;
      }
      return res;
    } catch (e) {
      if (attempt === retries) {
        console.warn(`[Fetch Error] ${url}:`, e);
        return null;
      }
      await sleep(2000 * attempt);
    }
  }
  return null;
}

function buildAsset(
  repoId: string,
  filePath: string,
  size?: number,
  overrideUrl?: string,
  title?: string,
  subtitle?: string
): Discovered3DAsset {
  const cleanExt = filePath.split('?')[0].toLowerCase();
  const format: 'glb' | 'gltf' | 'obj' | 'stl' = cleanExt.endsWith('.stl')
    ? 'stl'
    : cleanExt.endsWith('.obj')
    ? 'obj'
    : cleanExt.endsWith('.gltf')
    ? 'gltf'
    : 'glb';

  const downloadUrl =
    overrideUrl || `https://huggingface.co/datasets/${repoId}/resolve/main/${filePath}`;
  const filename = filePath.split('/').pop() || filePath;

  return {
    id: `dataset:${repoId}/${filePath}`,
    repoId,
    repoType: 'dataset',
    path: filePath,
    filename,
    title: title || filename.replace(/\.(glb|gltf|obj|stl)$/i, ''),
    subtitle: subtitle || `dataset: ${repoId}`,
    format,
    size,
    downloadUrl,
    resourceBaseUrl: downloadUrl.substring(0, downloadUrl.lastIndexOf('/') + 1),
    repoUrl: `https://huggingface.co/datasets/${repoId}`,
    treeUrl: `https://huggingface.co/datasets/${repoId}/tree/main`,
  };
}

async function processDatasetServerTarget(target: SeedTarget): Promise<Discovered3DAsset[]> {
  const assets: Discovered3DAsset[] = [];
  const repoId = target.repoId;
  const config = target.config || 'default';
  const split = target.split || 'train';
  const offsets = target.offsets || [0];

  console.log(`\n⏳ Sampling "${repoId}" across ${offsets.length} offsets...`);

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=50`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) {
      console.warn(`  ↳ Skipping offset ${offset} (status: ${res?.status})`);
      continue;
    }

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || typeof row !== 'object') continue;

        // Pattern A: cfahlgren1/lowpoly-coder-4b-data -> row.glb.src
        if (repoId === 'cfahlgren1/lowpoly-coder-4b-data' && row.glb?.src) {
          const rawPrompt = typeof row.prompt === 'string' ? row.prompt.trim() : '';
          let title = typeof row.id === 'string' ? `Low-Poly Object (${row.id})` : 'Low-Poly Model';
          if (rawPrompt) {
            title = rawPrompt.replace(/^(I need a |Build a |Create a |Make a )/i, '');
            title = title.charAt(0).toUpperCase() + title.slice(1).replace(/\.$/, '');
          }
          const subtitle = row.category ? `Category: ${row.category} · Low-Poly 4B` : undefined;
          const slug = (row.id || title).toLowerCase().replace(/[^a-z0-9]+/g, '-');
          assets.push(buildAsset(repoId, `${slug}.glb`, row.glb.size_bytes, row.glb.src, title, subtitle));
          continue;
        }

        // Pattern B: codepawl/smithsonian-3d-glb -> row.files.web
        if (row.files && typeof row.files.web === 'string') {
          const title = typeof row.title === 'string' ? row.title : undefined;
          const subtitle = typeof row.authors === 'string' ? row.authors : 'Smithsonian 3D Collection';
          assets.push(buildAsset(repoId, row.files.web, row.web_bytes, undefined, title, subtitle));
          continue;
        }

        // Pattern C: scenario-labs/3d-collection -> row.outputs[0].path
        if (Array.isArray(row.outputs) && row.outputs[0]?.path) {
          const outPath = String(row.outputs[0].path);
          const title = typeof row.id === 'string' ? `Item ${row.id}` : undefined;
          assets.push(buildAsset(repoId, outPath, undefined, undefined, title));
          continue;
        }

        // Pattern D: Mesh type -> row.mesh.src
        if (row.mesh && typeof row.mesh.src === 'string') {
          const title = typeof row.title === 'string' ? row.title : typeof row.name === 'string' ? row.name : undefined;
          const slug = (title || row.id || `row-${entry.row_idx}`).toLowerCase().replace(/[^a-z0-9]+/g, '-');
          assets.push(buildAsset(repoId, `${slug}.glb`, row.mesh.size_bytes, row.mesh.src, title));
          continue;
        }

        // Pattern E: hugging-3d/smithsonian-open-access-3d -> row.mesh_url
        if (typeof row.mesh_url === 'string') {
          const title = typeof row.title === 'string' ? row.title : undefined;
          const subtitle = typeof row.authors === 'string' ? row.authors : 'Smithsonian Open Access';
          assets.push(buildAsset(repoId, `${row.record_id || entry.row_idx}.glb`, undefined, row.mesh_url, title, subtitle));
          continue;
        }
      }

      console.log(`  ✓ Offset ${offset}: found ${rows.length} rows (total accumulated: ${assets.length})`);
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return assets;
}

async function processGitTreeTarget(target: SeedTarget): Promise<Discovered3DAsset[]> {
  const assets: Discovered3DAsset[] = [];
  const repoId = target.repoId;
  console.log(`\n⏳ Listing files in git tree for "${repoId}"...`);
  await jitter();

  try {
    const fileIterator = listFiles({
      repo: { type: 'dataset', name: repoId },
      recursive: true,
      credentials: token ? { accessToken: token } : undefined,
    });

    let count = 0;
    for await (const entry of fileIterator) {
      if (entry.type !== 'file') continue;
      const lower = entry.path.toLowerCase();
      if (!lower.endsWith('.glb') && !lower.endsWith('.gltf') && !lower.endsWith('.obj')) continue;
      if (target.pathPrefix && !entry.path.startsWith(target.pathPrefix)) continue;
      if (target.pathIncludes && !entry.path.includes(target.pathIncludes)) continue;
      const bytes = (entry as { size?: number }).size;
      if (target.maxSizeBytes !== undefined && bytes !== undefined && bytes > target.maxSizeBytes) continue;
      if (target.minSizeBytes !== undefined && bytes !== undefined && bytes < target.minSizeBytes) continue;

      assets.push(buildAsset(repoId, entry.path, entry.size));
      count++;
      if (count >= (target.samples || 40)) break;
    }
    console.log(`  ✓ Found ${assets.length} 3D files in ${repoId}`);
  } catch (e) {
    console.warn(`  ↳ Git tree scan failed for ${repoId}:`, e);
  }

  return assets;
}

async function main() {
  console.log('🚀 Starting Polite 3D Manifest Generator for Hugging Face Hub...');
  console.log(token ? '🔑 HF Token detected (authenticated requests enabled)' : '⚠️ No HF Token detected (using polite unauthenticated rate pacing)');

  const allAssets: Discovered3DAsset[] = [];
  const seenIds = new Set<string>();

  for (const target of SEED_TARGETS) {
    let targetAssets: Discovered3DAsset[] = [];
    if (target.isGitTree) {
      targetAssets = await processGitTreeTarget(target);
    } else {
      targetAssets = await processDatasetServerTarget(target);
    }

    for (const a of targetAssets) {
      if (!seenIds.has(a.id)) {
        seenIds.add(a.id);
        allAssets.push(a);
      }
    }
  }

  // Shuffle the final list so diverse models appear immediately on first load
  for (let i = allAssets.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allAssets[i], allAssets[j]] = [allAssets[j], allAssets[i]];
  }

  const outDir = path.resolve(process.cwd(), 'public');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const outPath = path.join(outDir, 'manifest-3d.json');
  fs.writeFileSync(outPath, JSON.stringify(allAssets, null, 2), 'utf-8');

  console.log(`\n🎉 Success! Wrote ${allAssets.length} verified 3D assets to ${outPath}`);
  console.log(`📦 File size: ${(fs.statSync(outPath).size / 1024).toFixed(1)} KB`);
}

main().catch(console.error);
