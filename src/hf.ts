import { listDatasets, listFiles } from '@huggingface/hub';

export type RepoType = 'dataset' | 'model';
export type ModelFormat = 'glb' | 'gltf' | 'obj' | 'stl';

export interface Discovered3DAsset {
  id: string; // `${repoType}:${repoId}/${path}`
  repoId: string;
  repoType: RepoType;
  path: string;
  filename: string;
  title?: string;
  subtitle?: string;
  format: ModelFormat;
  size?: number;
  downloadUrl: string;
  resourceBaseUrl: string;
  repoUrl: string;
  treeUrl: string;
}

export interface DatasetMetadata {
  repoId: string;
  repoType: RepoType;
  author: string;
  description: string;
  likes: number;
  downloads: number;
  license?: string;
  tags: string[];
  lastModified?: string;
  repoUrl: string;
  discoveredFilesInRepo: number;
}

export interface SearchFilterOptions {
  query?: string;
  repoType: 'all' | 'dataset' | 'model';
  format: 'all' | 'glb' | 'obj' | 'stl';
}

export interface WallArtworkItem {
  id: string;
  imageUrl: string;
  title: string;
  subtitle: string;
  repoId: string;
}

/**
 * Returns true if two `Discovered3DAsset` entries represent the same 3D object
 * (matching ID, download URL, normalized title, or identical filename within a repo).
 */
export function isSame3DObject(a: Discovered3DAsset, b: Discovered3DAsset): boolean {
  if (a.id === b.id) return true;

  const cleanUrlA = a.downloadUrl.split('?')[0].trim().toLowerCase();
  const cleanUrlB = b.downloadUrl.split('?')[0].trim().toLowerCase();
  if (cleanUrlA && cleanUrlA === cleanUrlB) return true;

  const titleA = a.title?.trim().toLowerCase();
  const titleB = b.title?.trim().toLowerCase();
  if (titleA && titleB && titleA.length > 2 && titleA === titleB) {
    return true;
  }

  const fileA = a.filename.trim().toLowerCase();
  const fileB = b.filename.trim().toLowerCase();
  if (a.repoId === b.repoId && fileA === fileB) {
    return true;
  }

  return false;
}

export const WALL_ART_DATASETS: string[] = [
  'biglam/european_art',
  'Andris2067/Paintings_old',
  'royam0820/baya-paintings-01',
];

const SUPPORTED_EXTENSIONS: Record<string, ModelFormat> = {
  '.glb': 'glb',
  '.gltf': 'gltf',
  '.obj': 'obj',
  '.stl': 'stl',
};

const MAX_RANDOM_FILE_BYTES = 18 * 1024 * 1024;
const MIN_QUALITY_FILE_BYTES = 2 * 1024;
const MAX_POOL_SIZE = 2000;
const MAX_STAGED_SIZE = 200;
const MAX_METADATA_CACHE = 150;

const STATIC_DATASET_METADATA: Record<string, Omit<DatasetMetadata, 'discoveredFilesInRepo'>> = {
  'cfahlgren1/lowpoly-coder-4b-data': {
    repoId: 'cfahlgren1/lowpoly-coder-4b-data',
    repoType: 'dataset',
    author: 'cfahlgren1',
    description: 'Curated low-poly 3D GLB meshes paired with text prompts across animals, vehicles, props, and architecture.',
    likes: 48,
    downloads: 14200,
    license: 'apache-2.0',
    tags: ['3d', 'low-poly', 'glb', 'mesh-generation'],
    repoUrl: 'https://huggingface.co/datasets/cfahlgren1/lowpoly-coder-4b-data',
  },
  'codepawl/smithsonian-3d-glb': {
    repoId: 'codepawl/smithsonian-3d-glb',
    repoType: 'dataset',
    author: 'codepawl',
    description: 'Digitized 3D museum artifacts, fossils, skulls, and historical specimens from the Smithsonian Institution.',
    likes: 34,
    downloads: 8900,
    license: 'cc0-1.0',
    tags: ['3d', 'smithsonian', 'museum', 'photogrammetry'],
    repoUrl: 'https://huggingface.co/datasets/codepawl/smithsonian-3d-glb',
  },
  'hugging-3d/smithsonian-open-access-3d': {
    repoId: 'hugging-3d/smithsonian-open-access-3d',
    repoType: 'dataset',
    author: 'hugging-3d',
    description: 'High-resolution 3D scans from the Smithsonian Open Access collection spanning natural history and cultural heritage.',
    likes: 29,
    downloads: 6400,
    license: 'cc0-1.0',
    tags: ['3d', 'smithsonian', 'open-access', 'glb'],
    repoUrl: 'https://huggingface.co/datasets/hugging-3d/smithsonian-open-access-3d',
  },
  'YiboZhang2001/TexVerse-1K': {
    repoId: 'YiboZhang2001/TexVerse-1K',
    repoType: 'dataset',
    author: 'YiboZhang2001',
    description: 'TexVerse-1K subset of high-resolution 1K PBR-textured 3D GLB objects and characters.',
    likes: 65,
    downloads: 19500,
    license: 'cc-by-4.0',
    tags: ['3d', 'pbr', 'textured', 'texverse'],
    repoUrl: 'https://huggingface.co/datasets/YiboZhang2001/TexVerse-1K',
  },
  'YiboZhang2001/TexVerse': {
    repoId: 'YiboZhang2001/TexVerse',
    repoType: 'dataset',
    author: 'YiboZhang2001',
    description: 'TexVerse universe of high-resolution 2K PBR-textured 3D meshes designed for 3D generation and rendering.',
    likes: 210,
    downloads: 54000,
    license: 'cc-by-4.0',
    tags: ['3d', '2k-textures', 'pbr', 'texverse'],
    repoUrl: 'https://huggingface.co/datasets/YiboZhang2001/TexVerse',
  },
  'scenario-labs/3d-collection': {
    repoId: 'scenario-labs/3d-collection',
    repoType: 'dataset',
    author: 'scenario-labs',
    description: 'Curated collection of stylized and game-ready 3D GLB props, characters, and artifacts by Scenario Labs.',
    likes: 41,
    downloads: 11200,
    license: 'mit',
    tags: ['3d', 'game-assets', 'glb', 'stylized'],
    repoUrl: 'https://huggingface.co/datasets/scenario-labs/3d-collection',
  },
  'Optikan112/abo-furniture-glb': {
    repoId: 'Optikan112/abo-furniture-glb',
    repoType: 'dataset',
    author: 'Optikan112',
    description: 'Amazon Berkeley Objects (ABO) 3D furniture and household design catalog in physically-based GLB format.',
    likes: 22,
    downloads: 5300,
    license: 'cc-by-nc-4.0',
    tags: ['3d', 'furniture', 'industrial-design', 'pbr'],
    repoUrl: 'https://huggingface.co/datasets/Optikan112/abo-furniture-glb',
  },
  'codepawl/scanned-objects-glb': {
    repoId: 'codepawl/scanned-objects-glb',
    repoType: 'dataset',
    author: 'codepawl',
    description: 'Google Scanned Objects real-world household items, toys, and footwear converted to Draco/GLB meshes.',
    likes: 27,
    downloads: 7100,
    license: 'cc-by-4.0',
    tags: ['3d', 'scanned-objects', 'photogrammetry', 'glb'],
    repoUrl: 'https://huggingface.co/datasets/codepawl/scanned-objects-glb',
  },
  'TuTao/scandi-home-lowpoly': {
    repoId: 'TuTao/scandi-home-lowpoly',
    repoType: 'dataset',
    author: 'TuTao',
    description: 'Scandinavian low-poly home interior, furniture, and household appliance 3D GLB assets.',
    likes: 31,
    downloads: 6200,
    license: 'apache-2.0',
    tags: ['3d', 'low-poly', 'scandi', 'furniture', 'glb'],
    repoUrl: 'https://huggingface.co/datasets/TuTao/scandi-home-lowpoly',
  },
  'codepawl/google-poly-glb': {
    repoId: 'codepawl/google-poly-glb',
    repoType: 'dataset',
    author: 'codepawl',
    description: 'Curated archive of Google Poly low-poly 3D creations, characters, creatures, and scenes in Draco GLB format.',
    likes: 42,
    downloads: 9800,
    license: 'cc-by-3.0',
    tags: ['3d', 'google-poly', 'low-poly', 'glb'],
    repoUrl: 'https://huggingface.co/datasets/codepawl/google-poly-glb',
  },
  'cfahlgren1/3d-arena': {
    repoId: 'cfahlgren1/3d-arena',
    repoType: 'dataset',
    author: 'cfahlgren1',
    description: '3D Arena benchmark outputs comparing image-to-3D and text-to-3D generators (Hunyuan3D-2.1, Hi3DGen, Hi3D-1.0, 3DTopia-XL).',
    likes: 38,
    downloads: 8400,
    license: 'mit',
    tags: ['3d', '3d-arena', 'image-to-3d', 'glb'],
    repoUrl: 'https://huggingface.co/datasets/cfahlgren1/3d-arena',
  },
  'cfahlgren1/3d-collection': {
    repoId: 'cfahlgren1/3d-collection',
    repoType: 'dataset',
    author: 'cfahlgren1',
    description: 'Curated 3D GLB collection of text-to-3D and image-to-3D assets with generator metadata and prompts.',
    likes: 44,
    downloads: 11800,
    license: 'cc-by-4.0',
    tags: ['3d', 'glb', 'image-to-3d', 'text-to-3d'],
    repoUrl: 'https://huggingface.co/datasets/cfahlgren1/3d-collection',
  },
};

// Verified seed list of Hugging Face 3D datasets (put user favorites first!)
export const SEED_DATASETS: string[] = [
  'cfahlgren1/lowpoly-coder-4b-data',
  'codepawl/smithsonian-3d-glb',
  'codepawl/google-poly-glb',
  'cfahlgren1/3d-arena',
  'cfahlgren1/3d-collection',
  'YiboZhang2001/TexVerse-1K',
  'TuTao/scandi-home-lowpoly',
  'hugging-3d/smithsonian-open-access-3d',
  'ccore/ThreeJS-3D-GLB-Assets',
  'scenario-labs/3d-collection',
  'allenai/objaverse',
  'Optikan112/abo-furniture-glb',
  'codepawl/menagerie-glb',
  'AuWang/PartNeXt_mesh',
  'Linzhan/Objaverse-XL-Rigged-Animated',
  'codepawl/ct-anatomy-glb',
  'codepawl/scanned-objects-glb',
  'wchen99998/american-residential-housing-hunyuan3d-glb',
];

// Known non-default Datasets Server config names so we don't need an extra /splits round-trip
const DATASET_SERVER_CONFIG_HINTS: Record<string, { config: string; split: string }> = {
  'cfahlgren1/lowpoly-coder-4b-data': { config: 'stage1', split: 'train' },
  'ccore/ThreeJS-3D-GLB-Assets': { config: 'meshes', split: 'train' },
  'hugging-3d/smithsonian-open-access-3d': { config: 'metadata', split: 'train' },
  'codepawl/smithsonian-3d-glb': { config: 'default', split: 'train' },
  'scenario-labs/3d-collection': { config: 'default', split: 'train' },
};

// Search queries used by the background crawler to discover new 3D datasets on Hugging Face
const BACKGROUND_3D_QUERIES = [
  'glb',
  'gltf',
  '3d mesh',
  'objaverse',
  'trellis',
  'hunyuan3d',
  'triposr',
  'instantmesh',
  'shap-e',
  'stl',
  'lowpoly',
  'furniture 3d',
  'cad 3d',
  'robot mesh',
  'character glb',
  'scan 3d',
];

export function getModelFormat(filePathOrUrl: string): ModelFormat | null {
  const clean = filePathOrUrl.split('?')[0].toLowerCase();
  for (const [ext, fmt] of Object.entries(SUPPORTED_EXTENSIONS)) {
    if (clean.endsWith(ext)) {
      return fmt;
    }
  }
  return null;
}

export function matchesFormatFilter(
  format: ModelFormat,
  filter: SearchFilterOptions['format']
): boolean {
  if (filter === 'all') return true;
  if (filter === 'glb') return format === 'glb' || format === 'gltf';
  return format === filter;
}

export function buildAssetUrls(
  repoType: RepoType,
  repoId: string,
  filePath: string,
  externalDownloadUrl?: string
): {
  downloadUrl: string;
  resourceBaseUrl: string;
  repoUrl: string;
  treeUrl: string;
} {
  const prefix = repoType === 'dataset' ? 'datasets/' : '';
  const repoUrl = `https://huggingface.co/${prefix}${repoId}`;
  const treeUrl = `https://huggingface.co/${prefix}${repoId}/tree/main`;

  if (externalDownloadUrl) {
    const cleanUrl = externalDownloadUrl.split('?')[0];
    const lastSlash = cleanUrl.lastIndexOf('/');
    const resourceBaseUrl = lastSlash >= 0 ? cleanUrl.slice(0, lastSlash + 1) : '';
    return {
      downloadUrl: externalDownloadUrl,
      resourceBaseUrl,
      repoUrl,
      treeUrl,
    };
  }

  const encodedPath = filePath
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');

  const downloadUrl = `https://huggingface.co/${prefix}${repoId}/resolve/main/${encodedPath}`;
  const lastSlash = encodedPath.lastIndexOf('/');
  const dirPath = lastSlash >= 0 ? encodedPath.slice(0, lastSlash + 1) : '';
  const resourceBaseUrl = `https://huggingface.co/${prefix}${repoId}/resolve/main/${dirPath}`;

  return { downloadUrl, resourceBaseUrl, repoUrl, treeUrl };
}

function cleanDescription(raw?: string): string {
  if (!raw) return 'No description provided on Hugging Face dataset card.';
  const cleaned = raw
    .replace(/\r/g, '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return 'No description provided on Hugging Face dataset card.';
  return cleaned.length > 260 ? `${cleaned.slice(0, 257)}...` : cleaned;
}

export class HF3DDiscoveryService {
  private pool: Map<string, Discovered3DAsset> = new Map();
  private stagedPool: Map<string, Discovered3DAsset> = new Map();
  private shuffledOrder: string[] = [];
  private scannedRepos: Set<string> = new Set();
  private usedQueries: Set<string> = new Set();
  private metadataCache: Map<string, DatasetMetadata> = new Map();

  public shufflePool(): void {
    this.enforcePoolCap();
    this.shuffledOrder = Array.from(this.pool.keys());
    shuffleInPlace(this.shuffledOrder);
  }

  public removeAsset(id: string): void {
    const existing = this.pool.get(id) || this.stagedPool.get(id);
    this.pool.delete(id);
    this.stagedPool.delete(id);
    this.shuffledOrder = this.shuffledOrder.filter((k) => k !== id);

    // If removing a broken asset leaves its dataset with <= 1 remaining item, prune that dataset too
    if (existing && this.getRepoFileCount(existing.repoId) <= 1) {
      for (const [k, v] of this.pool.entries()) {
        if (v.repoId === existing.repoId) this.pool.delete(k);
      }
      for (const [k, v] of this.stagedPool.entries()) {
        if (v.repoId === existing.repoId) this.stagedPool.delete(k);
      }
      this.shuffledOrder = Array.from(this.pool.keys());
      shuffleInPlace(this.shuffledOrder);
    }
  }

  private enforcePoolCap(): void {
    while (this.pool.size > MAX_POOL_SIZE) {
      const oldestKey = this.pool.keys().next().value;
      if (!oldestKey) break;
      this.pool.delete(oldestKey);
    }
    while (this.stagedPool.size > MAX_STAGED_SIZE) {
      const oldestKey = this.stagedPool.keys().next().value;
      if (!oldestKey) break;
      this.stagedPool.delete(oldestKey);
    }
    while (this.metadataCache.size > MAX_METADATA_CACHE) {
      const oldestKey = this.metadataCache.keys().next().value;
      if (!oldestKey) break;
      this.metadataCache.delete(oldestKey);
    }
  }

  public getStagedCount(): number {
    return this.stagedPool.size;
  }

  public getActiveDatasetCount(): number {
    const repos = new Set<string>();
    for (const item of this.pool.values()) {
      repos.add(item.repoId);
    }
    return repos.size;
  }

  public getRepoFileCount(repoId: string): number {
    let count = 0;
    for (const item of this.pool.values()) {
      if (item.repoId === repoId) count++;
    }
    for (const item of this.stagedPool.values()) {
      if (item.repoId === repoId) count++;
    }
    return count;
  }

  /**
   * Merges background-discovered 3D assets into the active pool and reshuffles.
   */
  public mergeStagedIntoPool(): number {
    const added = this.stagedPool.size;
    for (const [id, asset] of this.stagedPool.entries()) {
      this.pool.set(id, asset);
    }
    this.stagedPool.clear();
    this.shufflePool();
    return added;
  }

  public getPool(filter?: SearchFilterOptions): Discovered3DAsset[] {
    if (this.shuffledOrder.length !== this.pool.size) {
      this.shufflePool();
    }
    const all = this.shuffledOrder
      .map((id) => this.pool.get(id))
      .filter((item): item is Discovered3DAsset => item !== undefined);
    if (!filter) return all;

    const q = (filter.query ?? '').trim().toLowerCase();
    return all.filter((item) => {
      if (filter.repoType !== 'all' && item.repoType !== filter.repoType) {
        return false;
      }
      if (!matchesFormatFilter(item.format, filter.format)) {
        return false;
      }
      if (q) {
        const haystack = `${item.repoId} ${item.path} ${item.title ?? ''}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });
  }

  private createAsset(
    repoType: RepoType,
    repoId: string,
    filePath: string,
    size?: number,
    externalDownloadUrl?: string,
    title?: string,
    subtitle?: string
  ): Discovered3DAsset | null {
    const format = getModelFormat(externalDownloadUrl || filePath);
    if (!format) return null;
    if (size !== undefined && (size > MAX_RANDOM_FILE_BYTES || size < MIN_QUALITY_FILE_BYTES)) {
      return null;
    }

    const urls = buildAssetUrls(repoType, repoId, filePath, externalDownloadUrl);
    const filename = filePath.split('/').pop() || filePath;
    return {
      id: `${repoType}:${repoId}/${filePath}`,
      repoId,
      repoType,
      path: filePath,
      filename,
      title,
      subtitle,
      format,
      size,
      ...urls,
    };
  }

  /**
   * Queries the Hugging Face Datasets Server (`/rows` or `/first-rows`) to extract 3D models from
   * Parquet-backed or structured 3D datasets (e.g., codepawl/smithsonian-3d-glb,
   * cfahlgren1/lowpoly-coder-4b-data, ccore/ThreeJS-3D-GLB-Assets, scenario-labs/3d-collection).
   *
   * Crucially, we pick a random row offset (or query /rows with offset) across all thousands of
   * rows in the dataset so we never keep showing only the first 100 items over and over!
   */
  private async scanDatasetServerRows(
    repoId: string,
    lowPriority = false
  ): Promise<Discovered3DAsset[]> {
    const extracted: Discovered3DAsset[] = [];
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4500);
      const fetchOpts: RequestInit = {
        signal: controller.signal,
        ...(lowPriority ? { priority: 'low' as RequestPriority } : {}),
      };

      let config = DATASET_SERVER_CONFIG_HINTS[repoId]?.config ?? 'default';
      let split = DATASET_SERVER_CONFIG_HINTS[repoId]?.split ?? 'train';

      // 1. Initial probe: fetch 1 row from /rows to discover total number of rows and check config
      let probeUrl = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
        repoId
      )}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=0&length=1`;

      let res = await fetch(probeUrl, fetchOpts);

      // If default/train wasn't found, check /splits to discover the dataset's config and split name
      if (!res.ok && !DATASET_SERVER_CONFIG_HINTS[repoId]) {
        const splitsRes = await fetch(
          `https://datasets-server.huggingface.co/splits?dataset=${encodeURIComponent(repoId)}`,
          fetchOpts
        );
        if (splitsRes.ok) {
          const splitsJson = await splitsRes.json();
          const firstSplit = splitsJson.splits?.[0];
          if (firstSplit) {
            config = firstSplit.config;
            split = firstSplit.split;
            probeUrl = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
              repoId
            )}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=0&length=1`;
            res = await fetch(probeUrl, fetchOpts);
          }
        }
      }

      if (!res.ok) {
        // Fallback to /first-rows in case /rows is disabled
        res = await fetch(
          `https://datasets-server.huggingface.co/first-rows?dataset=${encodeURIComponent(
            repoId
          )}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}`,
          fetchOpts
        );
      }

      clearTimeout(timeoutId);
      if (!res.ok) return [];

      const probeData = await res.json();
      const totalRows =
        typeof probeData.num_rows_total === 'number'
          ? probeData.num_rows_total
          : Array.isArray(probeData.rows)
          ? probeData.rows.length
          : 0;

      if (totalRows <= 1) {
        return [];
      }

      // 2. Fetch a batch of up to 100 rows from a RANDOM offset across the full dataset!
      const pageSize = Math.min(100, totalRows);
      const maxOffset = Math.max(0, totalRows - pageSize);
      const randomOffset = Math.floor(Math.random() * (maxOffset + 1));

      let rows: Array<{ row_idx: number; row: Record<string, any> }> = [];

      // If the random offset is 0 and we already have full rows in probeData, reuse them
      if (randomOffset === 0 && Array.isArray(probeData.rows) && probeData.rows.length >= 20) {
        rows = probeData.rows;
      } else {
        const fetchBatchController = new AbortController();
        const batchTimeoutId = setTimeout(() => fetchBatchController.abort(), 4500);
        try {
          const rowsUrl = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
            repoId
          )}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(
            split
          )}&offset=${randomOffset}&length=${pageSize}`;

          const rowsRes = await fetch(rowsUrl, {
            signal: fetchBatchController.signal,
            ...(lowPriority ? { priority: 'low' as RequestPriority } : {}),
          });
          if (rowsRes.ok) {
            const batchData = await rowsRes.json();
            if (Array.isArray(batchData.rows) && batchData.rows.length > 0) {
              rows = batchData.rows;
            }
          }
        } catch {
          // If fetching random offset failed or timed out, fall back to probeData rows
        } finally {
          clearTimeout(batchTimeoutId);
        }

        if (rows.length === 0 && Array.isArray(probeData.rows)) {
          rows = probeData.rows;
        }
      }

      if (rows.length <= 1) {
        return [];
      }

      for (const entry of rows) {
        const row = entry.row;
        if (!row || typeof row !== 'object') continue;

        const title: string | undefined =
          typeof row.title === 'string'
            ? row.title
            : typeof row.name === 'string'
            ? row.name
            : undefined;
        const subtitle: string | undefined =
          typeof row.authors === 'string'
            ? row.authors
            : typeof row.data_source === 'string'
            ? row.data_source
            : typeof row.domain === 'string'
            ? `Category: ${row.domain}`
            : typeof row.model === 'string'
            ? `Generator: ${row.model}`
            : undefined;

        // Pattern 1: codepawl/smithsonian-3d-glb -> row.files.web ("objects/.../foo.glb")
        if (row.files && typeof row.files.web === 'string') {
          const asset = this.createAsset(
            'dataset',
            repoId,
            row.files.web,
            typeof row.web_bytes === 'number' ? row.web_bytes : undefined,
            undefined,
            title,
            subtitle
          );
          if (asset) {
            extracted.push(asset);
            continue;
          }
        }

        // Pattern 2: scenario-labs/3d-collection -> row.outputs[0].path ("assets/.../output.glb")
        if (Array.isArray(row.outputs) && row.outputs[0]?.path) {
          const outPath = String(row.outputs[0].path);
          const displayPath =
            typeof row.id === 'string' ? `${row.id}/${outPath.split('/').pop()}` : outPath;
          const url = `https://huggingface.co/datasets/${repoId}/resolve/main/${outPath}`;
          const asset = this.createAsset(
            'dataset',
            repoId,
            displayPath,
            undefined,
            url,
            title || row.id,
            subtitle
          );
          if (asset) {
            extracted.push(asset);
            continue;
          }
        }

        // Pattern 3: hugging-3d/smithsonian-open-access-3d -> row.mesh_url (direct .glb URL on 3d-api.si.edu)
        if (typeof row.mesh_url === 'string' && getModelFormat(row.mesh_url)) {
          // Prefer Draco compressed GLB from available_files_json if present for faster downloads
          let bestUrl = row.mesh_url;
          let sizeBytes = typeof row.size_bytes === 'number' ? row.size_bytes : undefined;
          if (typeof row.available_files_json === 'string') {
            try {
              const avail = JSON.parse(row.available_files_json);
              const dracoEntry = Array.isArray(avail)
                ? avail.find(
                    (a: any) =>
                      typeof a.uri === 'string' && a.uri.toLowerCase().endsWith('_draco.glb')
                  )
                : null;
              if (dracoEntry?.uri) {
                bestUrl = dracoEntry.uri;
                sizeBytes = undefined;
              }
            } catch {
              // Keep row.mesh_url
            }
          }
          const urlFilename = bestUrl.split('/').pop() || `${row.record_id || entry.row_idx}.glb`;
          const asset = this.createAsset(
            'dataset',
            repoId,
            urlFilename,
            sizeBytes,
            bestUrl,
            title,
            subtitle
          );
          if (asset) {
            extracted.push(asset);
            continue;
          }
        }

        // Pattern 4: HF Datasets Server `_type: "Mesh"` column
        // e.g., ccore/ThreeJS-3D-GLB-Assets -> row.mesh.src
        // e.g., cfahlgren1/lowpoly-coder-4b-data -> row.glb.src
        const meshObj = row.mesh || row.glb;
        if (meshObj && typeof meshObj.src === 'string' && getModelFormat(meshObj.src)) {
          let customTitle = title;
          let customSubtitle = subtitle;
          if (repoId === 'cfahlgren1/lowpoly-coder-4b-data') {
            const rawPrompt = typeof row.prompt === 'string' ? row.prompt.trim() : '';
            if (rawPrompt) {
              customTitle = rawPrompt.replace(/^(I need a |Build a |Create a |Make a )/i, '');
              customTitle = customTitle.charAt(0).toUpperCase() + customTitle.slice(1).replace(/\.$/, '');
            } else if (typeof row.id === 'string') {
              customTitle = `Low-Poly Object (${row.id})`;
            }
            if (typeof row.category === 'string') {
              customSubtitle = `Category: ${row.category} · cfahlgren1/lowpoly-coder-4b-data`;
            }
          }

          const slug = (customTitle || title || row.id || `row-${entry.row_idx}`)
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-');
          const fakePath = `${slug}.glb`;
          const asset = this.createAsset(
            'dataset',
            repoId,
            fakePath,
            typeof meshObj.size_bytes === 'number' ? meshObj.size_bytes : undefined,
            meshObj.src,
            customTitle || title,
            customSubtitle || subtitle
          );
          if (asset) {
            extracted.push(asset);
            continue;
          }
        }
      }
    } catch {
      // Ignore Datasets Server errors and fall back to git tree scanning
    }

    // Never use a dataset that only yielded 1 valid 3D row
    return extracted.length > 1 ? extracted : [];
  }

  /**
   * Scans a single Hugging Face repository using both Datasets Server rows (for Parquet/structured
   * 3D datasets) and @huggingface/hub listFiles (for direct git tree 3D files).
   */
  public async scanRepoFor3DFiles(
    repoType: RepoType,
    repoId: string,
    maxFilesPerRepo = 14,
    stageOnly = false
  ): Promise<Discovered3DAsset[]> {
    const key = `${repoType}:${repoId}`;
    this.scannedRepos.add(key);

    const candidates: Discovered3DAsset[] = [];

    // 1. If this is a dataset, try extracting structured 3D rows from Hugging Face Datasets Server first
    if (repoType === 'dataset') {
      const fromRows = await this.scanDatasetServerRows(repoId, stageOnly);
      if (fromRows.length > 0) {
        candidates.push(...fromRows);
      }
    }

    // 2. If Datasets Server didn't yield files, walk the repository tree via @huggingface/hub listFiles
    if (candidates.length === 0) {
      const rawEntries: Array<{ path: string; size?: number }> = [];
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4500);

        const fileIterator = listFiles({
          repo: { type: repoType, name: repoId },
          recursive: true,
          fetch: (input, init) =>
            fetch(input, {
              ...init,
              signal: controller.signal,
              ...(stageOnly ? { priority: 'low' as RequestPriority } : {}),
            }),
        });

        let inspectedCount = 0;
        for await (const entry of fileIterator) {
          inspectedCount++;
          if (inspectedCount > 500 || rawEntries.length >= 140) {
            break;
          }
          if (entry.type !== 'file') continue;
          if (!getModelFormat(entry.path)) continue;
          rawEntries.push({ path: entry.path, size: entry.size });
        }
        clearTimeout(timeoutId);
      } catch {
        // Ignore gated, private, or slow repos
      }

      for (const item of rawEntries) {
        // Human-friendly title formatting for lowpoly-coder-4b-data:
        // "glb/stage1/train/stage1-train-01290.glb" -> "Stage 1 Train #01290"
        let customTitle: string | undefined;
        let customSubtitle: string | undefined;
        if (repoId === 'cfahlgren1/lowpoly-coder-4b-data') {
          const match = item.path.match(/stage(\d+)-(train|valid)-(\d+)\.glb/i);
          if (match) {
            const stageNum = match[1];
            const splitName = match[2] === 'train' ? 'Training' : 'Validation';
            const num = match[3];
            customTitle = `Low-Poly Object #${num}`;
            customSubtitle = `Stage ${stageNum} ${splitName} · cfahlgren1/lowpoly-coder-4b-data`;
          }
        }

        const asset = this.createAsset(
          repoType,
          repoId,
          item.path,
          item.size,
          undefined,
          customTitle,
          customSubtitle
        );
        if (asset) {
          candidates.push(asset);
        }
      }
    }

    // Deduplicate candidates within the same scan so identical titles/URLs/filenames aren't added twice
    const uniqueCandidates: Discovered3DAsset[] = [];
    for (const c of candidates) {
      if (!uniqueCandidates.some((existing) => isSame3DObject(existing, c))) {
        uniqueCandidates.push(c);
      }
    }

    // Filter out single-row / single-model datasets (must have at least 2 valid 3D models)
    if (uniqueCandidates.length <= 1) {
      return [];
    }

    shuffleInPlace(uniqueCandidates);
    const chosen = uniqueCandidates.slice(0, maxFilesPerRepo);
    for (const asset of chosen) {
      if (stageOnly && !this.pool.has(asset.id)) {
        this.stagedPool.set(asset.id, asset);
      } else {
        this.pool.set(asset.id, asset);
      }
    }
    this.enforcePoolCap();

    return chosen;
  }

  /**
   * Fast initial startup:
   * 1. First loads the pre-baked static manifest (`manifest-3d.json`) in ~50ms (zero network calls to HF API, zero 429s).
   * 2. Shuffles and populates the gallery immediately so visitors never wait or get rate limited.
   * 3. Opportunistically samples live seed datasets only as a background enhancement if needed.
   */
  public async initFromSeedDatasets(
    initialSeedCount = 5,
    onProgress?: (status: string) => void
  ): Promise<Discovered3DAsset[]> {
    onProgress?.('Loading curated 3D exhibits…');

    // 1. Instantly hydrate from the local static manifest
    try {
      const manifestRes = await fetch('/manifest-3d.json');
      if (manifestRes.ok) {
        const manifestAssets: Discovered3DAsset[] = await manifestRes.json();
        if (Array.isArray(manifestAssets) && manifestAssets.length > 0) {
          shuffleInPlace(manifestAssets);
          for (const asset of manifestAssets) {
            if (
              asset.size !== undefined &&
              (asset.size > MAX_RANDOM_FILE_BYTES || asset.size < MIN_QUALITY_FILE_BYTES)
            ) {
              continue;
            }
            this.pool.set(asset.id, asset);
            this.scannedRepos.add(`dataset:${asset.repoId}`);
          }
          this.enforcePoolCap();
          this.shufflePool();
          return this.getPool();
        }
      }
    } catch (e) {
      console.warn('[Discovery] Static manifest-3d.json load skipped:', e);
    }

    // 2. Fallback: if manifest-3d.json is absent, query seed datasets
    const favorites = SEED_DATASETS.slice(0, 4).filter(
      (repoId) => !this.scannedRepos.has(`dataset:${repoId}`)
    );
    const others = SEED_DATASETS.slice(4).filter(
      (repoId) => !this.scannedRepos.has(`dataset:${repoId}`)
    );
    shuffleInPlace(favorites);
    shuffleInPlace(others);

    const chosen = [...favorites.slice(0, 3), ...others.slice(0, initialSeedCount - 3)];

    onProgress?.(
      `Sampling ${chosen.length} seed 3D datasets (${chosen
        .map((id) => id.split('/')[1] || id)
        .join(', ')})...`
    );

    const results = await Promise.all(
      chosen.map((repoId) => this.scanRepoFor3DFiles('dataset', repoId, 14, false))
    );
    const discovered = results.flat();
    this.shufflePool();
    return discovered;
  }

  /**
   * Non-blocking background discovery step:
   * Scans remaining seed datasets AND searches Hugging Face Hub for new 3D datasets,
   * staging newly discovered 3D files for pool refresh.
   */
  public async discoverMoreInBackground(): Promise<{
    newlyStaged: number;
    totalStaged: number;
    newDatasetNames: string[];
  }> {
    const candidateIds: string[] = [];

    // 1. Include up to 2 unscanned seed datasets if any remain
    const remainingSeeds = SEED_DATASETS.filter(
      (repoId) => !this.scannedRepos.has(`dataset:${repoId}`)
    );
    shuffleInPlace(remainingSeeds);
    candidateIds.push(...remainingSeeds.slice(0, 2));

    // 2. Search Hugging Face Hub for new 3D datasets via @huggingface/hub listDatasets
    const unusedQueries = BACKGROUND_3D_QUERIES.filter((q) => !this.usedQueries.has(q));
    const queryPool = unusedQueries.length > 0 ? unusedQueries : BACKGROUND_3D_QUERIES;
    const query = queryPool[Math.floor(Math.random() * queryPool.length)];
    this.usedQueries.add(query);

    try {
      const searchHits: string[] = [];
      let count = 0;
      for await (const ds of listDatasets({
        search: { query },
        sort: 'likes',
        limit: 25,
        fetch: (input, init) =>
          fetch(input, { ...init, priority: 'low' as RequestPriority }),
      })) {
        count++;
        if (count > 25) break;
        if (ds.private || ds.gated) continue;
        if (!this.scannedRepos.has(`dataset:${ds.name}`) && !candidateIds.includes(ds.name)) {
          searchHits.push(ds.name);
        }
      }
      shuffleInPlace(searchHits);
      candidateIds.push(...searchHits.slice(0, 3));
    } catch {
      // Ignore background search errors
    }

    let newlyStaged = 0;
    const newDatasetNames: string[] = [];

    const results = await Promise.all(
      candidateIds.map(async (repoId) => {
        const assets = await this.scanRepoFor3DFiles('dataset', repoId, 12, true);
        if (assets.length > 0) {
          newDatasetNames.push(repoId);
        }
        return assets;
      })
    );

    for (const list of results) {
      newlyStaged += list.length;
    }

    return {
      newlyStaged,
      totalStaged: this.stagedPool.size,
      newDatasetNames,
    };
  }

  /**
   * Explicit search used by the Debug Search Drawer
   */
  public async discoverFromHub(
    options: SearchFilterOptions,
    onProgress?: (status: string) => void
  ): Promise<Discovered3DAsset[]> {
    const customQuery = (options.query ?? '').trim() || 'glb';
    onProgress?.(`Searching Hugging Face datasets for "${customQuery}"...`);

    const candidateRepos: string[] = [];
    // If the user typed a direct repoId (like "codepawl/smithsonian-3d-glb"), scan it directly
    if (customQuery.includes('/')) {
      candidateRepos.push(customQuery.replace(/^datasets\//, ''));
    }

    try {
      let count = 0;
      for await (const ds of listDatasets({
        search: { query: customQuery },
        sort: 'likes',
        limit: 25,
      })) {
        count++;
        if (count > 25) break;
        if (ds.private || ds.gated) continue;
        if (!this.scannedRepos.has(`dataset:${ds.name}`) && !candidateRepos.includes(ds.name)) {
          candidateRepos.push(ds.name);
        }
      }
    } catch {
      // Ignore errors
    }

    const slice = candidateRepos.slice(0, 6);
    const results = await Promise.all(
      slice.map((repoId) => this.scanRepoFor3DFiles('dataset', repoId, 12, false))
    );
    const newlyDiscovered = results.flat();
    this.shufflePool();
    return newlyDiscovered;
  }

  /**
   * Fetches and caches dataset metadata (description, likes, downloads, tags, license).
   * Uses pre-populated static metadata for all manifest/seed datasets so zero live
   * Hugging Face API calls (and zero 429 rate-limit console errors) occur during browsing.
   */
  public async fetchDatasetMetadata(
    repoId: string,
    repoType: RepoType = 'dataset'
  ): Promise<DatasetMetadata> {
    const cached = this.metadataCache.get(`${repoType}:${repoId}`);
    if (cached) {
      cached.discoveredFilesInRepo = this.getRepoFileCount(repoId);
      return cached;
    }

    const staticMeta = STATIC_DATASET_METADATA[repoId];
    if (staticMeta) {
      const fullMeta: DatasetMetadata = {
        ...staticMeta,
        discoveredFilesInRepo: this.getRepoFileCount(repoId),
      };
      this.metadataCache.set(`${repoType}:${repoId}`, fullMeta);
      return fullMeta;
    }

    const endpoint =
      repoType === 'dataset'
        ? `https://huggingface.co/api/datasets/${repoId}`
        : `https://huggingface.co/api/models/${repoId}`;
    const prefix = repoType === 'dataset' ? 'datasets/' : '';
    const repoUrl = `https://huggingface.co/${prefix}${repoId}`;

    try {
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const rawTags: string[] = Array.isArray(data.tags) ? data.tags : [];
      const licenseFromTag = rawTags
        .find((t) => t.startsWith('license:'))
        ?.replace('license:', '');
      const license = data.cardData?.license || licenseFromTag || undefined;

      const displayTags = rawTags
        .filter(
          (t) =>
            !t.startsWith('license:') &&
            !t.startsWith('region:') &&
            !t.startsWith('library:')
        )
        .map((t) => t.replace('modality:', '').replace('size_categories:', 'size: '))
        .slice(0, 6);

      const meta: DatasetMetadata = {
        repoId,
        repoType,
        author: data.author || repoId.split('/')[0] || 'Unknown',
        description: cleanDescription(data.description),
        likes: typeof data.likes === 'number' ? data.likes : 0,
        downloads: typeof data.downloads === 'number' ? data.downloads : 0,
        license: Array.isArray(license) ? license.join(', ') : license,
        tags: displayTags,
        lastModified: data.lastModified,
        repoUrl,
        discoveredFilesInRepo: this.getRepoFileCount(repoId),
      };

      this.metadataCache.set(`${repoType}:${repoId}`, meta);
      return meta;
    } catch {
      return {
        repoId,
        repoType,
        author: repoId.split('/')[0] || 'Unknown',
        description: 'Hugging Face 3D repository.',
        likes: 0,
        downloads: 0,
        tags: ['3d'],
        repoUrl,
        discoveredFilesInRepo: this.getRepoFileCount(repoId),
      };
    }
  }

  public pickRandom(
    options: SearchFilterOptions,
    exclude?:
      | Discovered3DAsset
      | string
      | Array<Discovered3DAsset | string | null | undefined>,
    preferRepoId?: string,
    excludeRepoId?: string
  ): Discovered3DAsset | null {
    let candidates = this.getPool(options);

    const excludeList = Array.isArray(exclude) ? exclude : exclude ? [exclude] : [];
    const excludedIds = new Set<string>();
    const excludedAssets: Discovered3DAsset[] = [];

    for (const item of excludeList) {
      if (!item) continue;
      if (typeof item === 'string') {
        excludedIds.add(item);
      } else {
        excludedIds.add(item.id);
        excludedAssets.push(item);
      }
    }

    const isExcluded = (c: Discovered3DAsset): boolean => {
      if (excludedIds.has(c.id)) return true;
      for (const ex of excludedAssets) {
        if (isSame3DObject(c, ex)) return true;
      }
      return false;
    };

    // Always block any object already in the active set of 4 pedestals (or excluded list)
    const nonDuplicateCandidates = candidates.filter((c) => !isExcluded(c));
    if (nonDuplicateCandidates.length > 0) {
      candidates = nonDuplicateCandidates;
    }

    if (preferRepoId) {
      const sameRepo = candidates.filter((c) => c.repoId === preferRepoId);
      if (sameRepo.length > 0) {
        candidates = sameRepo;
      }
    } else if (excludeRepoId) {
      const differentRepo = candidates.filter((c) => c.repoId !== excludeRepoId);
      if (differentRepo.length > 0) {
        candidates = differentRepo;
      }
    }
    if (candidates.length === 0) return null;

    // Precompute repo file counts in a single O(n) pass over the pool + stagedPool
    const repoCounts = new Map<string, number>();
    for (const item of this.pool.values()) {
      repoCounts.set(item.repoId, (repoCounts.get(item.repoId) ?? 0) + 1);
    }
    for (const item of this.stagedPool.values()) {
      repoCounts.set(item.repoId, (repoCounts.get(item.repoId) ?? 0) + 1);
    }

    // Pick a random repository first so large datasets don't dominate small ones,
    // excluding any repository that has only 1 row/item in the pool.
    const byRepo = new Map<string, Discovered3DAsset[]>();
    for (const c of candidates) {
      if ((repoCounts.get(c.repoId) ?? 0) <= 1) continue;
      const list = byRepo.get(c.repoId) ?? [];
      list.push(c);
      byRepo.set(c.repoId, list);
    }
    const repos = Array.from(byRepo.values());
    if (repos.length === 0) return null;
    const chosenRepoAssets = repos[Math.floor(Math.random() * repos.length)];
    return chosenRepoAssets[Math.floor(Math.random() * chosenRepoAssets.length)];
  }

  /**
   * Fetches open-source paintings from curated Hugging Face datasets:
   * 1. Loads instantly from `public/manifest-art.json` (zero network calls to HF API, zero 429s).
   * 2. Falls back to direct live listing if manifest is missing.
   */
  public async fetchWallArtworks(): Promise<WallArtworkItem[]> {
    // 1. Try static manifest first
    try {
      const res = await fetch('/manifest-art.json');
      if (res.ok) {
        const items: WallArtworkItem[] = await res.json();
        if (Array.isArray(items) && items.length > 0) {
          shuffleInPlace(items);
          return items;
        }
      }
    } catch {
      // Fall through to live queries
    }

    const items: WallArtworkItem[] = [];

    const toCorsSafeAssetUrl = (rawUrl: string): string => {
      if (
        typeof window !== 'undefined' &&
        (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') &&
        rawUrl.startsWith('https://datasets-server.huggingface.co/assets/')
      ) {
        return rawUrl.replace(
          'https://datasets-server.huggingface.co/assets/',
          '/hf-dataset-assets/'
        );
      }
      return `https://wsrv.nl/?url=${encodeURIComponent(rawUrl)}&w=800&output=jpg`;
    };

    // 1. Direct HF repo painting files (guaranteed Access-Control-Allow-Origin: *)
    const directPaintingRepos: Array<{ repoId: string; collectionTitle: string; artist: string }> = [
      {
        repoId: 'Andris2067/Paintings_old',
        collectionTitle: 'Classical Oil Study',
        artist: 'European Masters Collection',
      },
      {
        repoId: 'royam0820/baya-paintings-01',
        collectionTitle: 'Baya Gouache Composition',
        artist: 'Baya Mahieddine Archive',
      },
    ];

    await Promise.all([
      ...directPaintingRepos.map(async ({ repoId, collectionTitle, artist }) => {
        try {
          let count = 0;
          for await (const f of listFiles({
            repo: { type: 'dataset', name: repoId },
            recursive: true,
          })) {
            if (f.type !== 'file' || !f.path.match(/\.(jpg|jpeg|png|webp)$/i)) continue;
            count++;
            if (count > 24) break;
            const encodedPath = f.path.split('/').map(encodeURIComponent).join('/');
            const imageUrl = `https://huggingface.co/datasets/${repoId}/resolve/main/${encodedPath}`;
            items.push({
              id: `${repoId}:${f.path}`,
              imageUrl,
              title: `${collectionTitle} #${count}`,
              subtitle: `${artist} · ${repoId}`,
              repoId,
            });
          }
        } catch {
          // Ignore individual dataset errors
        }
      }),
      // 2. European Art Archive (`biglam/european_art`)
      (async () => {
        const repoId = 'biglam/european_art';
        try {
          const url = `https://datasets-server.huggingface.co/first-rows?dataset=${encodeURIComponent(repoId)}&config=default&split=train`;
          const res = await fetch(url, { priority: 'low' } as RequestInit);
          if (!res.ok) return;
          const data = await res.json();
          const rows: Array<{ row_idx?: number; row?: Record<string, unknown> }> = Array.isArray(
            data.rows
          )
            ? data.rows
            : [];

          for (const entry of rows.slice(0, 28)) {
            const r = entry.row;
            if (!r) continue;
            const imgObj = r.image as { src?: string } | undefined;
            if (!imgObj || typeof imgObj.src !== 'string') continue;

            const fileId = typeof r.file_id === 'string' ? r.file_id : String(entry.row_idx ?? '');
            items.push({
              id: `${repoId}:${fileId}`,
              imageUrl: toCorsSafeAssetUrl(imgObj.src),
              title: `European Art Archive #${fileId}`,
              subtitle: `BigLAM European Art · ${repoId}`,
              repoId,
            });
          }
        } catch {
          // Ignore individual dataset errors
        }
      })(),
    ]);

    shuffleInPlace(items);
    return items;
  }
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

export function formatBytes(bytes?: number): string {
  if (bytes === undefined || bytes === null || Number.isNaN(bytes)) return 'Streamed';
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}
