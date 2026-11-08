import fs from 'node:fs';
import path from 'node:path';
import { listFiles } from '@huggingface/hub';

/**
 * Polite, throttled generator that compiles a diverse static manifest of framed wall art
 * from curated Hugging Face art datasets:
 * 1. Artificio/WikiArt (Classical, Impressionist, Realist paintings)
 * 2. jiovine/pixel-art-nouns (Playful pixel art generative nouns)
 * 3. metmuseum/openaccess (Metropolitan Museum of Art Open Access masterworks)
 * 4. Nguyen17/laion_art_en (High-aesthetic artistic photography & illustration, pre-verified for 200 OK)
 * 5. Andris2067/Paintings_old (European Masters classical oil studies)
 * 6. royam0820/baya-paintings-01 (Baya Mahieddine modern gouache archives)
 */

interface WallArtworkItem {
  id: string;
  imageUrl: string;
  title: string;
  subtitle: string;
  repoId: string;
}

const token =
  process.env.HUGGINGFACE_API_KEY ||
  process.env.HF_TOKEN ||
  process.env.HUGGINGFACE_TOKEN ||
  process.env.HF_API_KEY;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jitter(): Promise<void> {
  return sleep(1200 + Math.floor(Math.random() * 1500));
}

function getAuthHeaders(): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchWithRetry(url: string, retries = 2, timeoutMs = 4500): Promise<Response | null> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const res = await fetch(url, { headers: getAuthHeaders(), signal: controller.signal });
      clearTimeout(timer);

      if (res.status === 429) {
        const retryAfter = Number(res.headers.get('retry-after')) || attempt * 6;
        console.warn(`[Rate Limited 429] Backing off for ${retryAfter}s... (attempt ${attempt}/${retries})`);
        await sleep(retryAfter * 1000);
        continue;
      }
      if (res.status >= 500) {
        // Skip 502/504 instantly without hanging
        console.warn(`  ↳ Server status ${res.status} on ${url.slice(0, 70)}... skipping`);
        return null;
      }
      return res;
    } catch {
      if (attempt === retries) return null;
      await sleep(1000 * attempt);
    }
  }
  return null;
}

async function verifyImageReachable(url: string, timeoutMs = 3500): Promise<boolean> {
  try {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { method: 'HEAD', signal: controller.signal });
    clearTimeout(id);
    return res.ok;
  } catch {
    return false;
  }
}

async function fetchWikiArtArtworks(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];
  const repoId = 'Artificio/WikiArt';
  console.log(`\n🎨 Sampling "${repoId}" across multiple offsets...`);

  const offsets = [0, 250, 800, 2200, 5000, 12000, 25000];

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=default&split=train&offset=${offset}&length=15`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) continue;

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || !row.image?.src) continue;

        const title = (row.title as string) || 'Untitled Study';
        const artist = (row.artist as string) || 'Master Unknown';
        const style = (row.style as string) || (row.genre as string) || '';
        const date = (row.date as string) ? ` (${row.date})` : '';

        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(row.image.src)}&w=800&output=jpg`;

        items.push({
          id: `${repoId}:${entry.row_idx}`,
          imageUrl: safeImageUrl,
          title,
          subtitle: `${artist}${date}${style ? ` · ${style}` : ''}`,
          repoId,
        });
      }
      console.log(`  ✓ Offset ${offset}: collected ${rows.length} paintings (total: ${items.length})`);
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return items;
}

async function fetchPixelArtNouns(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];
  const repoId = 'jiovine/pixel-art-nouns';
  console.log(`\n👾 Sampling "${repoId}" pixel art...`);

  const offsets = [0, 150, 600, 1800];

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=default&split=train&offset=${offset}&length=10`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) continue;

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || !row.image?.src) continue;

        const rawText = (row.text as string) || 'Pixel Art Character';
        // Clean text into punchy title
        let title = rawText.split(',')[0].replace(/^a character with /i, '').replace(/^a /i, '');
        title = title.charAt(0).toUpperCase() + title.slice(1);

        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(row.image.src)}&w=640&output=png`;

        items.push({
          id: `${repoId}:${entry.row_idx}`,
          imageUrl: safeImageUrl,
          title: `Noun · ${title}`,
          subtitle: `Pixel Art Noun #${entry.row_idx} · ${repoId}`,
          repoId,
        });
      }
      console.log(`  ✓ Offset ${offset}: collected ${rows.length} pixel nouns (total: ${items.length})`);
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return items;
}

async function fetchLaionArtEn(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];
  const repoId = 'Nguyen17/laion_art_en';
  console.log(`\n📸 Sampling high-aesthetic works from "${repoId}" (verifying live URLs)...`);

  const offsets = [10, 200, 1000];

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=default&split=train&offset=${offset}&length=25`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) continue;

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || typeof row.URL !== 'string' || !row.URL.startsWith('http')) continue;
        if (typeof row.aesthetic === 'number' && row.aesthetic < 6.5) continue;

        const imgUrl = row.URL;
        // Verify live reachability
        const alive = await verifyImageReachable(imgUrl, 2500);
        if (!alive) continue;

        const rawText = (row.TEXT as string) || 'Aesthetic Study';
        const cleanTitle = rawText.split(/[-–|]/)[0].trim().slice(0, 48);

        // Wrap through wsrv.nl proxy for fast global CDN and CORS
        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(imgUrl)}&w=800&output=jpg`;

        items.push({
          id: `${repoId}:${entry.row_idx}`,
          imageUrl: safeImageUrl,
          title: cleanTitle,
          subtitle: `Score: ${Number(row.aesthetic || 7).toFixed(1)}/10 · ${repoId}`,
          repoId,
        });

        if (items.length >= 24) break;
      }
      console.log(`  ✓ Offset ${offset}: verified ${items.length} live artworks from LAION Art`);
      if (items.length >= 24) break;
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return items;
}

async function fetchMitsuaArtMuseums(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];
  const repoId = 'Mitsua/art-museums-pd-440k';
  console.log(`\n🏛️ Sampling Public Domain Museum Masterpieces from "${repoId}"...`);

  const offsets = [0, 80, 250, 700, 1500, 3500];

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=default&split=train&offset=${offset}&length=12`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) continue;

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || !row.jpg?.src) continue;

        const info = (row.json && typeof row.json === 'object') ? row.json : {};
        const title = (info.Title as string) || 'Museum Masterwork';
        const author = (info.Author && info.Author !== 'N/A') ? info.Author : 'Public Domain Master';
        const source = info.source ? ` · ${String(info.source).toUpperCase()}` : '';

        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(row.jpg.src)}&w=800&output=jpg`;

        items.push({
          id: `${repoId}:${entry.row_idx}`,
          imageUrl: safeImageUrl,
          title,
          subtitle: `${author}${source} · ${repoId}`,
          repoId,
        });
      }
      console.log(`  ✓ Offset ${offset}: collected ${rows.length} museum masterworks (total: ${items.length})`);
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return items;
}

async function fetchMetMuseumArtworks(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];
  const repoId = 'metmuseum/openaccess';
  console.log(`\n🏛️ Sampling "${repoId}" masterworks...`);

  const offsets = [0, 40, 120];

  for (const offset of offsets) {
    await jitter();
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(
      repoId
    )}&config=default&split=train&offset=${offset}&length=15`;

    const res = await fetchWithRetry(url);
    if (!res || !res.ok) continue;

    try {
      const data = await res.json();
      const rows: Array<{ row_idx: number; row: Record<string, any> }> = Array.isArray(data.rows)
        ? data.rows
        : [];

      for (const entry of rows) {
        const row = entry.row;
        if (!row || typeof row.primaryImage !== 'string' || !row.primaryImage.startsWith('http')) continue;

        const title = (row.title as string) || 'Museum Artifact';
        const artist = (row.artistDisplayName as string) || (row.culture as string) || 'The Met Collection';
        const date = (row.objectDate as string) ? ` · ${row.objectDate}` : '';

        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(row.primaryImage)}&w=800&output=jpg`;
        items.push({
          id: `${repoId}:${row.objectID || entry.row_idx}`,
          imageUrl: safeImageUrl,
          title,
          subtitle: `${artist}${date}`,
          repoId,
        });
      }
      console.log(`  ✓ Offset ${offset}: collected ${rows.length} Met items (total: ${items.length})`);
    } catch (e) {
      console.warn(`  ↳ Error parsing offset ${offset}:`, e);
    }
  }

  return items;
}

async function fetchDirectGitPaintings(): Promise<WallArtworkItem[]> {
  const items: WallArtworkItem[] = [];

  const repos: Array<{ repoId: string; collectionTitle: string; artist: string; maxFiles: number }> = [
    {
      repoId: 'Andris2067/Paintings_old',
      collectionTitle: 'Classical Oil Study',
      artist: 'European Masters Archive',
      maxFiles: 35,
    },
  ];

  for (const { repoId, collectionTitle, artist, maxFiles } of repos) {
    console.log(`\n⏳ Listing git tree files in "${repoId}"...`);
    await jitter();

    try {
      let count = 0;
      for await (const f of listFiles({
        repo: { type: 'dataset', name: repoId },
        recursive: true,
        credentials: token ? { accessToken: token } : undefined,
      })) {
        if (f.type !== 'file' || !f.path.match(/\.(jpg|jpeg|png|webp)$/i)) continue;
        count++;
        if (count > maxFiles) break;

        const encodedPath = f.path.split('/').map(encodeURIComponent).join('/');
        const rawUrl = `https://huggingface.co/datasets/${repoId}/resolve/main/${encodedPath}`;
        const safeImageUrl = `https://wsrv.nl/?url=${encodeURIComponent(rawUrl)}&w=800&output=jpg`;

        items.push({
          id: `${repoId}:${f.path}`,
          imageUrl: safeImageUrl,
          title: `${collectionTitle} #${count}`,
          subtitle: `${artist} · ${repoId}`,
          repoId,
        });
      }
      console.log(`  ✓ Collected ${count} paintings from ${repoId}`);
    } catch (e) {
      console.warn(`  ↳ Failed to list files in ${repoId}:`, e);
    }
  }

  return items;
}

async function main() {
  console.log('🎨 Starting Expanded Polite Wall Art Manifest Generator...');
  console.log(token ? '🔑 HF Token detected' : '⚠️ No HF Token detected (unauthenticated)');

  const allItems: WallArtworkItem[] = [];
  const seenIds = new Set<string>();

  // 1. Artificio/WikiArt
  const wikiArt = await fetchWikiArtArtworks();
  for (const w of wikiArt) {
    if (!seenIds.has(w.id)) {
      seenIds.add(w.id);
      allItems.push(w);
    }
  }

  // 2. jiovine/pixel-art-nouns
  const pixelNouns = await fetchPixelArtNouns();
  for (const p of pixelNouns) {
    if (!seenIds.has(p.id)) {
      seenIds.add(p.id);
      allItems.push(p);
    }
  }

  // 3. Nguyen17/laion_art_en
  const laionArt = await fetchLaionArtEn();
  for (const l of laionArt) {
    if (!seenIds.has(l.id)) {
      seenIds.add(l.id);
      allItems.push(l);
    }
  }

  // 4. Mitsua/art-museums-pd-440k
  const mitsuaArt = await fetchMitsuaArtMuseums();
  for (const mi of mitsuaArt) {
    if (!seenIds.has(mi.id)) {
      seenIds.add(mi.id);
      allItems.push(mi);
    }
  }

  // 5. The Met Museum Open Access
  const metArt = await fetchMetMuseumArtworks();
  for (const m of metArt) {
    if (!seenIds.has(m.id)) {
      seenIds.add(m.id);
      allItems.push(m);
    }
  }

  // 6. Direct Git Oil & Gouache Repos
  const gitArt = await fetchDirectGitPaintings();
  for (const g of gitArt) {
    if (!seenIds.has(g.id)) {
      seenIds.add(g.id);
      allItems.push(g);
    }
  }

  // Shuffle final list so paintings are completely mixed across styles on each gallery wall
  for (let i = allItems.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allItems[i], allItems[j]] = [allItems[j], allItems[i]];
  }

  const outDir = path.resolve(process.cwd(), 'public');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const outPath = path.join(outDir, 'manifest-art.json');
  fs.writeFileSync(outPath, JSON.stringify(allItems, null, 2), 'utf-8');

  console.log(`\n🎉 Success! Wrote ${allItems.length} diverse framed wall paintings to ${outPath}`);
  console.log(`📦 File size: ${(fs.statSync(outPath).size / 1024).toFixed(1)} KB`);
}

main().catch(console.error);
