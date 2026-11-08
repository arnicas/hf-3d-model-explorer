/**
 * Append samples from a single HF dataset to the existing public/manifest-3d.json
 * WITHOUT regenerating the whole manifest (which is slow + rate-limit prone).
 *
 * Usage:
 *   bun run scripts/add-dataset-samples.ts <repoId> [count] [pathPrefix] [maxMB] [--replace]
 *
 * Example:
 *   bun run scripts/add-dataset-samples.ts YiboZhang2001/TexVerse 60 glbs/glbs_2k/ 60 --replace
 *
 *   --replace  removes any existing entries for that repoId first, so you can
 *              re-sample with different filters without accumulating stale rows.
 */
import fs from 'node:fs';
import path from 'node:path';
import { listFiles } from '@huggingface/hub';

const argv = process.argv.slice(2);
const replace = argv.includes('--replace');
const pos = argv.filter((a) => !a.startsWith('--'));
const [repoId, countArg, pathPrefix, maxMbArg] = pos;

if (!repoId) {
  console.error(
    'Usage: bun run scripts/add-dataset-samples.ts <repoId> [count] [pathPrefix] [maxMB] [--replace]'
  );
  process.exit(1);
}
const wanted = Number(countArg || 40);
const maxMb = maxMbArg ? Number(maxMbArg) : 15; // keep manifest assets <= 15 MB for snappy loading
const maxBytes = maxMb * 1024 * 1024;

const token =
  process.env.HUGGINGFACE_API_KEY ||
  process.env.HF_TOKEN ||
  process.env.HUGGINGFACE_TOKEN ||
  process.env.HF_API_KEY;

const manifestPath = path.resolve(process.cwd(), 'public/manifest-3d.json');
let manifest: any[] = fs.existsSync(manifestPath)
  ? JSON.parse(fs.readFileSync(manifestPath, 'utf-8'))
  : [];

if (replace) {
  const before = manifest.length;
  manifest = manifest.filter((a) => a.repoId !== repoId);
  console.log(`--replace: dropped ${before - manifest.length} existing ${repoId} entries`);
}

const seen = new Set(manifest.map((a) => a.id));
const added: any[] = [];
let scanned = 0;
let skippedBig = 0;

const iterator = listFiles({
  repo: { type: 'dataset', name: repoId },
  recursive: true,
  credentials: token ? { accessToken: token } : undefined,
});

for await (const entry of iterator) {
  if (added.length >= wanted) break;
  if (entry.type !== 'file') continue;
  const lower = entry.path.toLowerCase();
  if (!lower.endsWith('.glb') && !lower.endsWith('.gltf') && !lower.endsWith('.obj')) continue;
  if (pathPrefix && !entry.path.startsWith(pathPrefix)) continue;
  scanned++;

  const bytes = (entry as any).size as number | undefined;
  if (bytes !== undefined && bytes > maxBytes) {
    skippedBig++;
    continue;
  }

  const id = `dataset:${repoId}/${entry.path}`;
  if (seen.has(id)) continue;
  seen.add(id);

  const ext = lower.endsWith('.obj') ? 'obj' : lower.endsWith('.gltf') ? 'gltf' : 'glb';
  const filename = entry.path.split('/').pop() || entry.path;
  const downloadUrl = `https://huggingface.co/datasets/${repoId}/resolve/main/${entry.path}`;

  added.push({
    id,
    repoId,
    repoType: 'dataset',
    path: entry.path,
    filename,
    title: filename.replace(/\.(glb|gltf|obj)$/i, ''),
    subtitle: `dataset: ${repoId}`,
    format: ext,
    size: bytes,
    downloadUrl,
    resourceBaseUrl: downloadUrl.substring(0, downloadUrl.lastIndexOf('/') + 1),
    repoUrl: `https://huggingface.co/datasets/${repoId}`,
    treeUrl: `https://huggingface.co/datasets/${repoId}/tree/main`,
  });
}

const merged = [...manifest, ...added];
fs.writeFileSync(manifestPath, JSON.stringify(merged, null, 2), 'utf-8');

const totalBytes = added.reduce((s, a) => s + (a.size || 0), 0);
const sizes = added.map((a) => (a.size || 0) / 1048576);
console.log(
  `Added ${added.length} asset(s) from ${repoId}` +
    (pathPrefix ? ` (prefix "${pathPrefix}")` : '') +
    `\n  scanned ${scanned} candidate file(s), skipped ${skippedBig} over ${maxMb}MB` +
    `\n  manifest now: ${merged.length} assets` +
    `\n  new payload: ${(totalBytes / 1048576).toFixed(1)} MB total, ` +
    `avg ${(totalBytes / 1048576 / Math.max(added.length, 1)).toFixed(1)} MB, ` +
    `max ${sizes.length ? Math.max(...sizes).toFixed(1) : 0} MB`
);
if (added.length) {
  console.log('  first:', added[0].path);
  console.log('  last: ', added[added.length - 1].path);
}
