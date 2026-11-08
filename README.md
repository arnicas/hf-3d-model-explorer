# HF 3D Roulette (Studio & Museum Showroom)

An interactive WebGL 3D experience that samples and curates random 3D objects from public datasets on Hugging Face Hub, featuring two interactive viewing modes—a single-model **Studio View** and a walkable **Museum Showroom** with a central **Thought Piece** assemblage—powered by Three.js and multi-angle **Gemini Vision** sculpture naming.

Created by **Lynn Cherny / @arnicas** ([ghostweather.com](https://ghostweather.com)), made using **Gemini Argon** in **Antigravity**.

---

## Two Interactive Views

Switch between the two views at any time using the top-left mode toggle button or pressing <kbd>M</kbd>:

### 1. Studio View (Single-Model Turntable)
* **Focused 3D Inspection**: Displays one random 3D model at a time on a warm studio turntable with OrbitControls, contact shadows, and automatic maquette scale normalization.
* **Dataset & Geometry Inspector**: The right-hand sidebar shows the source Hugging Face dataset, author, license, description, tags, and live geometry stats (vertices, triangles, meshes, file size, format) along with direct links to download the 3D file or browse the repository tree.
* **Viewport Controls**: Toggle auto-rotation (<kbd>A</kbd>), wireframe (<kbd>W</kbd>), studio floor (<kbd>G</kbd>), Y/Z up-axis flip (<kbd>U</kbd>), camera reset (<kbd>C</kbd>), or 30-second idle auto-advance (<kbd>I</kbd>).

### 2. Museum Showroom View (4 Pedestals + Central "Thought Piece" Dais)
* **Walkable 4-Pedestal Gallery**: Places 4 distinct 3D models onto individual marble-and-walnut museum plinths (<kbd>1</kbd>–<kbd>4</kbd>) arranged in a semi-circular gallery arc, surrounded by framed wall paintings streamed from Hugging Face art datasets (`Andris2067/Paintings_old`, `royam0820/baya-paintings-01`, `Nguyen17/laion_art_en`, and Met Open Access).
* **Proximity LOD & 3D Lectern Plaques**: As you walk around with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> or Arrow keys (or click any pedestal/plaque to glide to it), distant pedestals automatically transition between Full PBR Detail, Medium Lambert LOD, and Wireframe LOD. Each pedestal features an angled 3D lectern plaque displaying its title, dataset, triangle count, and scale.
* **Central "Thought Piece" Dais (<kbd>5</kbd> / `⚡ Thought Piece`)**:
  * Positioned at the center of the rotunda, the **Thought Piece** fuses 3 models from the surrounding pedestals into a single composite sculpture.
  * **Totem & Collision Modes (<kbd>T</kbd>)**: Defaults to a vertical **Totem Stack** (which analyzes 10-band vertical cross-section profiles and vertex medoids to stack the heaviest base on the bottom and interlock joints in 3D space) and can be toggled to a chaotic **3-Model Collision**.
  * **Automatic Random Patina on Untextured Meshes**: Any untextured clay/scan meshes included in the Thought Piece automatically receive a randomized procedural museum patina (Gilded Cast Bronze, Oxidized Verdigris Copper, Brushed Damascus Chrome, or Gold-Veined Nero Marquina Marble), while textured and multi-color low-poly models keep their native materials.
  * **Automatic & Manual Re-Assembly**: The Thought Piece automatically rebuilds whenever **2 new sculptures** are loaded into the museum pedestals (showing `"Under construction"` on its plaque whenever empty), and can also be re-assembled manually at any time with <kbd>R</kbd> when standing near the central dais.

---

## How Gemini Captions the Combined "Thought Piece" Artwork

Whenever 3 models are automatically combined onto the central **Thought Piece** dais in the Museum Showroom:

1. **Multi-Angle Snapshot Photography**: [`ThreeModelViewer.captureDaisSculptureSnapshots()`](src/viewer.ts) captures **3 multi-angle 256×256 JPEG snapshots** (`quality: 0.8`) from three camera angles (`-0.68`, `0.68`, and `2.38` radians) around the combined sculpture and immediately restores the user camera frame so no flicker occurs. Because each image is 256×256 (≤384×384), Gemini bills a single tile (~258 tokens) per image (~774 image tokens total per call).
2. **Multimodal Gemini Naming (`@google/genai`)**: [`generateSculptureNameForDais()`](src/gemini-visitor.ts) sends the 3 JPEG views to **`gemini-3.5-flash-lite`** with `ThinkingLevel.MINIMAL`, prompting it to create a **2 to 6 word sculpture title** in structured JSON (`{ "name": "..." }`).
3. **Live Plaque Engraving**: As soon as Gemini returns the title, the 3D lectern plaque in front of the central dais re-renders with the new title (e.g., `✦ EXHIBIT TITLE · NAMED BY GEMINI FROM 3 CAMERA ANGLES`).
4. **Cost-Conscious Manual Controls**: Gemini is only called when a new Thought Piece is automatically assembled (on initial Showroom entry and after every 2 new sculptures are added to the museum). Manually clicking **💥 Re-Assemble (<kbd>R</kbd>)** or toggling **🗿 Mode: Totem / Smash (<kbd>T</kbd>)** re-arranges the current 3 objects and keeps the existing Gemini title with **zero additional Gemini API calls**.

---

## Quickstart & Running Locally

This project is built with modern ESM and requires **[Bun](https://bun.sh/)** (recommended: Bun v1.2+).

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# Provide your Gemini API key (supports either name)
VITE_GEMINI_API_KEY=your_gemini_api_key_here
# or
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Note:** If no Gemini API key is provided, the gallery seamlessly falls back to pre-authored in-memory gallery visitor commentary with full spatial audio.

### 2. Development

```bash
# Install dependencies
bun install

# Start Vite development server
bun run dev
```

Open **`http://localhost:5173/`** in Chrome, Edge, or Brave.

---

## Building for Production

To create an optimized, production-ready static build:

```bash
bun run build
# (or: npm run build)
```

### What this does:
1. **TypeScript Verification (`tsc`)**: Runs static type checks across Three.js scenes, the Web Worker, and Gemini SDK.
2. **Vite Production Bundler**: Bundles and tree-shakes assets into the **`dist/`** folder:
   * `dist/index.html` (Application HTML)
   * `dist/assets/index-*.js` (Three.js showroom, UI controls, Hugging Face crawler, Gemini Vision client)
   * `dist/assets/index-*.css` (Gallery styling, HUD layouts, and responsive UI)
   * `dist/assets/kokoro.worker-*.js` (Background neural TTS worker)
   * `dist/assets/ort-wasm-simd-threaded.*.wasm` (ONNX Runtime WASM binary for in-browser speech synthesis)

### Testing the Production Build Locally

```bash
bun run preview
```
This spins up a local web server serving the `dist/` directory at **`http://localhost:4173/`** exactly as it will run when deployed to static hosts (GitHub Pages, Cloudflare Pages, Netlify, Vercel, Firebase Hosting, etc.).

---

## AI Models Called

### 1. Vision Analysis & Multimodal Commentary: **Gemini 3.5 Flash Lite**
* **Model Identifier**: `gemini-3.5-flash-lite`
* **SDK**: `@google/genai` (v3.0.0 official SDK)
* **API Call**: Direct in-browser client-side API call via `ai.models.generateContentStream`
* **Configuration**:
  * `thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL }` — keeps latency fast and costs minimal.
  * `responseMimeType: "application/json"` with strict `responseSchema` (`{ "comments": string[] }`).
* **Input**:
  * A 256×256 JPEG snapshot generated by an offscreen WebGL renderer against a warm cream backdrop (`#f1eee7`) with dynamic bounding-box camera framing.
  * Rotating gallery visitor persona prompt (tired kid, pretentious art student, gamer roasting hitboxes, museum guard, eccentric collector, lost tourist).
  * Prompt instructions formatted specifically for TTS cadence (ellipses `...`, em-dashes `—`, rhetorical `?`, capitalized emphasis).
* **Caching**: Results are saved to browser `localStorage` (`hf3d_gemini_whisper_v5:<assetId>`), ensuring $0 quota re-use across repeat viewings.

### 2. In-Browser Neural Text-to-Speech: **Kokoro-82M**
* **Model Identifier**: `onnx-community/Kokoro-82M-v1.0-ONNX` (quantization: `q8`)
* **Engine**: `kokoro-js` (runs fully in-browser via WebAssembly and Web Worker, zero cloud TTS costs)
* **Worker Execution**: Dedicated background worker ([`src/kokoro.worker.ts`](src/kokoro.worker.ts)) to keep 60 FPS Three.js rendering silky smooth.
* **Voices Used**: Cycles across 11 distinct American & British male and female personas without consecutive repeats:
  * American Female: `af_nicole` (whispery), `af_heart` (warm), `af_bella` (expressive), `af_sky` (breezy)
  * American Male: `am_adam` (young adult/gamer), `am_michael` (relaxed), `am_eric` (techie/kid)
  * British Female: `bf_emma` (thoughtful student), `bf_isabella` (analytical)
  * British Male: `bm_george` (curatorial/guard), `bm_lewis` (gentleman)

---

## 3D Asset Loading & Compression

* **Three.js**: `three` (v0.174.0)
* **3D Loaders Supported**:
  * `GLTFLoader` (with Draco and Meshopt extensions)
  * `OBJLoader` & `MTLLoader`
  * `STLLoader`
  * `FBXLoader`
  * `PLYLoader`
* **Meshopt Decoder**: `meshoptimizer` (`MeshoptDecoder`) registered directly on `GLTFLoader` to decompress complex scans (e.g., Smithsonian 3D archive objects).
* **Hugging Face Hub API**:
  * `@huggingface/hub`: Direct git-tree file discovery (`listFiles`).
  * **HF Datasets Server API**: Real-time pagination queries via `/rows?dataset=...&offset={randomOffset}&length=100` ensuring random selection across datasets with tens of thousands of rows (e.g. `cfahlgren1/lowpoly-coder-4b-data`, `codepawl/smithsonian-3d-glb`).

---

## Static Asset Manifests (Fast & Rate-Limit Proof)

To protect the application against Hugging Face API rate limits (`429 Too Many Requests`) for public visitors, the app boots instantly using pre-compiled static manifests located in `public/`:
* **`public/manifest-3d.json`** (1,231 verified `.glb` 3D assets across `cfahlgren1/lowpoly-coder-4b-data`, `TuTao/scandi-home-lowpoly`, `codepawl/smithsonian-3d-glb`, `cfahlgren1/3d-arena`, `cfahlgren1/3d-collection`, `codepawl/google-poly-glb`, `hugging-3d/smithsonian-open-access-3d`, `YiboZhang2001/TexVerse-1K`, `YiboZhang2001/TexVerse`, `Optikan112/abo-furniture-glb`, and `codepawl/scanned-objects-glb`).
* **`public/manifest-art.json`** (80 permanent framed classical and modern wall paintings).

When a visitor opens the app, all 3D models and paintings hydrate in **~50 milliseconds** with **zero calls** to Hugging Face metadata endpoints.

### Recompiling the Manifests (Optional)

If you ever wish to re-scrape or expand the manifests:

```bash
# Set your token in .env (local only, never exposed to visitors):
HUGGINGFACE_API_KEY=hf_...

# Generate 3D models manifest
bun run generate-manifest

# Generate framed wall art manifest
bun run scripts/generate-art-manifest.ts
```

To add samples from one dataset without regenerating the whole manifest (fast, no `429` storms):

```bash
# <repoId> [count] [pathPrefix] [maxMB] [--replace]
bun run add-dataset YiboZhang2001/TexVerse 60 glbs/glbs_2k/ 60 --replace
```

---

## How Hugging Face Is Polled for Additional Models

During normal browsing, the app relies on `public/manifest-3d.json` (1,231 models) and embedded `STATIC_DATASET_METADATA` ([`src/hf.ts`](src/hf.ts)), making **zero live Hugging Face API requests** so visitors never trigger `429 Too Many Requests` errors.

Live Hugging Face polling ([`HF3DDiscoveryService.discoverMoreInBackground()`](src/hf.ts)) is only invoked when:
1. **Low Pool Threshold**: The active pool drops below 40 models (`matchingPool.length < 40 && stagedCount < 40`), or `manifest-3d.json` is unavailable.
2. **Manual Refresh (Hidden by Default)**: The **Refresh Pool** button (`#btn-refresh-pool` in [`index.html`](index.html), hidden by default) is unhidden and clicked (`handleRefreshPool()`).

When live polling runs, it stages newly discovered 3D files into a background buffer (`stagedPool`, surfaced via the `+N` badge on **Refresh Pool**) in two phases:
1. **Phase 1 — Curated Seed Dataset Rotation (`SEED_DATASETS`)**:
   * Steps through one unvisited repository per wave from `SEED_DATASETS`.
   * First queries the **Hugging Face Datasets Server API** (`/first-rows` and `/rows?dataset=...&offset={randomOffset}&length=100`) to sample random slices across large Parquet/tabular 3D datasets without downloading full tables.
   * If the dataset is not tabular, falls back to `@huggingface/hub` **`listFiles()`** (recursive git-tree walk), filtering for `.glb`, `.gltf`, `.obj`, and `.stl` files between `MIN_QUALITY_FILE_BYTES` (2 KB) and `MAX_RANDOM_FILE_BYTES` (18 MB).
2. **Phase 2 — Live Hub Keyword Crawl (`CRAWL_QUERIES`)**:
   * Once all seed datasets have been visited, subsequent waves cycle through search keywords (`glb`, `3d`, `objaverse`, `gltf`, `meshes`, `3d-models`, `photogrammetry`, `lowpoly`, `furniture`, `architecture`, `cultural-heritage`, etc.) via `@huggingface/hub` `listDatasets()` and `listModels()`, scanning up to 2 newly discovered repositories per wave.
   * Staged models are merged into the active pool (`MAX_POOL_SIZE = 2000`) when **Refresh Pool** is clicked or automatically on the 30-second idle cycle once `>= 20` staged items are ready.

---

## Performance & Manifest Notes

### 1. `pickRandom()` Uniform Repository Sampling (`O(n)`)

`HF3DDiscoveryService.pickRandom()` ([`src/hf.ts`](src/hf.ts)) samples a repository uniformly at random first and then picks a random asset within that repository so large datasets do not drown out smaller collections. Rather than calling `getRepoFileCount()` per candidate (which previously caused `O(n²)` scans over the pool), `pickRandom()` precomputes `repoCounts` in a single **`O(n)`** pass before grouping candidates by `repoId`.

### 2. Manifest File-Size Guards

`MAX_RANDOM_FILE_BYTES` (18 MB) and `MIN_QUALITY_FILE_BYTES` (2 KB) are enforced in `createAsset()` during live Hub discovery. Because assets loaded directly from `public/manifest-3d.json` bypass `createAsset()`, file-size filtering (`<= 12–15 MB`) is enforced at manifest generation time via `maxSizeBytes` in [`scripts/generate-manifest.ts`](scripts/generate-manifest.ts) and the `[maxMB]` argument of `add-dataset`.

---

## Gallery Controls

| Action | Control |
| :--- | :--- |
| **Walk around Gallery** | <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> or Arrow Keys |
| **Look around** | Click + Drag mouse |
| **Jump / Walk to Pedestal** | <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>, <kbd>4</kbd>, <kbd>5</kbd> (`⚡ Thought Piece`) or click any lectern plaque |
| **Re-Assemble Thought Piece** | <kbd>R</kbd> or click **💥 Re-Assemble** (when near the Thought Piece dais) |
| **Toggle Totem / Collision Mode** | <kbd>T</kbd> or click **🗿 Mode: Totem** (when near the Thought Piece dais) |
| **Roll New Exhibit** | <kbd>Space</kbd> (loads a new model; Thought Piece auto-rebuilds every 2 new museum exhibits) |
| **Toggle View Mode** | <kbd>M</kbd> (switches between Single-Model Studio and 4-Pedestal Museum Showroom) |
| **Toggle Spatial Audio** | Click the **🔊 Audio** sidebar pill |
| **Toggle Auto-Idle Next** | <kbd>I</kbd> or click **Idle Next: 30s** pill |
| **Open HF Dataset / Download** | Click `#N 📊 Dataset ↗` or `#N ⬇️ Download 3D ↗` in the top header |
