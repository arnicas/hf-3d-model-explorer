import {
  formatBytes,
  HF3DDiscoveryService,
  isSame3DObject,
  type DatasetMetadata,
  type Discovered3DAsset,
  type SearchFilterOptions,
  type WallArtworkItem,
} from './hf';
import { ThreeModelViewer } from './viewer';
// import { generateVisitorWhispersForExhibit } from './gemini-visitor';

const appRoot = document.getElementById('app') as HTMLElement;
const canvas = document.getElementById('three-canvas') as HTMLCanvasElement;
const viewportContainer = document.getElementById('viewport-container') as HTMLElement;
const loadingOverlay = document.getElementById('loading-overlay') as HTMLElement;
const loadingTitle = document.getElementById('loading-title') as HTMLElement;
const loadingSubtitle = document.getElementById('loading-subtitle') as HTMLElement;
const toastBanner = document.getElementById('toast-banner') as HTMLElement;
const bgDiscoveryStatus = document.getElementById('bg-discovery-status') as HTMLElement;

// Mode Switcher & Showroom HUD Elements
const btnToggleMode = document.getElementById('btn-toggle-mode') as HTMLButtonElement;
const modeToggleIcon = document.getElementById('mode-toggle-icon') as HTMLElement;
const modeToggleText = document.getElementById('mode-toggle-text') as HTMLElement;
const showroomHud = document.getElementById('showroom-hud') as HTMLElement;
const btnShowroomRoll = document.getElementById('btn-showroom-roll') as HTMLButtonElement;
const showroomCardsLayer = document.getElementById('showroom-cards-layer') as HTMLElement;
const showroomDatasetLink = document.getElementById('showroom-dataset-link') as HTMLAnchorElement;
const showroomDownloadLink = document.getElementById('showroom-download-link') as HTMLAnchorElement;
const pedestalJumpButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('button.pedestal-jump-btn[data-slot]')
);

const btnRandom = document.getElementById('btn-random') as HTMLButtonElement;
const btnRefreshPool = document.getElementById('btn-refresh-pool') as HTMLButtonElement;
const stagedBadge = document.getElementById('staged-badge') as HTMLElement;

// Dataset Info Elements
const datasetLicenseBadge = document.getElementById('dataset-license-badge') as HTMLElement;
const datasetRepoLink = document.getElementById('dataset-repo-link') as HTMLAnchorElement;
const datasetAuthor = document.getElementById('dataset-author') as HTMLElement;
const datasetLikes = document.getElementById('dataset-likes') as HTMLElement;
const datasetDownloads = document.getElementById('dataset-downloads') as HTMLElement;
const datasetFileCount = document.getElementById('dataset-file-count') as HTMLElement;
const datasetDescription = document.getElementById('dataset-description') as HTMLElement;
const datasetTags = document.getElementById('dataset-tags') as HTMLElement;

// Current 3D Object Elements
const modelFormatBadge = document.getElementById('model-format-badge') as HTMLElement;
const metaTitleBlock = document.getElementById('meta-title-block') as HTMLElement;
const metaObjectTitle = document.getElementById('meta-object-title') as HTMLElement;
const metaObjectSubtitle = document.getElementById('meta-object-subtitle') as HTMLElement;
const metaFilePath = document.getElementById('meta-file-path') as HTMLElement;
const statVertices = document.getElementById('stat-vertices') as HTMLElement;
const statTriangles = document.getElementById('stat-triangles') as HTMLElement;
const statMeshes = document.getElementById('stat-meshes') as HTMLElement;
const statSize = document.getElementById('stat-size') as HTMLElement;
const metaDownloadLink = document.getElementById('meta-download-link') as HTMLAnchorElement;
const metaHfTreeLink = document.getElementById('meta-hf-tree-link') as HTMLAnchorElement;

// Debug Drawer & Memory Telemetry Elements
const memGeoms = document.getElementById('mem-geoms') as HTMLElement;
const memTex = document.getElementById('mem-tex') as HTMLElement;
const memHeap = document.getElementById('mem-heap') as HTMLElement;
const searchForm = document.getElementById('search-form') as HTMLFormElement;
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const repoTypeSelect = document.getElementById('repo-type-select') as HTMLSelectElement;
const formatSelect = document.getElementById('format-select') as HTMLSelectElement;
const poolCountBadge = document.getElementById('pool-count-badge') as HTMLElement;
const discoveredList = document.getElementById('discovered-list') as HTMLUListElement;

// Viewport Toolbar
const btnIdleCycle = document.getElementById('btn-idle-cycle') as HTMLButtonElement;
const idleCycleLabel = document.getElementById('idle-cycle-label') as HTMLElement;
const btnAudioToggle = document.getElementById('btn-audio-toggle') as HTMLButtonElement;
const audioToggleLabel = document.getElementById('audio-toggle-label') as HTMLElement;
const btnAutorotate = document.getElementById('btn-autorotate') as HTMLButtonElement;
const btnWireframe = document.getElementById('btn-wireframe') as HTMLButtonElement;
const btnGrid = document.getElementById('btn-grid') as HTMLButtonElement;
const btnFlipUp = document.getElementById('btn-flip-up') as HTMLButtonElement;
const btnResetCam = document.getElementById('btn-reset-cam') as HTMLButtonElement;

const discovery = new HF3DDiscoveryService();
const viewer = new ThreeModelViewer(canvas, viewportContainer);

const IDLE_TIMEOUT_SECONDS = 30;
let idleCycleEnabled = true;
let idleSecondsRemaining = IDLE_TIMEOUT_SECONDS;

let currentAsset: Discovered3DAsset | null = null;
const recentHistory: Discovered3DAsset[] = [];
let wallArtPool: WallArtworkItem[] = [];
let totalModelsLoaded = 0;
let isBusy = false;
let isBackgroundCrawling = false;
let toastTimer: number | null = null;

function updateFocusedPedestalUI(slotIndex: number): void {
  for (const btn of pedestalJumpButtons) {
    const idx = Number(btn.dataset.slot);
    btn.classList.toggle('active', idx === slotIndex);
  }
  const slotInfo = viewer.getSlotInfo(slotIndex);
  if (slotInfo?.asset) {
    currentAsset = slotInfo.asset;
    const repoShort = slotInfo.asset.repoId.split('/')[1] || slotInfo.asset.repoId;
    showroomDatasetLink.href = slotInfo.asset.repoUrl;
    showroomDatasetLink.textContent = `#${slotIndex + 1} 📊 ${repoShort} ↗`;
    showroomDownloadLink.href = slotInfo.asset.downloadUrl;
    showroomDownloadLink.textContent = `#${slotIndex + 1} ⬇️ Download 3D ↗`;
    if (slotInfo.datasetMeta) {
      renderDatasetMetadata(slotInfo.datasetMeta);
    }
  }
}

viewer.setCallbacks({
  onPedestalSelect: (slotIdx) => updateFocusedPedestalUI(slotIdx),
});

function resetIdleTimer(): void {
  idleSecondsRemaining = IDLE_TIMEOUT_SECONDS;
  updateIdleButtonUI();
}

function updateIdleButtonUI(): void {
  if (!idleCycleEnabled) {
    idleCycleLabel.textContent = 'Idle Next: Off';
  } else {
    idleCycleLabel.textContent = `Idle Next: ${idleSecondsRemaining}s`;
  }
}

function updateMemoryTelemetryUI(): void {
  const stats = viewer.getMemoryStats();
  memGeoms.textContent = String(stats.gpuGeometries);
  memTex.textContent = String(stats.gpuTextures);
  memHeap.textContent = stats.jsHeapMB !== null ? `${stats.jsHeapMB} MB` : 'N/A';
}

function getFilterOptions(): SearchFilterOptions {
  return {
    query: searchInput.value.trim(),
    repoType: (repoTypeSelect.value as SearchFilterOptions['repoType']) || 'all',
    format: (formatSelect.value as SearchFilterOptions['format']) || 'all',
  };
}

function setLoadingState(busy: boolean, title = '', subtitle = ''): void {
  isBusy = busy;
  btnRandom.disabled = busy;
  btnShowroomRoll.disabled = busy;

  if (viewer.getMode() === 'showroom') {
    // Never cover or blur the 3D museum with a full-screen overlay;
    // the plinth spotlight/smoke/elevator animation and spatial chime announce new arrivals!
    loadingOverlay.hidden = true;
    btnShowroomRoll.innerHTML = busy
      ? '⏳ Curating Exhibit…'
      : '🎲 Roll New Exhibit <kbd class="shortcut-hint">Space</kbd>';
    return;
  }

  if (busy) {
    loadingTitle.textContent = title || 'Loading 3D Model...';
    loadingSubtitle.textContent = subtitle || '';
    loadingOverlay.hidden = false;
  } else {
    loadingOverlay.hidden = true;
  }
}

function showToast(message: string, durationMs = 4000): void {
  if (toastTimer) {
    window.clearTimeout(toastTimer);
  }
  toastBanner.textContent = message;
  toastBanner.hidden = false;
  toastTimer = window.setTimeout(() => {
    toastBanner.hidden = true;
  }, durationMs);
}

function updateStagedUI(): void {
  const staged = discovery.getStagedCount();
  const activeDatasets = discovery.getActiveDatasetCount();
  const activeFiles = discovery.getPool().length;

  if (staged > 0) {
    stagedBadge.textContent = `+${staged}`;
    stagedBadge.hidden = false;
    bgDiscoveryStatus.textContent = `${activeDatasets} datasets active (${activeFiles} models) · ${staged} new ready`;
  } else {
    stagedBadge.hidden = true;
    bgDiscoveryStatus.textContent = `${activeDatasets} datasets active · ${activeFiles} models in pool`;
  }
}

function renderDatasetMetadata(meta: DatasetMetadata): void {
  datasetRepoLink.textContent = meta.repoId;
  datasetRepoLink.href = meta.repoUrl;
  datasetAuthor.textContent = `by ${meta.author}`;

  datasetLikes.textContent = meta.likes.toLocaleString();
  datasetDownloads.textContent = meta.downloads.toLocaleString();
  datasetFileCount.textContent = meta.discoveredFilesInRepo.toLocaleString();
  datasetDescription.textContent = meta.description;

  if (meta.license) {
    datasetLicenseBadge.textContent = meta.license;
    datasetLicenseBadge.hidden = false;
  } else {
    datasetLicenseBadge.hidden = true;
  }

  datasetTags.innerHTML = '';
  for (const tag of meta.tags) {
    const span = document.createElement('span');
    span.className = 'dataset-tag';
    span.textContent = tag;
    datasetTags.appendChild(span);
  }
}

function renderPoolList(): void {
  const filters = getFilterOptions();
  const pool = discovery.getPool(filters);
  poolCountBadge.textContent = `${pool.length} ${pool.length === 1 ? 'file' : 'files'}`;
  updateStagedUI();

  discoveredList.innerHTML = '';
  const slice = pool.slice(0, 60);
  for (const asset of slice) {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `discovered-item${currentAsset?.id === asset.id ? ' active' : ''}`;
    btn.title = `${asset.repoType}: ${asset.repoId}\n${asset.path}`;

    const infoSpan = document.createElement('span');
    infoSpan.className = 'discovered-item-info';
    infoSpan.textContent = asset.title || asset.filename;

    const repoSpan = document.createElement('span');
    repoSpan.className = 'discovered-item-repo';
    repoSpan.textContent = `${asset.repoType === 'dataset' ? '📊' : '📦'} ${asset.repoId}`;
    infoSpan.appendChild(repoSpan);

    const extSpan = document.createElement('span');
    extSpan.className = 'discovered-item-ext';
    extSpan.textContent = `.${asset.format}`;

    btn.append(infoSpan, extSpan);
    btn.addEventListener('click', () => {
      if (!isBusy) {
        resetIdleTimer();
        void loadSpecificAsset(asset);
      }
    });

    li.appendChild(btn);
    discoveredList.appendChild(li);
  }
}

/**
 * Returns all 3D assets currently displayed on the 4 showroom plinths plus the recent history set,
 * so the same object is never loaded twice into the same set of 4.
 */
function getActiveSetAssets(): Discovered3DAsset[] {
  const active: Discovered3DAsset[] = [...viewer.getLoadedSlotAssets()];
  for (const h of recentHistory) {
    if (!active.some((a) => isSame3DObject(a, h))) {
      active.push(h);
    }
  }
  if (currentAsset && !active.some((a) => isSame3DObject(a, currentAsset!))) {
    active.push(currentAsset);
  }
  return active;
}

async function loadSpecificAsset(
  asset: Discovered3DAsset,
  targetPedestalSlot?: number,
  focusOnLoad = true
): Promise<boolean> {
  // Block loading the same object onto a second plinth in the set of 4
  if (viewer.getMode() === 'showroom' || targetPedestalSlot !== undefined) {
    const existingSlot = viewer.findSlotWithAsset(asset);
    if (existingSlot !== -1) {
      if (focusOnLoad && viewer.getMode() === 'showroom') {
        viewer.focusPedestal(existingSlot, true);
        updateFocusedPedestalUI(existingSlot);
        showToast(`Already displayed on Plinth #${existingSlot + 1} — focusing exhibit!`);
      }
      return false;
    }
  }

  setLoadingState(
    true,
    `Loading ${asset.title || asset.filename}...`,
    `Dataset: ${asset.repoId} (${formatBytes(asset.size)})`
  );

  const metaPromise = discovery.fetchDatasetMetadata(asset.repoId, asset.repoType);

  try {
    const { stats, slotIndex } = await viewer.loadAsset(
      asset,
      (pct) => {
        if (pct !== null) {
          loadingTitle.textContent = `Downloading ${asset.title || asset.filename} (${pct}%)...`;
        }
      },
      targetPedestalSlot,
      focusOnLoad
    );

    currentAsset = asset;
    const existingIdx = recentHistory.findIndex((a) => isSame3DObject(a, asset));
    if (existingIdx !== -1) {
      recentHistory.splice(existingIdx, 1);
    }
    recentHistory.push(asset);
    if (recentHistory.length > 4) {
      recentHistory.shift();
    }

    totalModelsLoaded++;
    // Gently rotate one of the 6 framed wall paintings every 4th model load
    if (wallArtPool.length > 0 && totalModelsLoaded % 4 === 0) {
      void viewer.applyWallArtworks(wallArtPool, true);
    }

    modelFormatBadge.textContent = asset.format.toUpperCase();
    metaFilePath.textContent = asset.path;

    if (asset.title) {
      metaObjectTitle.textContent = asset.title;
      metaTitleBlock.hidden = false;
      if (asset.subtitle) {
        metaObjectSubtitle.textContent = asset.subtitle;
        metaObjectSubtitle.hidden = false;
      } else {
        metaObjectSubtitle.hidden = true;
      }
    } else {
      metaTitleBlock.hidden = true;
    }

    statVertices.textContent = stats.vertices.toLocaleString();
    statTriangles.textContent = stats.triangles.toLocaleString();
    statMeshes.textContent = stats.meshes.toLocaleString();
    statSize.textContent = formatBytes(asset.size);

    metaDownloadLink.href = asset.downloadUrl;
    metaDownloadLink.hidden = false;
    metaHfTreeLink.href = asset.treeUrl;
    metaHfTreeLink.hidden = false;

    const datasetMeta = await metaPromise;
    viewer.setSlotDatasetMetadata(slotIndex, datasetMeta);
    renderDatasetMetadata(datasetMeta);

    if (viewer.getMode() === 'showroom') {
      if (focusOnLoad || slotIndex === viewer.getFocusedSlotIndex()) {
        updateFocusedPedestalUI(slotIndex);
      }

      // Non-blocking: Gemini multimodal visitor remarks (Temporarily commented out)
      /*
      window.setTimeout(() => {
        try {
          const snapshotBase64 = viewer.capturePedestalThumbnail(slotIndex);
          if (!snapshotBase64) return;

          void generateVisitorWhispersForExhibit({
            imageBase64Jpeg: snapshotBase64,
            assetId: asset.id,
          }).then((comments) => {
            if (comments && comments.length > 0) {
              console.log(
                `%c[Exhibit #${slotIndex + 1}] Gemini Comments Loaded for "${asset.title || asset.filename}":`,
                'color: #22c55e; font-weight: bold; font-size: 12px;',
                comments
              );
              viewer.setSlotCustomVisitorComments(slotIndex, comments);
              if (slotIndex === viewer.getFocusedSlotIndex()) {
                updateFocusedPedestalUI(slotIndex);
              }
            }
          });
        } catch (e) {
          console.warn('[Gemini Hook] Snapshot error:', e);
        }
      }, 500);
      */
    }

    setLoadingState(false);
    resetIdleTimer();
    renderPoolList();
    updateMemoryTelemetryUI();
    return true;
  } catch (err) {
    console.warn(`Failed to load 3D asset ${asset.id}:`, err);
    discovery.removeAsset(asset.id);
    setLoadingState(false);
    resetIdleTimer();
    updateMemoryTelemetryUI();
    return false;
  }
}

async function triggerBackgroundDiscoveryWave(): Promise<void> {
  if (isBackgroundCrawling) return;
  isBackgroundCrawling = true;

  try {
    await discovery.discoverMoreInBackground();
    updateStagedUI();
  } finally {
    isBackgroundCrawling = false;
  }
}

async function loadRandomModel(options?: {
  forceNewHubSearch?: boolean;
  sameRepoOnly?: boolean;
  targetPedestalSlot?: number;
  focusOnLoad?: boolean;
}): Promise<void> {
  if (isBusy) return;
  resetIdleTimer();

  const filters = getFilterOptions();
  let matchingPool = discovery.getPool(filters);

  if (options?.forceNewHubSearch) {
    setLoadingState(
      true,
      'Searching Hugging Face Datasets...',
      `Looking for 3D datasets matching "${filters.query || '3D'}"`
    );
    await discovery.discoverFromHub(filters, (status) => {
      loadingSubtitle.textContent = status;
    });
    matchingPool = discovery.getPool(filters);
    renderPoolList();
  } else if (matchingPool.length === 0) {
    setLoadingState(
      true,
      'Sampling Seed 3D Datasets...',
      'Fetching 3D object lists from curated Hugging Face datasets'
    );
    await discovery.initFromSeedDatasets(5, (status) => {
      loadingSubtitle.textContent = status;
    });
    matchingPool = discovery.getPool(filters);
    renderPoolList();
  }

  if (matchingPool.length === 0) {
    setLoadingState(false);
    showToast('No matching 3D files found for the current filter.');
    return;
  }

  const maxAttempts = 6;
  const triedCandidates: Discovered3DAsset[] = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const excluded = [...getActiveSetAssets(), ...triedCandidates];
    const candidate = discovery.pickRandom(
      filters,
      excluded,
      options?.sameRepoOnly ? currentAsset?.repoId : undefined,
      options?.sameRepoOnly ? undefined : currentAsset?.repoId
    );
    if (!candidate) break;
    triedCandidates.push(candidate);

    const ok = await loadSpecificAsset(
      candidate,
      options?.targetPedestalSlot,
      options?.focusOnLoad ?? true
    );
    if (ok) {
      if (discovery.getStagedCount() < 40) {
        void triggerBackgroundDiscoveryWave();
      }
      return;
    }
  }

  showToast('Could not parse the selected 3D file. Rolling another works!');
}

async function toggleViewerMode(): Promise<void> {
  if (isBusy) return;
  resetIdleTimer();

  const nextMode = viewer.getMode() === 'studio' ? 'showroom' : 'studio';
  viewer.setMode(nextMode);

  const isShowroom = nextMode === 'showroom';
  appRoot.classList.toggle('showroom-mode', isShowroom);
  showroomHud.hidden = !isShowroom;
  btnAudioToggle.hidden = !isShowroom;
  btnGrid.hidden = isShowroom;
  showroomCardsLayer.hidden = true;

  if (isShowroom) {
    modeToggleIcon.textContent = '🔬';
    modeToggleText.textContent = 'Return to Studio View';

    // Sync any recent models loaded while in Studio Mode onto the pedestals first
    for (const histAsset of [...recentHistory]) {
      if (viewer.findSlotWithAsset(histAsset) === -1) {
        await loadSpecificAsset(histAsset, undefined, false);
      }
    }

    // Populate any remaining empty pedestals so all 4 museum plinths have unique exhibits!
    for (let slotIdx = 0; slotIdx < 4; slotIdx++) {
      const info = viewer.getSlotInfo(slotIdx);
      if (!info?.asset) {
        await loadRandomModel({ targetPedestalSlot: slotIdx, focusOnLoad: false });
      }
    }
    const latestSlotIdx = viewer.getFocusedSlotIndex();
    viewer.focusPedestal(latestSlotIdx, true);
    updateFocusedPedestalUI(latestSlotIdx);
    showToast('Showroom Mode: Walk with WASD / Arrows or click any pedestal (#1–#4) to inspect!');
  } else {
    modeToggleIcon.textContent = '🏛️';
    modeToggleText.textContent = 'Showroom View (4 Models)';
    // If studio doesn't have a model loaded yet, load one
    if (currentAsset) {
      await loadSpecificAsset(currentAsset);
    }
  }
}

async function handleRefreshPool(): Promise<void> {
  if (isBusy) return;
  resetIdleTimer();

  if (discovery.getStagedCount() === 0) {
    btnRefreshPool.disabled = true;
    bgDiscoveryStatus.textContent = 'Scanning Hugging Face for more 3D datasets...';
    await triggerBackgroundDiscoveryWave();
    btnRefreshPool.disabled = false;
  }

  const added = discovery.mergeStagedIntoPool();
  renderPoolList();

  if (added > 0) {
    showToast(
      `Refreshed pool with +${added} new 3D models (${discovery.getActiveDatasetCount()} datasets active)!`
    );
    void triggerBackgroundDiscoveryWave();
  } else {
    discovery.shufflePool();
    renderPoolList();
    showToast('Pool reshuffled! Scanning for more datasets in the background...');
    void triggerBackgroundDiscoveryWave();
  }
}

// Event Listeners
btnToggleMode.addEventListener('click', () => {
  void toggleViewerMode();
});

btnShowroomRoll.addEventListener('click', () => {
  void loadRandomModel();
});

for (const btn of pedestalJumpButtons) {
  btn.addEventListener('click', () => {
    const slotIdx = Number(btn.dataset.slot);
    viewer.focusPedestal(slotIdx, true);
    updateFocusedPedestalUI(slotIdx);
  });
}

btnRandom.addEventListener('click', () => {
  void loadRandomModel();
});

btnRefreshPool.addEventListener('click', () => {
  void handleRefreshPool();
});

searchForm.addEventListener('submit', (e) => {
  e.preventDefault();
  void loadRandomModel({ forceNewHubSearch: true });
});

repoTypeSelect.addEventListener('change', () => {
  renderPoolList();
  void loadRandomModel();
});

formatSelect.addEventListener('change', () => {
  renderPoolList();
  void loadRandomModel();
});

// Viewport Toolbar Buttons
btnIdleCycle.addEventListener('click', () => {
  idleCycleEnabled = !idleCycleEnabled;
  resetIdleTimer();
  btnIdleCycle.classList.toggle('active', idleCycleEnabled);
  btnIdleCycle.setAttribute('aria-pressed', String(idleCycleEnabled));
});

btnAudioToggle.addEventListener('click', () => {
  const next = !viewer.getAudioEnabled();
  viewer.setAudioEnabled(next);
  btnAudioToggle.classList.toggle('active', next);
  btnAudioToggle.setAttribute('aria-pressed', String(next));
  audioToggleLabel.textContent = next ? '🔊 Audio' : '🔇 Muted';
});

btnAutorotate.addEventListener('click', () => {
  const next = !viewer.getAutoRotate();
  viewer.setAutoRotate(next);
  btnAutorotate.classList.toggle('active', next);
  btnAutorotate.setAttribute('aria-pressed', String(next));
});

btnWireframe.addEventListener('click', () => {
  const next = !viewer.getWireframe();
  viewer.setWireframe(next);
  btnWireframe.classList.toggle('active', next);
  btnWireframe.setAttribute('aria-pressed', String(next));
});

btnGrid.addEventListener('click', () => {
  const next = !viewer.getGridVisible();
  viewer.setGridVisible(next);
  btnGrid.classList.toggle('active', next);
  btnGrid.setAttribute('aria-pressed', String(next));
});

btnFlipUp.addEventListener('click', () => {
  viewer.flipUpAxis();
});

btnResetCam.addEventListener('click', () => {
  viewer.resetCameraView();
});

// Reset the idle countdown on user interaction in Studio Mode
// (In Showroom Mode, let the 30s exhibit rotation keep ticking while you walk around!)
const ACTIVITY_EVENTS: Array<keyof WindowEventMap> = [
  'pointerdown',
  'pointermove',
  'wheel',
  'touchstart',
  'keydown',
];
for (const evtName of ACTIVITY_EVENTS) {
  window.addEventListener(
    evtName,
    () => {
      if (evtName === 'pointerdown' || evtName === 'keydown' || evtName === 'touchstart') {
        viewer.unlockAudio();
      }
      if (viewer.getMode() === 'studio') {
        resetIdleTimer();
      }
    },
    { passive: true }
  );
}

// 1-second tick for Idle Auto-Advance & Memory Telemetry
window.setInterval(() => {
  updateMemoryTelemetryUI();
  if (!idleCycleEnabled || isBusy || document.hidden) return;

  idleSecondsRemaining = Math.max(0, idleSecondsRemaining - 1);
  updateIdleButtonUI();

  if (idleSecondsRemaining === 0) {
    idleSecondsRemaining = IDLE_TIMEOUT_SECONDS;
    if (discovery.getStagedCount() >= 20) {
      discovery.mergeStagedIntoPool();
    }
    const inShowroom = viewer.getMode() === 'showroom';
    void loadRandomModel({ focusOnLoad: !inShowroom });
  }
}, 1000);

// Keyboard shortcuts when not typing in an input
window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement | null)?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

  const inShowroom = viewer.getMode() === 'showroom';

  if (e.code === 'Space') {
    e.preventDefault();
    void loadRandomModel();
  } else if (e.key === 'm' || e.key === 'M') {
    void toggleViewerMode();
  } else if (e.key === 'i' || e.key === 'I') {
    btnIdleCycle.click();
  } else if (!inShowroom && (e.key === 'a' || e.key === 'A')) {
    // In showroom mode, W/A/S/D are used for walking!
    btnAutorotate.click();
  } else if (!inShowroom && (e.key === 'w' || e.key === 'W')) {
    btnWireframe.click();
  } else if (e.key === 'g' || e.key === 'G') {
    btnGrid.click();
  } else if (e.key === 'u' || e.key === 'U') {
    btnFlipUp.click();
  } else if (e.key === 'c' || e.key === 'C') {
    btnResetCam.click();
  } else if (inShowroom && ['1', '2', '3', '4'].includes(e.key)) {
    const slotIdx = Number(e.key) - 1;
    viewer.focusPedestal(slotIdx, true);
    updateFocusedPedestalUI(slotIdx);
  }
});

// Start from seed datasets and kick off background discovery + wall paintings
async function bootstrap(): Promise<void> {
  updateIdleButtonUI();
  void discovery.fetchWallArtworks().then((artworks) => {
    wallArtPool = artworks;
    void viewer.applyWallArtworks(wallArtPool, false);
  });
  await loadRandomModel();
  window.setTimeout(() => {
    void triggerBackgroundDiscoveryWave();
  }, 2500);
}

void bootstrap();
