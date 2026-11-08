import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { clone as cloneSkeleton } from 'three/examples/jsm/utils/SkeletonUtils.js';
import {
  isSame3DObject,
  type DatasetMetadata,
  type Discovered3DAsset,
  type WallArtworkItem,
} from './hf';
import { SpatialAudioManager } from './audio';
import {
  generateSculptureNameForDais,
  pickFallbackSculptureName,
} from './gemini-visitor';

export type ViewerMode = 'studio' | 'showroom';
export type LodTier = 'close' | 'medium' | 'far';
export type DaisAssemblyMode = 'collision' | 'totem';
export type DaisFinishMode = 'auto-patina' | 'bronze' | 'chrome' | 'marble';

interface ProceduralPreset {
  map: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
  color: number;
  metalness: number;
  roughness: number;
  bumpScale: number;
}

export interface LoadedModelStats {
  vertices: number;
  triangles: number;
  meshes: number;
  scaleFactor?: number;
  scaleLabel?: string;
}

export interface ViewerMemoryStats {
  gpuGeometries: number;
  gpuTextures: number;
  jsHeapMB: number | null;
}

export interface PedestalSlotState {
  slotIndex: number;
  asset: Discovered3DAsset | null;
  datasetMeta: DatasetMetadata | null;
  stats: LoadedModelStats | null;
  lodTier: LodTier;
  distance: number;
  screenX: number;
  screenY: number;
  visibleOnScreen: boolean;
  isFocused: boolean;
}

interface InternalPedestalSlot {
  index: number;
  position: THREE.Vector3;
  plinthGroup: THREE.Group;
  modelPivot: THREE.Group;
  placardAnchor: THREE.Object3D;
  lecternScreenMesh: THREE.Mesh;
  ringMesh: THREE.Mesh;
  contactShadowMesh: THREE.Mesh;
  bayLight: THREE.PointLight;
  lightCone: THREE.Mesh;
  lightConeMat: THREE.MeshBasicMaterial;
  haloRippleMesh: THREE.Mesh;
  haloRippleMat: THREE.MeshBasicMaterial;
  smokePuffPoints: THREE.Points;
  smokePuffMat: THREE.PointsMaterial;
  smokeSeeds: Float32Array;
  dustMotesPoints: THREE.Points;
  dustMotesMat: THREE.PointsMaterial;
  dustSeeds: Float32Array;
  revealTimer: number;
  whisperedForCurrentAsset: boolean;
  customVisitorComments: string[];
  commentPlayIndex: number;
  modelRoot: THREE.Object3D | null;
  mixer: THREE.AnimationMixer | null;
  asset: Discovered3DAsset | null;
  datasetMeta: DatasetMetadata | null;
  stats: LoadedModelStats | null;
  lodTier: LodTier;
  originalMaterials: Map<THREE.Mesh, THREE.Material | THREE.Material[]>;
  loadedAt: number;
}

interface InternalDaisSlot {
  index: number;
  position: THREE.Vector3;
  daisGroup: THREE.Group;
  lecternGroup: THREE.Group;
  modelPivot: THREE.Group;
  lecternScreenMesh: THREE.Mesh;
  ringMesh: THREE.Mesh;
  contactShadowMesh: THREE.Mesh;
  daisLight: THREE.PointLight;
  lightCone: THREE.Mesh;
  lightConeMat: THREE.MeshBasicMaterial;
  haloRippleMesh: THREE.Mesh;
  haloRippleMat: THREE.MeshBasicMaterial;
  smokePuffPoints: THREE.Points;
  smokePuffMat: THREE.PointsMaterial;
  smokeSeeds: Float32Array;
  dustMotesPoints: THREE.Points;
  dustMotesMat: THREE.PointsMaterial;
  dustSeeds: Float32Array;
  revealTimer: number;
  modelRoot: THREE.Object3D | null;
  clonedMaterials: THREE.Material[];
  sourceSlotIndices: number[];
  sourceAssets: Discovered3DAsset[];
  stats: LoadedModelStats | null;
  sculptureName: string | null;
  sculptureNameIsGemini: boolean;
  isNamingSculpture: boolean;
  namingRequestId: number;
  collisionCount: number;
  combinationCursor: number;
}

const REVEAL_DURATION = 1.65;

// Spacious 4-bay gallery promenade arc: wide enough (~14.4m span) for all 3 LOD tiers
// (Close < 5.8m, Medium 5.8–11.2m, Far > 11.2m) to transition naturally as you walk!
const PEDESTAL_POSITIONS: THREE.Vector3[] = [
  new THREE.Vector3(-7.2, 0, -1.8),
  new THREE.Vector3(-2.4, 0, -3.5),
  new THREE.Vector3(2.4, 0, -3.5),
  new THREE.Vector3(7.2, 0, -1.8),
];

const PLINTH_HEIGHT = 1.05;
export const DAIS_SLOT_INDEX = 4;
const DAIS_POSITION = new THREE.Vector3(0, 0, 1.05);
const DAIS_HEIGHT = 0.62;

export class ThreeModelViewer {
  private canvas: HTMLCanvasElement;
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private studioFloor: THREE.Group;
  private studioContactShadow: THREE.Mesh;

  // Studio single-model group
  private studioPivot: THREE.Group;
  private currentModelRoot: THREE.Object3D | null = null;
  private studioMixer: THREE.AnimationMixer | null = null;

  // Showroom 4-pedestal environment + Central Exquisite Corpse Dais
  private showroomGroup: THREE.Group;
  private slots: InternalPedestalSlot[] = [];
  private daisSlot!: InternalDaisSlot;
  private wallPaintingMeshes: THREE.Mesh[] = [];
  private nextWallFrameIndex = 0;
  private activeArtIds = new Set<string>(); // Tracks currently displayed artworks to guarantee zero repeats
  private recentArtHistory: string[] = []; // Long FIFO queue preventing re-display of the same artwork for a long cycle
  private artCursor = 0;
  private focusedSlotIndex = 0;
  private nextReplaceSlotIndex = 0;
  private newShowroomModelsSinceDaisRebuild = 0;
  private farProxyMaterial: THREE.MeshBasicMaterial;
  private mediumProxyMaterial: THREE.MeshLambertMaterial;

  // Camera smooth glide target
  private camGlideActive = false;
  private camGlideGoalPos = new THREE.Vector3();
  private camGlideGoalTarget = new THREE.Vector3();

  // Keyboard walking state (WASD / Arrows in Showroom Mode)
  private moveKeys = {
    forward: false,
    backward: false,
    left: false,
    right: false,
  };
  private audio = new SpatialAudioManager(PEDESTAL_POSITIONS);
  private stepDistanceAccum = 0;
  private wasWalkingLastFrame = false;
  private galleryWhisperTimer = 0;
  private nextGalleryWhisperInterval = 28; // Dynamic randomized interval (28s to 48s)

  private mode: ViewerMode = 'studio';
  private onShowroomProjectionUpdate?: (states: PedestalSlotState[]) => void;
  private onPedestalSelect?: (slotIndex: number) => void;
  private onDaisSculptureNamed?: (name: string) => void;
  private daisSnapshotCamera = new THREE.PerspectiveCamera(38, 1, 0.1, 40);
  private daisAssemblyMode: DaisAssemblyMode = 'totem';
  private daisFinishMode: DaisFinishMode = 'auto-patina';
  private proceduralPresets: Record<'bronze' | 'verdigris' | 'chrome' | 'marble', ProceduralPreset> | null =
    null;

  private clock = new THREE.Clock();
  private gltfLoader: GLTFLoader;
  private dracoLoader: DRACOLoader;
  private objLoader: OBJLoader;
  private stlLoader: STLLoader;

  private activeAbortController: AbortController | null = null;
  private slotAbortControllers: Array<AbortController | null> = [null, null, null, null];
  private wireframeEnabled = false;
  private autoRotateEnabled = true;
  private studioGridPreferred = true;
  private resizeObserver: ResizeObserver;
  private raycaster = new THREE.Raycaster();
  private pointerDownPos = new THREE.Vector2();

  constructor(canvas: HTMLCanvasElement, container: HTMLElement) {
    this.canvas = canvas;
    this.container = container;

    THREE.Cache.enabled = false;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();

    // Studio environment lighting via PMREM + RoomEnvironment
    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    pmremGenerator.compileEquirectangularShader();
    const roomEnv = new RoomEnvironment();
    const envRenderTarget = pmremGenerator.fromScene(roomEnv, 0.04);
    this.scene.environment = envRenderTarget.texture;
    roomEnv.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.geometry?.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((m) => m.dispose());
        } else {
          mesh.material?.dispose();
        }
      }
    });
    pmremGenerator.dispose();

    // Shared lightweight LOD proxy materials for distant showroom models
    this.mediumProxyMaterial = new THREE.MeshLambertMaterial({
      color: 0xd6d3d1,
      side: THREE.DoubleSide,
    });
    this.farProxyMaterial = new THREE.MeshBasicMaterial({
      color: 0xfbbf24,
      wireframe: true,
      transparent: true,
      opacity: 0.42,
    });

    // Lights
    const ambientLight = new THREE.AmbientLight(0xfff5e6, 0.55);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff4e0, 1.75);
    keyLight.position.set(4, 8, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.75);
    rimLight.position.set(-5, 4, -4);
    this.scene.add(rimLight);

    // Warm colored matte Studio Floor plane (replaces the cold techy wireframe grid)
    this.studioFloor = new THREE.Group();
    const floorTex = this.createStudioFloorTexture();
    const floorDisc = new THREE.Mesh(
      new THREE.CircleGeometry(5.6, 72),
      new THREE.MeshStandardMaterial({
        map: floorTex,
        roughness: 0.72,
        metalness: 0.04,
        transparent: true,
      })
    );
    floorDisc.rotation.x = -Math.PI / 2;
    floorDisc.receiveShadow = true;
    this.studioFloor.add(floorDisc);

    // Subtle warm brass inlay ring framing the inner turntable stage
    const stageRing = new THREE.Mesh(
      new THREE.RingGeometry(1.36, 1.39, 64),
      new THREE.MeshBasicMaterial({
        color: 0xd4a359,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.55,
      })
    );
    stageRing.rotation.x = -Math.PI / 2;
    stageRing.position.y = 0.002;
    this.studioFloor.add(stageRing);

    // Soft ambient contact shadow beneath the Studio model
    this.studioContactShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(2.2, 2.2),
      new THREE.MeshBasicMaterial({
        map: this.createContactShadowTexture(),
        transparent: true,
        depthWrite: false,
        opacity: 0.68,
      })
    );
    this.studioContactShadow.rotation.x = -Math.PI / 2;
    this.studioContactShadow.position.y = 0.003;
    this.studioContactShadow.visible = false;
    this.studioFloor.add(this.studioContactShadow);

    this.scene.add(this.studioFloor);

    // Studio single-model pivot
    this.studioPivot = new THREE.Group();
    this.scene.add(this.studioPivot);

    // Showroom group (hidden in studio mode)
    this.showroomGroup = new THREE.Group();
    this.showroomGroup.visible = false;
    this.scene.add(this.showroomGroup);
    this.buildShowroomEnvironment();

    // Camera & OrbitControls
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(48, width / height, 0.01, 120);
    this.camera.position.set(2.8, 2.1, 3.4);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.6;
    this.controls.target.set(0, 0.8, 0);
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 20;

    // Cancel camera auto-glide if user manually orbits or zooms
    this.controls.addEventListener('start', () => {
      this.camGlideActive = false;
    });

    // Loaders setup
    this.dracoLoader = new DRACOLoader();
    this.dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
    this.gltfLoader = new GLTFLoader();
    this.gltfLoader.setDRACOLoader(this.dracoLoader);
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);

    this.objLoader = new OBJLoader();
    this.stlLoader = new STLLoader();

    this.setupShowroomInteractions();

    this.updateSize();
    this.resizeObserver = new ResizeObserver(() => this.updateSize());
    this.resizeObserver.observe(this.container);

    this.renderer.setAnimationLoop(() => this.renderTick());
  }

  /**
   * Generates a warm procedural oak wood plank texture so the showroom floor feels warm and cozy
   * with zero external texture downloads.
   */
  private createWoodFloorTexture(): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 512;
    const ctx = c.getContext('2d')!;

    ctx.fillStyle = '#3b281a';
    ctx.fillRect(0, 0, 512, 512);

    const planks = 8;
    const plankH = 512 / planks;
    const tones = ['#463020', '#3d291b', '#4d3524', '#382518', '#422c1d'];

    for (let row = 0; row < planks; row++) {
      const y = row * plankH;
      const splits = [0, ((row * 173) % 280) + 110, 512];
      for (let s = 0; s < splits.length - 1; s++) {
        const x0 = splits[s];
        const w = splits[s + 1] - x0;
        ctx.fillStyle = tones[(row + s) % tones.length];
        ctx.fillRect(x0 + 1, y + 1, w - 2, plankH - 2);

        // Subtle wood grain lines
        ctx.strokeStyle = 'rgba(20, 12, 6, 0.18)';
        ctx.lineWidth = 1;
        for (let g = 0; g < 6; g++) {
          const gy = y + ((g + 1) * plankH) / 7;
          ctx.beginPath();
          ctx.moveTo(x0 + 2, gy);
          ctx.lineTo(x0 + w - 2, gy + ((g % 2 === 0 ? 1 : -1) * 1.5));
          ctx.stroke();
        }
      }
    }

    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(9, 6);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  /**
   * Generates a warm, colored matte studio floor texture (warm sandstone/terracotta center
   * transitioning into rich museum slate with a smooth feathered outer edge).
   */
  private createStudioFloorTexture(): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 512;
    const ctx = c.getContext('2d')!;

    const grad = ctx.createRadialGradient(256, 256, 18, 256, 256, 252);
    grad.addColorStop(0, 'rgba(196, 164, 132, 0.96)'); // Warm sandstone / buff center
    grad.addColorStop(0.32, 'rgba(158, 123, 96, 0.92)'); // Warm clay / walnut-ochre mid-stage
    grad.addColorStop(0.68, 'rgba(92, 74, 64, 0.78)'); // Cozy warm umber surround
    grad.addColorStop(0.9, 'rgba(48, 41, 39, 0.35)');
    grad.addColorStop(1, 'rgba(30, 27, 26, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 512);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  /**
   * Generates a soft radial contact shadow texture for placing directly under each 3D exhibit.
   */
  private createContactShadowTexture(): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const ctx = c.getContext('2d')!;
    const grad = ctx.createRadialGradient(64, 64, 8, 64, 64, 60);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.55)');
    grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.22)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  /**
   * Generates a soft radial glow/puff sprite texture shared by the plinth smoke puff and golden dust motes.
   */
  private createSoftParticleTexture(): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 64;
    c.height = 64;
    const ctx = c.getContext('2d')!;
    const grad = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
    grad.addColorStop(0, 'rgba(255, 250, 235, 1)');
    grad.addColorStop(0.4, 'rgba(255, 235, 190, 0.55)');
    grad.addColorStop(1, 'rgba(255, 220, 160, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }

  /**
   * Builds a warm, architectural 4-bay museum hall (20.8m × 13.2m): spacious enough for
   * 3-tier distance LOD (< 5.8m Close, 5.8–11.2m Medium, > 11.2m Far) to work across the room,
   * while staying cozy with walnut bay pilasters, oak parquet floors, a long promenade rug,
   * twin tufted leather benches, wood-slat exhibit alcoves, framed paintings, and warm sconces.
   */
  private buildShowroomEnvironment(): void {
    const roomWidth = 20.8;
    const roomDepth = 13.2;
    const wallHeight = 5.1;
    const backZ = -5.6;
    const frontZ = 7.6;
    const centerZ = (backZ + frontZ) / 2;

    // 1. Warm Oak Parquet Floor
    const woodTex = this.createWoodFloorTexture();
    const floorGeo = new THREE.PlaneGeometry(roomWidth, roomDepth);
    const floorMat = new THREE.MeshStandardMaterial({
      map: woodTex,
      roughness: 0.52,
      metalness: 0.08,
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.set(0, 0, centerZ);
    floorMesh.receiveShadow = true;
    this.showroomGroup.add(floorMesh);

    // 2. Long Woven Gallery Promenade Rug + Brass Border
    const rugBorderGeo = new THREE.PlaneGeometry(14.6, 4.8);
    const rugBorderMat = new THREE.MeshStandardMaterial({
      color: 0x78552b,
      roughness: 0.7,
      metalness: 0.25,
    });
    const rugBorder = new THREE.Mesh(rugBorderGeo, rugBorderMat);
    rugBorder.rotation.x = -Math.PI / 2;
    rugBorder.position.set(0, 0.004, 1.05);
    rugBorder.receiveShadow = true;
    this.showroomGroup.add(rugBorder);

    const rugGeo = new THREE.PlaneGeometry(14.2, 4.4);
    const rugMat = new THREE.MeshStandardMaterial({
      color: 0x283142,
      roughness: 0.92,
      metalness: 0.02,
    });
    const rugMesh = new THREE.Mesh(rugGeo, rugMat);
    rugMesh.rotation.x = -Math.PI / 2;
    rugMesh.position.set(0, 0.008, 1.05);
    rugMesh.receiveShadow = true;
    this.showroomGroup.add(rugMesh);

    // 3. Twin Tufted Leather Museum Benches along the Promenade Rug (leaving a clear walk aisle)
    const cushionGeo = new THREE.BoxGeometry(2.3, 0.16, 0.78);
    const leatherMat = new THREE.MeshStandardMaterial({
      color: 0x6b341e,
      roughness: 0.45,
      metalness: 0.08,
    });
    const frameGeo = new THREE.BoxGeometry(2.34, 0.05, 0.82);
    const walnutMat = new THREE.MeshStandardMaterial({
      color: 0x2b1b11,
      roughness: 0.55,
      metalness: 0.1,
    });
    const legGeo = new THREE.CylinderGeometry(0.035, 0.025, 0.34, 12);
    const brassMat = new THREE.MeshStandardMaterial({
      color: 0xd4a359,
      roughness: 0.32,
      metalness: 0.78,
    });

    for (const benchX of [-5.2, 5.2]) {
      const benchGroup = new THREE.Group();
      benchGroup.position.set(benchX, 0, 5.75);

      const cushion = new THREE.Mesh(cushionGeo, leatherMat);
      cushion.position.y = 0.44;
      cushion.castShadow = true;
      cushion.receiveShadow = true;
      benchGroup.add(cushion);

      const benchFrame = new THREE.Mesh(frameGeo, walnutMat);
      benchFrame.position.y = 0.34;
      benchGroup.add(benchFrame);

      for (const lx of [-1.0, 1.0]) {
        for (const lz of [-0.3, 0.3]) {
          const leg = new THREE.Mesh(legGeo, brassMat);
          leg.position.set(lx, 0.17, lz);
          leg.castShadow = true;
          benchGroup.add(leg);
        }
      }
      this.showroomGroup.add(benchGroup);
    }

    // 4. Cozy Museum Walls (Warm Sage-Charcoal Upper Wall + Lower Walnut Wainscoting + Brass Chair Rail)
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x1e2530,
      roughness: 0.85,
      metalness: 0.05,
    });
    const wainscotHeight = 1.2;

    const addWallSegment = (
      w: number,
      x: number,
      z: number,
      rotY: number
    ) => {
      const segGroup = new THREE.Group();
      segGroup.position.set(x, 0, z);
      segGroup.rotation.y = rotY;

      const upperWall = new THREE.Mesh(
        new THREE.PlaneGeometry(w, wallHeight),
        wallMat
      );
      upperWall.position.y = wallHeight / 2;
      upperWall.receiveShadow = true;
      segGroup.add(upperWall);

      const lowerPanel = new THREE.Mesh(
        new THREE.BoxGeometry(w, wainscotHeight, 0.06),
        walnutMat
      );
      lowerPanel.position.set(0, wainscotHeight / 2, 0.03);
      lowerPanel.receiveShadow = true;
      segGroup.add(lowerPanel);

      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.038, 0.08),
        brassMat
      );
      rail.position.set(0, wainscotHeight, 0.04);
      segGroup.add(rail);

      const crown = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.16, 0.18),
        walnutMat
      );
      crown.position.set(0, wallHeight - 0.08, 0.09);
      segGroup.add(crown);

      this.showroomGroup.add(segGroup);
    };

    addWallSegment(roomWidth, 0, backZ, 0);
    addWallSegment(roomDepth, -roomWidth / 2, centerZ, Math.PI / 2);
    addWallSegment(roomDepth, roomWidth / 2, centerZ, -Math.PI / 2);
    addWallSegment(roomWidth, 0, frontZ, Math.PI);

    // Architectural Walnut Pilasters along the back wall to frame the outer gallery wings
    const pilasterGeo = new THREE.BoxGeometry(0.42, wallHeight, 0.32);
    const pilasterBrassGeo = new THREE.BoxGeometry(0.46, 0.06, 0.36);
    for (const pilX of [-9.8, 9.8]) {
      const pilaster = new THREE.Mesh(pilasterGeo, walnutMat);
      pilaster.position.set(pilX, wallHeight / 2, backZ + 0.16);
      pilaster.castShadow = true;
      pilaster.receiveShadow = true;
      this.showroomGroup.add(pilaster);

      const capRing = new THREE.Mesh(pilasterBrassGeo, brassMat);
      capRing.position.set(pilX, wainscotHeight, backZ + 0.16);
      this.showroomGroup.add(capRing);
    }

    // 5. Overhead Timber Ceiling Beams & Warm Gallery Lighting across the 4 Bays
    const beamGeo = new THREE.BoxGeometry(roomWidth, 0.2, 0.24);
    for (const bz of [-3.8, -1.2, 1.4, 4.0, 6.4]) {
      const beam = new THREE.Mesh(beamGeo, walnutMat);
      beam.position.set(0, wallHeight - 0.1, bz);
      this.showroomGroup.add(beam);
    }

    const bayLights: THREE.PointLight[] = [];
    for (let i = 0; i < 4; i++) {
      const pedPos = PEDESTAL_POSITIONS[i];
      const bayLight = new THREE.PointLight(0xffdca8, 2.2, 12, 1.35);
      bayLight.position.set(pedPos.x, 4.4, pedPos.z + 0.3);
      this.showroomGroup.add(bayLight);
      bayLights.push(bayLight);
    }

    const loungeLight = new THREE.PointLight(0xffe4b5, 2.5, 15, 1.35);
    loungeLight.position.set(DAIS_POSITION.x, 4.4, DAIS_POSITION.z + 0.25);
    this.showroomGroup.add(loungeLight);

    // 6. Decorative Acoustic Wood-Slat Feature Panels & Brass Picture Lights behind each of the 4 Bays
    const slatMat = new THREE.MeshStandardMaterial({
      color: 0x5c3a21,
      roughness: 0.55,
      metalness: 0.08,
    });
    for (const px of [-7.2, -2.4, 2.4, 7.2]) {
      const panelGroup = new THREE.Group();
      panelGroup.position.set(px, 2.85, backZ + 0.05);

      const backing = new THREE.Mesh(
        new THREE.BoxGeometry(2.1, 2.5, 0.03),
        new THREE.MeshStandardMaterial({ color: 0x11151c, roughness: 0.9 })
      );
      panelGroup.add(backing);

      const slatGeo = new THREE.BoxGeometry(0.065, 2.44, 0.05);
      for (let s = -8; s <= 8; s++) {
        const slat = new THREE.Mesh(slatGeo, slatMat);
        slat.position.set(s * 0.12, 0, 0.02);
        panelGroup.add(slat);
      }

      const sconceBar = new THREE.Mesh(
        new THREE.CylinderGeometry(0.022, 0.022, 0.68, 12),
        brassMat
      );
      sconceBar.rotation.z = Math.PI / 2;
      sconceBar.position.set(0, 1.35, 0.14);
      panelGroup.add(sconceBar);

      this.showroomGroup.add(panelGroup);
    }

    // Framed Open-Source Gallery Paintings on Back Wall (between bays) & Side Walls
    const addFramedCanvas = (
      x: number,
      z: number,
      rotY: number,
      canvasColor: number
    ) => {
      const artGroup = new THREE.Group();
      artGroup.position.set(x, 2.72, z);
      artGroup.rotation.y = rotY;

      const frame = new THREE.Mesh(
        new THREE.BoxGeometry(1.9, 1.3, 0.06),
        brassMat
      );
      artGroup.add(frame);

      const matBoard = new THREE.Mesh(
        new THREE.BoxGeometry(1.76, 1.16, 0.07),
        new THREE.MeshStandardMaterial({ color: 0xf5f5f4, roughness: 0.85 })
      );
      artGroup.add(matBoard);

      const artCanvas = new THREE.Mesh(
        new THREE.BoxGeometry(1.44, 0.9, 0.075),
        new THREE.MeshStandardMaterial({ color: canvasColor, roughness: 0.65 })
      );
      artGroup.add(artCanvas);

      // Front surface plane where open-source HF paintings/specimens are rendered
      const paintingPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(1.42, 0.88),
        new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.48,
          metalness: 0.04,
        })
      );
      paintingPlane.position.z = 0.042;
      artGroup.add(paintingPlane);
      this.wallPaintingMeshes.push(paintingPlane);
      this.applyProceduralArtworkFallback(paintingPlane, this.wallPaintingMeshes.length - 1);

      this.showroomGroup.add(artGroup);
    };

    // 3 Framed Paintings along the Front/Bay Wall between the 4 Pedestals
    addFramedCanvas(-4.8, backZ + 0.05, 0, 0x1e3a5f);
    addFramedCanvas(0, backZ + 0.05, 0, 0x3f2e21);
    addFramedCanvas(4.8, backZ + 0.05, 0, 0x2d3b2d);

    // 4 Framed Paintings along the Left & Right Promenade Wings
    addFramedCanvas(-roomWidth / 2 + 0.04, -1.6, Math.PI / 2, 0x1e3a5f);
    addFramedCanvas(-roomWidth / 2 + 0.04, 2.2, Math.PI / 2, 0x3f2e21);
    addFramedCanvas(roomWidth / 2 - 0.04, -1.6, -Math.PI / 2, 0x2d3b2d);
    addFramedCanvas(roomWidth / 2 - 0.04, 2.2, -Math.PI / 2, 0x3b232e);

    // 5 Framed Paintings along the Back Entrance/Promenade Wall facing the pedestals
    addFramedCanvas(-7.2, frontZ - 0.05, Math.PI, 0x2a3d45);
    addFramedCanvas(-3.6, frontZ - 0.05, Math.PI, 0x3d2a35);
    addFramedCanvas(0, frontZ - 0.05, Math.PI, 0x243328);
    addFramedCanvas(3.6, frontZ - 0.05, Math.PI, 0x3d3524);
    addFramedCanvas(7.2, frontZ - 0.05, Math.PI, 0x2b253d);

    // 7. Cozy Planters with Sculptural Greenery flanking the Gallery Bays
    const addCornerPlant = (x: number, z: number) => {
      const plantGroup = new THREE.Group();
      plantGroup.position.set(x, 0, z);

      const pot = new THREE.Mesh(
        new THREE.CylinderGeometry(0.38, 0.28, 0.72, 24),
        new THREE.MeshStandardMaterial({ color: 0x9a5b3e, roughness: 0.68 })
      );
      pot.position.y = 0.36;
      pot.castShadow = true;
      plantGroup.add(pot);

      const foliageMat = new THREE.MeshStandardMaterial({
        color: 0x2d5a3c,
        roughness: 0.65,
      });
      const leafPositions: Array<[number, number, number, number]> = [
        [0, 0.92, 0, 0.36],
        [-0.2, 1.18, 0.12, 0.3],
        [0.2, 1.1, -0.1, 0.28],
        [0.05, 1.38, 0.05, 0.26],
        [-0.12, 0.98, -0.16, 0.26],
      ];
      for (const [lx, ly, lz, r] of leafPositions) {
        const sphere = new THREE.Mesh(
          new THREE.DodecahedronGeometry(r, 1),
          foliageMat
        );
        sphere.position.set(lx, ly, lz);
        sphere.castShadow = true;
        plantGroup.add(sphere);
      }

      this.showroomGroup.add(plantGroup);
    };

    addCornerPlant(-9.1, -4.6);
    addCornerPlant(-4.8, -4.7);
    addCornerPlant(4.8, -4.7);
    addCornerPlant(9.1, -4.6);

    // 8. Build the 4 Museum Pedestals with Warm Walnut/Stone Plinths, Brass Trim, Spotlight Cones & Contact Shadows
    const baseGeo = new THREE.CylinderGeometry(0.76, 0.84, PLINTH_HEIGHT, 36);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x242c3a,
      roughness: 0.42,
      metalness: 0.18,
    });
    const collarGeo = new THREE.CylinderGeometry(0.8, 0.8, 0.045, 36);
    const capGeo = new THREE.CylinderGeometry(0.78, 0.78, 0.05, 36);
    const capMat = new THREE.MeshStandardMaterial({
      color: 0xf5f0e6,
      roughness: 0.28,
      metalness: 0.08,
    });
    const ringGeo = new THREE.RingGeometry(0.74, 0.79, 48);
    const shadowTex = this.createContactShadowTexture();
    const shadowPlaneGeo = new THREE.PlaneGeometry(1.25, 1.25);

    // Soft volumetric light cone above each pedestal + shared radial particle sprite
    const coneGeo = new THREE.ConeGeometry(0.95, 2.9, 32, 1, true);
    const haloRippleGeo = new THREE.RingGeometry(0.68, 0.79, 48);
    const particleTex = this.createSoftParticleTexture();

    for (let i = 0; i < 4; i++) {
      const pos = PEDESTAL_POSITIONS[i];
      const plinthGroup = new THREE.Group();
      plinthGroup.position.copy(pos);

      const plinthMesh = new THREE.Mesh(baseGeo, baseMat);
      plinthMesh.position.y = PLINTH_HEIGHT / 2;
      plinthMesh.receiveShadow = true;
      plinthMesh.castShadow = true;
      plinthMesh.userData.slotIndex = i;
      plinthGroup.add(plinthMesh);

      const brassCollar = new THREE.Mesh(collarGeo, brassMat);
      brassCollar.position.y = PLINTH_HEIGHT - 0.03;
      brassCollar.userData.slotIndex = i;
      plinthGroup.add(brassCollar);

      // Warm cream marble top surface so objects stand out clearly
      const capMesh = new THREE.Mesh(capGeo, capMat);
      capMesh.position.y = PLINTH_HEIGHT + 0.025;
      capMesh.receiveShadow = true;
      capMesh.userData.slotIndex = i;
      plinthGroup.add(capMesh);

      const ringMat = new THREE.MeshBasicMaterial({
        color: i === 0 ? 0xfbbf24 : 0x38bdf8,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.y = PLINTH_HEIGHT + 0.052;
      plinthGroup.add(ringMesh);

      // Expanding golden brass halo ripple on exhibit arrival
      const haloRippleMat = new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const haloRippleMesh = new THREE.Mesh(haloRippleGeo, haloRippleMat);
      haloRippleMesh.rotation.x = -Math.PI / 2;
      haloRippleMesh.position.y = PLINTH_HEIGHT + 0.056;
      haloRippleMesh.visible = false;
      plinthGroup.add(haloRippleMesh);

      // Contact shadow pad directly on top of the marble cap so models never look detached
      const contactShadowMat = new THREE.MeshBasicMaterial({
        map: shadowTex,
        transparent: true,
        depthWrite: false,
        opacity: 0.75,
      });
      const contactShadowMesh = new THREE.Mesh(shadowPlaneGeo, contactShadowMat);
      contactShadowMesh.rotation.x = -Math.PI / 2;
      contactShadowMesh.position.y = PLINTH_HEIGHT + 0.053;
      contactShadowMesh.visible = false;
      plinthGroup.add(contactShadowMesh);

      // Individual overhead spotlight beam cone so each plinth can flare independently on reveal
      const lightConeMat = new THREE.MeshBasicMaterial({
        color: 0xfff3d1,
        transparent: true,
        opacity: 0.045,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const lightCone = new THREE.Mesh(coneGeo, lightConeMat);
      lightCone.position.y = PLINTH_HEIGHT + 1.85;
      plinthGroup.add(lightCone);

      // Pre-allocated low-lying marble-cap velvet smoke puff (36 particles, 0 runtime allocations)
      const smokeCount = 36;
      const smokePositions = new Float32Array(smokeCount * 3);
      const smokeSeeds = new Float32Array(smokeCount * 4);
      for (let s = 0; s < smokeCount; s++) {
        const angle = (s / smokeCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.25;
        const speed = 0.65 + Math.random() * 0.6;
        const heightDrift = 0.08 + Math.random() * 0.22;
        const phase = Math.random() * Math.PI * 2;
        smokeSeeds[s * 4] = angle;
        smokeSeeds[s * 4 + 1] = speed;
        smokeSeeds[s * 4 + 2] = heightDrift;
        smokeSeeds[s * 4 + 3] = phase;
      }
      const smokeGeo = new THREE.BufferGeometry();
      smokeGeo.setAttribute('position', new THREE.BufferAttribute(smokePositions, 3));
      const smokePuffMat = new THREE.PointsMaterial({
        map: particleTex,
        color: 0xf5ebd6,
        size: 0.42,
        transparent: true,
        opacity: 0,
        depthWrite: false,
      });
      const smokePuffPoints = new THREE.Points(smokeGeo, smokePuffMat);
      smokePuffPoints.position.y = PLINTH_HEIGHT + 0.06;
      smokePuffPoints.visible = false;
      plinthGroup.add(smokePuffPoints);

      // Pre-allocated rising golden theatrical dust motes (48 particles, 0 runtime allocations)
      const dustCount = 48;
      const dustPositions = new Float32Array(dustCount * 3);
      const dustSeeds = new Float32Array(dustCount * 4);
      for (let d = 0; d < dustCount; d++) {
        dustSeeds[d * 4] = Math.random() * Math.PI * 2; // base angle
        dustSeeds[d * 4 + 1] = 0.14 + Math.random() * 0.62; // radial distance
        dustSeeds[d * 4 + 2] = Math.random() * 0.45; // initial height offset
        dustSeeds[d * 4 + 3] = 0.55 + Math.random() * 0.75; // upward velocity factor
      }
      const dustGeo = new THREE.BufferGeometry();
      dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));
      const dustMotesMat = new THREE.PointsMaterial({
        map: particleTex,
        color: 0xffd97d,
        size: 0.11,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const dustMotesPoints = new THREE.Points(dustGeo, dustMotesMat);
      dustMotesPoints.position.y = PLINTH_HEIGHT + 0.06;
      dustMotesPoints.visible = false;
      plinthGroup.add(dustMotesPoints);

      const modelPivot = new THREE.Group();
      modelPivot.position.y = PLINTH_HEIGHT + 0.054;
      plinthGroup.add(modelPivot);

      // Physical 3D Angled Museum Lectern Plaque standing in front of the pedestal,
      // uniformly facing forward (+Z) toward the gallery benches and promenade
      const lecternGroup = new THREE.Group();
      lecternGroup.position.set(0, 0, 1.08);
      lecternGroup.rotation.y = 0;

      const lecternBase = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18, 0.22, 0.035, 24),
        brassMat
      );
      lecternBase.position.y = 0.0175;
      lecternBase.castShadow = true;
      lecternBase.receiveShadow = true;
      lecternBase.userData.slotIndex = i;
      lecternGroup.add(lecternBase);

      const lecternPost = new THREE.Mesh(
        new THREE.CylinderGeometry(0.032, 0.038, 0.78, 16),
        walnutMat
      );
      lecternPost.position.y = 0.4;
      lecternPost.castShadow = true;
      lecternPost.userData.slotIndex = i;
      lecternGroup.add(lecternPost);

      const tabletMount = new THREE.Group();
      tabletMount.position.set(0, 0.81, 0.04);
      tabletMount.rotation.x = -0.62; // Angled ~35° upward toward visitor eye level

      const bezelMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.98, 0.64, 0.036),
        brassMat
      );
      bezelMesh.castShadow = true;
      bezelMesh.receiveShadow = true;
      bezelMesh.userData.slotIndex = i;
      tabletMount.add(bezelMesh);

      const backingSlate = new THREE.Mesh(
        new THREE.BoxGeometry(0.94, 0.6, 0.04),
        new THREE.MeshStandardMaterial({ color: 0x0f141e, roughness: 0.7 })
      );
      backingSlate.userData.slotIndex = i;
      tabletMount.add(backingSlate);

      const lecternScreenMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(0.91, 0.57),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      lecternScreenMesh.position.z = 0.022;
      lecternScreenMesh.userData.slotIndex = i;
      tabletMount.add(lecternScreenMesh);

      lecternGroup.add(tabletMount);
      plinthGroup.add(lecternGroup);

      // Anchor point for optional 3D projection
      const placardAnchor = new THREE.Object3D();
      placardAnchor.position.set(0, 0.85, 1.08);
      plinthGroup.add(placardAnchor);

      this.showroomGroup.add(plinthGroup);

      const slotObj: InternalPedestalSlot = {
        index: i,
        position: pos.clone(),
        plinthGroup,
        modelPivot,
        placardAnchor,
        lecternScreenMesh,
        ringMesh,
        contactShadowMesh,
        bayLight: bayLights[i],
        lightCone,
        lightConeMat,
        haloRippleMesh,
        haloRippleMat,
        smokePuffPoints,
        smokePuffMat,
        smokeSeeds,
        dustMotesPoints,
        dustMotesMat,
        dustSeeds,
        revealTimer: 0,
        whisperedForCurrentAsset: false,
        customVisitorComments: [],
        commentPlayIndex: 0,
        modelRoot: null,
        mixer: null,
        asset: null,
        datasetMeta: null,
        stats: null,
        lodTier: 'close',
        originalMaterials: new Map(),
        loadedAt: 0,
      };
      this.slots.push(slotObj);
      this.updateSlotLecternTexture(slotObj);
    }

    // 9. Build the Central Rotunda "Exquisite Corpse" Dais (Slot #4) for 3-Model Collisions
    const daisGroup = new THREE.Group();
    daisGroup.position.copy(DAIS_POSITION);

    // Wide lower stepped walnut podium
    const daisStep1 = new THREE.Mesh(
      new THREE.CylinderGeometry(1.52, 1.62, 0.22, 48),
      walnutMat
    );
    daisStep1.position.y = 0.11;
    daisStep1.receiveShadow = true;
    daisStep1.castShadow = true;
    daisStep1.userData.slotIndex = DAIS_SLOT_INDEX;
    daisGroup.add(daisStep1);

    const daisStep1Rim = new THREE.Mesh(
      new THREE.CylinderGeometry(1.55, 1.55, 0.035, 48),
      brassMat
    );
    daisStep1Rim.position.y = 0.21;
    daisStep1Rim.userData.slotIndex = DAIS_SLOT_INDEX;
    daisGroup.add(daisStep1Rim);

    // Upper sculptural tier
    const daisStep2 = new THREE.Mesh(
      new THREE.CylinderGeometry(1.16, 1.24, DAIS_HEIGHT - 0.22, 48),
      baseMat
    );
    daisStep2.position.y = 0.22 + (DAIS_HEIGHT - 0.22) / 2;
    daisStep2.receiveShadow = true;
    daisStep2.castShadow = true;
    daisStep2.userData.slotIndex = DAIS_SLOT_INDEX;
    daisGroup.add(daisStep2);

    const daisBrassCollar = new THREE.Mesh(
      new THREE.CylinderGeometry(1.2, 1.2, 0.045, 48),
      brassMat
    );
    daisBrassCollar.position.y = DAIS_HEIGHT - 0.03;
    daisBrassCollar.userData.slotIndex = DAIS_SLOT_INDEX;
    daisGroup.add(daisBrassCollar);

    // Warm cream marble top stage
    const daisCap = new THREE.Mesh(
      new THREE.CylinderGeometry(1.18, 1.18, 0.05, 48),
      capMat
    );
    daisCap.position.y = DAIS_HEIGHT + 0.025;
    daisCap.receiveShadow = true;
    daisCap.userData.slotIndex = DAIS_SLOT_INDEX;
    daisGroup.add(daisCap);

    // Glowing perimeter ring (gold when focused, royal amethyst when idle)
    const daisRingMat = new THREE.MeshBasicMaterial({
      color: 0xc084fc,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    const daisRingMesh = new THREE.Mesh(
      new THREE.RingGeometry(1.1, 1.17, 64),
      daisRingMat
    );
    daisRingMesh.rotation.x = -Math.PI / 2;
    daisRingMesh.position.y = DAIS_HEIGHT + 0.052;
    daisGroup.add(daisRingMesh);

    const daisHaloMat = new THREE.MeshBasicMaterial({
      color: 0xfbbf24,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const daisHaloMesh = new THREE.Mesh(
      new THREE.RingGeometry(1.02, 1.17, 64),
      daisHaloMat
    );
    daisHaloMesh.rotation.x = -Math.PI / 2;
    daisHaloMesh.position.y = DAIS_HEIGHT + 0.056;
    daisHaloMesh.visible = false;
    daisGroup.add(daisHaloMesh);

    const daisShadowMat = new THREE.MeshBasicMaterial({
      map: shadowTex,
      transparent: true,
      depthWrite: false,
      opacity: 0.78,
    });
    const daisContactShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1.9, 1.9),
      daisShadowMat
    );
    daisContactShadow.rotation.x = -Math.PI / 2;
    daisContactShadow.position.y = DAIS_HEIGHT + 0.053;
    daisContactShadow.visible = false;
    daisGroup.add(daisContactShadow);

    const daisConeMat = new THREE.MeshBasicMaterial({
      color: 0xfff3d1,
      transparent: true,
      opacity: 0.05,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const daisCone = new THREE.Mesh(
      new THREE.ConeGeometry(1.38, 3.2, 36, 1, true),
      daisConeMat
    );
    daisCone.position.y = DAIS_HEIGHT + 1.95;
    daisGroup.add(daisCone);

    // Pre-allocated Dais velvet smoke puff
    const dSmokeCount = 42;
    const dSmokePositions = new Float32Array(dSmokeCount * 3);
    const dSmokeSeeds = new Float32Array(dSmokeCount * 4);
    for (let s = 0; s < dSmokeCount; s++) {
      dSmokeSeeds[s * 4] = (s / dSmokeCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.25;
      dSmokeSeeds[s * 4 + 1] = 0.85 + Math.random() * 0.65;
      dSmokeSeeds[s * 4 + 2] = 0.1 + Math.random() * 0.25;
      dSmokeSeeds[s * 4 + 3] = Math.random() * Math.PI * 2;
    }
    const dSmokeGeo = new THREE.BufferGeometry();
    dSmokeGeo.setAttribute('position', new THREE.BufferAttribute(dSmokePositions, 3));
    const dSmokeMat = new THREE.PointsMaterial({
      map: particleTex,
      color: 0xf5ebd6,
      size: 0.48,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const dSmokePoints = new THREE.Points(dSmokeGeo, dSmokeMat);
    dSmokePoints.position.y = DAIS_HEIGHT + 0.06;
    dSmokePoints.visible = false;
    daisGroup.add(dSmokePoints);

    // Pre-allocated Dais rising golden dust motes
    const dDustCount = 54;
    const dDustPositions = new Float32Array(dDustCount * 3);
    const dDustSeeds = new Float32Array(dDustCount * 4);
    for (let d = 0; d < dDustCount; d++) {
      dDustSeeds[d * 4] = Math.random() * Math.PI * 2;
      dDustSeeds[d * 4 + 1] = 0.2 + Math.random() * 0.85;
      dDustSeeds[d * 4 + 2] = Math.random() * 0.5;
      dDustSeeds[d * 4 + 3] = 0.55 + Math.random() * 0.8;
    }
    const dDustGeo = new THREE.BufferGeometry();
    dDustGeo.setAttribute('position', new THREE.BufferAttribute(dDustPositions, 3));
    const dDustMat = new THREE.PointsMaterial({
      map: particleTex,
      color: 0xffd97d,
      size: 0.12,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const dDustPoints = new THREE.Points(dDustGeo, dDustMat);
    dDustPoints.position.y = DAIS_HEIGHT + 0.06;
    dDustPoints.visible = false;
    daisGroup.add(dDustPoints);

    const daisModelPivot = new THREE.Group();
    daisModelPivot.position.y = DAIS_HEIGHT + 0.054;
    daisGroup.add(daisModelPivot);

    // Angled Museum Lectern Plaque in front of the Central Dais
    const daisLecternGroup = new THREE.Group();
    daisLecternGroup.position.set(0, 0, 1.72);
    daisLecternGroup.rotation.y = 0;

    const daisLecternBase = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.22, 0.035, 24),
      brassMat
    );
    daisLecternBase.position.y = 0.0175;
    daisLecternBase.castShadow = true;
    daisLecternBase.receiveShadow = true;
    daisLecternBase.userData.slotIndex = DAIS_SLOT_INDEX;
    daisLecternGroup.add(daisLecternBase);

    const daisLecternPost = new THREE.Mesh(
      new THREE.CylinderGeometry(0.032, 0.038, 0.78, 16),
      walnutMat
    );
    daisLecternPost.position.y = 0.4;
    daisLecternPost.castShadow = true;
    daisLecternPost.userData.slotIndex = DAIS_SLOT_INDEX;
    daisLecternGroup.add(daisLecternPost);

    const daisTabletMount = new THREE.Group();
    daisTabletMount.position.set(0, 0.81, 0.04);
    daisTabletMount.rotation.x = -0.62;

    const daisBezelMesh = new THREE.Mesh(
      new THREE.BoxGeometry(1.06, 0.68, 0.036),
      brassMat
    );
    daisBezelMesh.castShadow = true;
    daisBezelMesh.receiveShadow = true;
    daisBezelMesh.userData.slotIndex = DAIS_SLOT_INDEX;
    daisTabletMount.add(daisBezelMesh);

    const daisBackingSlate = new THREE.Mesh(
      new THREE.BoxGeometry(1.02, 0.64, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x0f141e, roughness: 0.7 })
    );
    daisBackingSlate.userData.slotIndex = DAIS_SLOT_INDEX;
    daisTabletMount.add(daisBackingSlate);

    const daisLecternScreen = new THREE.Mesh(
      new THREE.PlaneGeometry(0.99, 0.61),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    daisLecternScreen.position.z = 0.022;
    daisLecternScreen.userData.slotIndex = DAIS_SLOT_INDEX;
    daisLecternScreen.userData.isDaisLectern = true;
    daisTabletMount.add(daisLecternScreen);

    daisLecternGroup.add(daisTabletMount);
    daisGroup.add(daisLecternGroup);
    this.showroomGroup.add(daisGroup);

    this.daisSlot = {
      index: DAIS_SLOT_INDEX,
      position: DAIS_POSITION.clone(),
      daisGroup,
      lecternGroup: daisLecternGroup,
      modelPivot: daisModelPivot,
      lecternScreenMesh: daisLecternScreen,
      ringMesh: daisRingMesh,
      contactShadowMesh: daisContactShadow,
      daisLight: loungeLight,
      lightCone: daisCone,
      lightConeMat: daisConeMat,
      haloRippleMesh: daisHaloMesh,
      haloRippleMat: daisHaloMat,
      smokePuffPoints: dSmokePoints,
      smokePuffMat: dSmokeMat,
      smokeSeeds: dSmokeSeeds,
      dustMotesPoints: dDustPoints,
      dustMotesMat: dDustMat,
      dustSeeds: dDustSeeds,
      revealTimer: 0,
      modelRoot: null,
      clonedMaterials: [],
      sourceSlotIndices: [],
      sourceAssets: [],
      stats: null,
      sculptureName: null,
      sculptureNameIsGemini: false,
      isNamingSculpture: false,
      namingRequestId: 0,
      collisionCount: 0,
      combinationCursor: 0,
    };
    this.updateDaisLecternTexture();
  }

  private setupShowroomInteractions(): void {
    window.addEventListener('keydown', (e) => {
      this.audio.unlock();
      if (this.mode !== 'showroom') return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.code) {
        case 'KeyW':
        case 'ArrowUp':
          this.moveKeys.forward = true;
          this.camGlideActive = false;
          break;
        case 'KeyS':
        case 'ArrowDown':
          this.moveKeys.backward = true;
          this.camGlideActive = false;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          this.moveKeys.left = true;
          this.camGlideActive = false;
          break;
        case 'KeyD':
        case 'ArrowRight':
          this.moveKeys.right = true;
          this.camGlideActive = false;
          break;
      }
    });

    window.addEventListener('keyup', (e) => {
      switch (e.code) {
        case 'KeyW':
        case 'ArrowUp':
          this.moveKeys.forward = false;
          break;
        case 'KeyS':
        case 'ArrowDown':
          this.moveKeys.backward = false;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          this.moveKeys.left = false;
          break;
        case 'KeyD':
        case 'ArrowRight':
          this.moveKeys.right = false;
          break;
      }
    });

    this.canvas.addEventListener('pointerdown', (e) => {
      this.audio.unlock();
      this.pointerDownPos.set(e.clientX, e.clientY);
    });

    this.canvas.addEventListener('pointerup', (e) => {
      if (this.mode !== 'showroom') return;
      const dx = e.clientX - this.pointerDownPos.x;
      const dy = e.clientY - this.pointerDownPos.y;
      if (Math.hypot(dx, dy) > 6) return;

      const rect = this.canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      this.raycaster.setFromCamera(ndc, this.camera);
      const hits = this.raycaster.intersectObjects(this.showroomGroup.children, true);
      for (const hit of hits) {
        let cur: THREE.Object3D | null = hit.object;
        while (cur) {
          if (typeof cur.userData?.slotIndex === 'number') {
            const idx = cur.userData.slotIndex as number;
            if (idx === DAIS_SLOT_INDEX && this.focusedSlotIndex === DAIS_SLOT_INDEX && hit.object.userData?.isDaisLectern) {
              this.rebuildDaisCollision(true);
            } else {
              this.focusPedestal(idx);
            }
            this.onPedestalSelect?.(idx);
            return;
          }
          cur = cur.parent;
        }
      }
    });
  }

  public setCallbacks(callbacks: {
    onShowroomProjectionUpdate?: (states: PedestalSlotState[]) => void;
    onPedestalSelect?: (slotIndex: number) => void;
    onDaisSculptureNamed?: (name: string) => void;
  }): void {
    this.onShowroomProjectionUpdate = callbacks.onShowroomProjectionUpdate;
    this.onPedestalSelect = callbacks.onPedestalSelect;
    this.onDaisSculptureNamed = callbacks.onDaisSculptureNamed;
  }

  public setMode(mode: ViewerMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    this.camGlideActive = false;

    if (mode === 'showroom') {
      this.studioPivot.visible = false;
      this.showroomGroup.visible = true;
      this.studioFloor.visible = false;
      this.controls.autoRotate = false;
      this.focusPedestal(this.focusedSlotIndex, false);
      this.audio.warmUpKokoroWorker();
    } else {
      this.audio.cancelActiveWhisper();
      this.showroomGroup.visible = false;
      this.studioPivot.visible = true;
      this.studioFloor.visible = this.studioGridPreferred;
      this.controls.autoRotate = this.autoRotateEnabled;
      this.resetCameraView();
    }
  }

  public getMode(): ViewerMode {
    return this.mode;
  }

  public getFilledPedestalCount(): number {
    return this.slots.filter((s) => s.modelRoot !== null).length;
  }

  public getFocusedSlotIndex(): number {
    return this.focusedSlotIndex;
  }

  public getSlotInfo(slotIndex: number): {
    asset: Discovered3DAsset | null;
    datasetMeta: DatasetMetadata | null;
    stats: LoadedModelStats | null;
  } | null {
    if (slotIndex === DAIS_SLOT_INDEX && this.daisSlot) {
      const primaryAsset = this.daisSlot.sourceAssets[0] || null;
      return {
        asset: primaryAsset,
        datasetMeta: null,
        stats: this.daisSlot.stats,
      };
    }
    const slot = this.slots[slotIndex];
    if (!slot) return null;
    return {
      asset: slot.asset,
      datasetMeta: slot.datasetMeta,
      stats: slot.stats,
    };
  }

  public getDaisInfo(): {
    sourceSlotIndices: number[];
    sourceAssets: Discovered3DAsset[];
    stats: LoadedModelStats | null;
    sculptureName: string | null;
    isNamingSculpture: boolean;
    collisionCount: number;
    assemblyMode: DaisAssemblyMode;
    finishMode: DaisFinishMode;
  } {
    return {
      sourceSlotIndices: [...this.daisSlot.sourceSlotIndices],
      sourceAssets: [...this.daisSlot.sourceAssets],
      stats: this.daisSlot.stats,
      sculptureName: this.daisSlot.sculptureName,
      isNamingSculpture: this.daisSlot.isNamingSculpture,
      collisionCount: this.daisSlot.collisionCount,
      assemblyMode: this.daisAssemblyMode,
      finishMode: this.daisFinishMode,
    };
  }

  public getDaisAssemblyMode(): DaisAssemblyMode {
    return this.daisAssemblyMode;
  }

  public toggleDaisAssemblyMode(): DaisAssemblyMode {
    this.daisAssemblyMode = this.daisAssemblyMode === 'collision' ? 'totem' : 'collision';
    this.rebuildDaisCollision(true, false, true);
    return this.daisAssemblyMode;
  }

  public getDaisFinishMode(): DaisFinishMode {
    return this.daisFinishMode;
  }

  public cycleDaisFinishMode(): DaisFinishMode {
    const order: DaisFinishMode[] = ['auto-patina', 'bronze', 'chrome', 'marble'];
    const idx = order.indexOf(this.daisFinishMode);
    this.daisFinishMode = order[(idx + 1) % order.length];
    this.rebuildDaisCollision(true);
    return this.daisFinishMode;
  }

  /**
   * Returns all `Discovered3DAsset` objects currently displayed across the 4 showroom plinths.
   */
  public getLoadedSlotAssets(): Discovered3DAsset[] {
    const assets: Discovered3DAsset[] = [];
    for (const slot of this.slots) {
      if (slot.asset) {
        assets.push(slot.asset);
      }
    }
    return assets;
  }

  /**
   * Checks whether `asset` (or an identical 3D object by ID, URL, or title) is already
   * loaded on any of the 4 showroom plinths. Returns the matching slot index (0–3), or -1.
   */
  public findSlotWithAsset(asset: Discovered3DAsset, ignoreSlotIndex?: number): number {
    for (const slot of this.slots) {
      if (ignoreSlotIndex !== undefined && slot.index === ignoreSlotIndex) continue;
      if (slot.asset && isSame3DObject(slot.asset, asset)) {
        return slot.index;
      }
    }
    return -1;
  }

  public setSlotDatasetMetadata(slotIndex: number, meta: DatasetMetadata): void {
    const slot = this.slots[slotIndex];
    if (slot) {
      slot.datasetMeta = meta;
      this.updateSlotLecternTexture(slot);
    }
  }

  public setSlotCustomVisitorComments(slotIndex: number, comments: string[]): void {
    const slot = this.slots[slotIndex];
    if (slot && comments.length > 0) {
      slot.customVisitorComments = [...comments];
      this.updateSlotLecternTexture(slot);
    }
  }

  public getSlotCustomComments(slotIndex: number): string[] {
    const slot = this.slots[slotIndex];
    return slot ? slot.customVisitorComments : [];
  }

  private offscreenRenderer?: THREE.WebGLRenderer;
  private offscreenCamera?: THREE.PerspectiveCamera;

  public capturePedestalThumbnail(slotIndex: number, width = 256, height = 256): string {
    const slot = this.slots[slotIndex];
    if (!slot || !slot.modelRoot) return '';

    if (!this.offscreenRenderer) {
      const offCanvas = document.createElement('canvas');
      offCanvas.width = width;
      offCanvas.height = height;
      this.offscreenRenderer = new THREE.WebGLRenderer({
        canvas: offCanvas,
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true,
      });
      this.offscreenRenderer.setSize(width, height, false);
      this.offscreenRenderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.offscreenRenderer.toneMappingExposure = 1.15;

      this.offscreenCamera = new THREE.PerspectiveCamera(40, width / height, 0.1, 40);
    }

    // Isolate a dedicated preview scene so we get a clean, well-lit snapshot
    // of the 3D model with neutral gallery studio lighting and warm cream backdrop
    const previewScene = new THREE.Scene();
    previewScene.background = new THREE.Color(0xf1eee7); // Warm bright gallery wall backdrop so dark models stand out

    const ambient = new THREE.AmbientLight(0xffffff, 1.25);
    previewScene.add(ambient);
    const key = new THREE.DirectionalLight(0xfff8eb, 2.8);
    key.position.set(3, 4, 3);
    previewScene.add(key);
    const fill = new THREE.DirectionalLight(0xe0f2fe, 1.4);
    fill.position.set(-3, 2, 2);
    previewScene.add(fill);
    const backLight = new THREE.DirectionalLight(0xffffff, 1.2);
    backLight.position.set(0, 3, -3);
    previewScene.add(backLight);

    // Compute bounding box to frame the camera accurately on this specific object
    const bounds = new THREE.Box3().setFromObject(slot.modelRoot);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bounds.getCenter(center);
    bounds.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 0.4);

    // Temporarily mount the model inside the preview scene
    const origParent = slot.modelRoot.parent;
    const origPos = slot.modelRoot.position.clone();
    const origRot = slot.modelRoot.rotation.clone();

    // Center model at origin in preview scene
    previewScene.add(slot.modelRoot);
    slot.modelRoot.position.set(-center.x, -center.y, -center.z);

    const camDist = maxDim * 1.85;
    this.offscreenCamera!.position.set(camDist * 0.72, camDist * 0.45, camDist * 0.88);
    this.offscreenCamera!.lookAt(0, 0, 0);

    previewScene.updateMatrixWorld(true);
    this.offscreenRenderer.render(previewScene, this.offscreenCamera!);

    // Reattach model back to its original pedestal pivot
    if (origParent) {
      origParent.add(slot.modelRoot);
      slot.modelRoot.position.copy(origPos);
      slot.modelRoot.rotation.copy(origRot);
    }

    const dataUrl = this.offscreenRenderer.domElement.toDataURL('image/jpeg', 0.82);
    return dataUrl.replace(/^data:image\/jpeg;base64,/, '');
  }

  public getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  private wrapCanvasLines(
    ctx: CanvasRenderingContext2D,
    text: string,
    maxWidth: number,
    maxLines: number
  ): string[] {
    const words = text.trim().split(/\s+/);
    const lines: string[] = [];
    let current = '';

    for (const word of words) {
      const candidate = current ? `${current} ${word}` : word;
      if (ctx.measureText(candidate).width <= maxWidth) {
        current = candidate;
      } else {
        if (current) lines.push(current);
        current = word;
        if (lines.length >= maxLines - 1) break;
      }
    }
    if (current && lines.length < maxLines) {
      lines.push(current);
    }
    if (lines.length === maxLines && words.length > 0) {
      const last = lines[maxLines - 1];
      if (ctx.measureText(last).width > maxWidth - 24) {
        lines[maxLines - 1] = `${last.slice(0, Math.max(10, last.length - 4))}…`;
      }
    }
    return lines;
  }

  /**
   * Renders the exhibit's title, dataset info, description, triangle count, maquette scale,
   * and live LOD status directly onto the physical 3D angled museum lectern plaque in front of the pedestal!
   */
  private updateSlotLecternTexture(slot: InternalPedestalSlot): void {
    const mesh = slot.lecternScreenMesh;
    if (!mesh) return;

    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 650;
    const ctx = c.getContext('2d')!;

    const isFocused = slot.index === this.focusedSlotIndex;

    // 1. Museum Slate Tablet Background
    const bgGrad = ctx.createLinearGradient(0, 0, 0, c.height);
    bgGrad.addColorStop(0, isFocused ? '#192233' : '#121824');
    bgGrad.addColorStop(1, '#0c1018');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, c.width, c.height);

    // 2. Double Brushed-Brass Curatorial Border & Corner Studs
    ctx.strokeStyle = isFocused ? '#fbbf24' : '#b88c46';
    ctx.lineWidth = isFocused ? 6 : 4;
    ctx.strokeRect(18, 18, c.width - 36, c.height - 36);

    ctx.strokeStyle = 'rgba(212, 163, 89, 0.35)';
    ctx.lineWidth = 2;
    ctx.strokeRect(30, 30, c.width - 60, c.height - 60);

    ctx.fillStyle = '#d4a359';
    for (const [cx, cy] of [
      [44, 44],
      [c.width - 44, 44],
      [44, c.height - 44],
      [c.width - 44, c.height - 44],
    ]) {
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // 3. Header Row: EXHIBIT #N + Live LOD Status Pill
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 26px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`EXHIBIT #${slot.index + 1}`, 62, 82);

    const lodLabel =
      slot.lodTier === 'close'
        ? '● FULL DETAIL (PBR)'
        : slot.lodTier === 'medium'
          ? '● MEDIUM LOD'
          : '● WIREFRAME LOD';
    const lodColor =
      slot.lodTier === 'close'
        ? '#34d399'
        : slot.lodTier === 'medium'
          ? '#38bdf8'
          : '#9ca3af';

    ctx.font = 'bold 22px ui-monospace, SFMono-Regular, Menlo, monospace';
    const lodTextW = ctx.measureText(lodLabel).width;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.fillRect(c.width - lodTextW - 92, 52, lodTextW + 34, 40);
    ctx.fillStyle = lodColor;
    ctx.fillText(lodLabel, c.width - lodTextW - 75, 79);

    // Header Brass Divider
    ctx.fillStyle = 'rgba(212, 163, 89, 0.35)';
    ctx.fillRect(62, 104, c.width - 124, 2);

    if (!slot.asset) {
      ctx.fillStyle = '#9ca3af';
      ctx.font = 'italic 34px system-ui, -apple-system, sans-serif';
      ctx.fillText('Curating exhibit from Hugging Face Hub…', 62, 310);
    } else {
      // 4. Exhibit Title (1–2 wrapped lines)
      const rawTitle = slot.asset.title || slot.asset.filename;
      ctx.fillStyle = '#f9fafb';
      ctx.font = 'bold 42px system-ui, -apple-system, sans-serif';
      const titleLines = this.wrapCanvasLines(ctx, rawTitle, c.width - 124, 2);
      let curY = 162;
      for (const line of titleLines) {
        ctx.fillText(line, 62, curY);
        curY += 50;
      }

      // 5. Dataset & Author Attribution
      curY += 6;
      ctx.fillStyle = '#fbbf24';
      ctx.font = '600 25px ui-monospace, SFMono-Regular, Menlo, monospace';
      const authorSuffix = slot.datasetMeta?.author ? `  ·  by ${slot.datasetMeta.author}` : '';
      const repoLine = `📊 ${slot.asset.repoId}${authorSuffix}`;
      ctx.fillText(repoLine.slice(0, 58), 62, curY);

      // 6. Curatorial Description (up to 3 wrapped lines)
      curY += 46;
      const rawDesc =
        slot.datasetMeta?.description ||
        slot.asset.subtitle ||
        '3D digital artifact streamed directly from Hugging Face Datasets.';
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '25px system-ui, -apple-system, sans-serif';
      const descLines = this.wrapCanvasLines(ctx, rawDesc, c.width - 124, 3);
      for (const line of descLines) {
        ctx.fillText(line, 62, curY);
        curY += 36;
      }

      // 7. Footer Spec Bar
      ctx.fillStyle = 'rgba(212, 163, 89, 0.35)';
      ctx.fillRect(62, c.height - 110, c.width - 124, 2);

      const tris = slot.stats ? `${slot.stats.triangles.toLocaleString()} tris` : '';
      const scaleTag =
        slot.stats?.scaleLabel && slot.stats.scaleLabel !== '1:1'
          ? `📐 ${slot.stats.scaleLabel}`
          : '';
      const specString = [
        `.${slot.asset.format.toUpperCase()}`,
        tris,
        scaleTag,
      ]
        .filter(Boolean)
        .join('   ·   ');

      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 23px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(specString.slice(0, 64), 62, c.height - 66);
    }

    const newTex = new THREE.CanvasTexture(c);
    newTex.colorSpace = THREE.SRGBColorSpace;
    newTex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());

    const mat = mesh.material as THREE.MeshBasicMaterial;
    if (mat.map) {
      this.disposeTexture(mat.map);
    }
    mat.map = newTex;
    mat.needsUpdate = true;
  }

  /**
   * Renders the Thought Piece lectern plaque showing the 3 combined objects,
   * their source pedestals & datasets, combined polygon count, and interactive controls.
   */
  private updateDaisLecternTexture(): void {
    if (!this.daisSlot) return;
    const mesh = this.daisSlot.lecternScreenMesh;
    if (!mesh) return;

    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 650;
    const ctx = c.getContext('2d')!;

    const isFocused = this.focusedSlotIndex === DAIS_SLOT_INDEX;

    // 1. Deep Royal Amethyst-Slate Tablet Background
    const bgGrad = ctx.createLinearGradient(0, 0, 0, c.height);
    bgGrad.addColorStop(0, isFocused ? '#231934' : '#161224');
    bgGrad.addColorStop(1, '#0c0a14');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, c.width, c.height);

    // 2. Double Brushed-Brass Curatorial Border & Corner Studs
    ctx.strokeStyle = isFocused ? '#fbbf24' : '#c084fc';
    ctx.lineWidth = isFocused ? 6 : 4;
    ctx.strokeRect(18, 18, c.width - 36, c.height - 36);

    ctx.strokeStyle = 'rgba(212, 163, 89, 0.38)';
    ctx.lineWidth = 2;
    ctx.strokeRect(30, 30, c.width - 60, c.height - 60);

    ctx.fillStyle = '#d4a359';
    for (const [cx, cy] of [
      [44, 44],
      [c.width - 44, 44],
      [44, c.height - 44],
      [c.width - 44, c.height - 44],
    ]) {
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // 3. Header Row
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 25px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('⚡ THOUGHT PIECE', 62, 82);

    const modeLabel =
      this.daisAssemblyMode === 'totem' ? '🗿 TOTEM STACK' : '💥 3-MODEL COLLISION';
    const modeBadge = `${modeLabel} #${Math.max(1, this.daisSlot.collisionCount)}`;
    ctx.font = 'bold 20px ui-monospace, SFMono-Regular, Menlo, monospace';
    const badgeW = ctx.measureText(modeBadge).width;
    ctx.fillStyle = 'rgba(192, 132, 252, 0.14)';
    ctx.fillRect(c.width - badgeW - 92, 52, badgeW + 34, 40);
    ctx.fillStyle = '#e9d5ff';
    ctx.fillText(modeBadge, c.width - badgeW - 75, 79);

    ctx.fillStyle = 'rgba(212, 163, 89, 0.35)';
    ctx.fillRect(62, 104, c.width - 124, 2);

    if (this.daisSlot.sourceAssets.length === 0 || !this.daisSlot.modelRoot) {
      ctx.fillStyle = '#fbbf24';
      ctx.font = 'bold 46px system-ui, -apple-system, sans-serif';
      ctx.fillText('Under construction', 62, 290);

      ctx.fillStyle = '#9ca3af';
      ctx.font = 'italic 26px system-ui, -apple-system, sans-serif';
      ctx.fillText('Assembling pedestal exhibits into a Thought Piece…', 62, 344);
    } else {
      // 4. Gemini Sculpture Name (or Mashup Fallback while curating)
      const shortNames = this.daisSlot.sourceAssets.map((a) => {
        const raw = a.title || a.filename.replace(/\.[^.]+$/, '');
        return raw.length > 22 ? `${raw.slice(0, 20)}…` : raw;
      });
      const mashupTitle = shortNames.join('  ×  ');

      ctx.fillStyle = this.daisSlot.sculptureName ? '#c084fc' : '#94a3b8';
      ctx.font = 'bold 19px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(
        this.daisSlot.sculptureName
          ? this.daisSlot.sculptureNameIsGemini
            ? '✦ EXHIBIT TITLE · NAMED BY GEMINI FROM 3 CAMERA ANGLES'
            : '✦ EXHIBIT TITLE'
          : this.daisSlot.isNamingSculpture
            ? '✦ PHOTOGRAPHING SCULPTURE & NAMING WITH GEMINI…'
            : '✦ THOUGHT PIECE',
        62,
        138
      );

      const displayTitle = this.daisSlot.sculptureName
        ? `“${this.daisSlot.sculptureName}”`
        : mashupTitle;

      ctx.fillStyle = this.daisSlot.sculptureName ? '#fef3c7' : '#f9fafb';
      ctx.font = 'bold 38px system-ui, -apple-system, sans-serif';
      const titleLines = this.wrapCanvasLines(ctx, displayTitle, c.width - 124, 2);
      let curY = 182;
      for (const line of titleLines) {
        ctx.fillText(line, 62, curY);
        curY += 44;
      }

      // 5. Source Plinths & Datasets Recipe
      curY += 8;
      for (let i = 0; i < this.daisSlot.sourceAssets.length; i++) {
        const asset = this.daisSlot.sourceAssets[i];
        const plinthNum = (this.daisSlot.sourceSlotIndices[i] ?? i) + 1;
        const partTitle = (asset.title || asset.filename).slice(0, 32);
        const repoShort = asset.repoId.split('/')[1] || asset.repoId;

        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 22px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText(`Plinth #${plinthNum}:`, 62, curY);

        ctx.fillStyle = '#e2e8f0';
        ctx.font = '23px system-ui, -apple-system, sans-serif';
        ctx.fillText(`${partTitle}  (${repoShort})`.slice(0, 56), 210, curY);
        curY += 36;
      }

      // 6. Footer Spec Bar + Interactive Re-Collide Prompt
      ctx.fillStyle = 'rgba(212, 163, 89, 0.35)';
      ctx.fillRect(62, c.height - 110, c.width - 124, 2);

      const tris = this.daisSlot.stats
        ? `${this.daisSlot.stats.triangles.toLocaleString()} combined tris`
        : '';
      const meshes = this.daisSlot.stats ? `${this.daisSlot.stats.meshes} meshes` : '';
      const specString = [
        `${this.daisSlot.sourceAssets.length} GLBs Fused`,
        tris,
        meshes,
        '[R] Re-Assemble · [T] Mode',
      ]
        .filter(Boolean)
        .join('  ·  ');

      ctx.fillStyle = '#e9d5ff';
      ctx.font = 'bold 22px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(specString.slice(0, 68), 62, c.height - 66);
    }

    const newTex = new THREE.CanvasTexture(c);
    newTex.colorSpace = THREE.SRGBColorSpace;
    newTex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());

    const mat = mesh.material as THREE.MeshBasicMaterial;
    if (mat.map) {
      this.disposeTexture(mat.map);
    }
    mat.map = newTex;
    mat.needsUpdate = true;
  }

  private clearDaisModels(): void {
    if (!this.daisSlot) return;
    if (this.daisSlot.modelRoot) {
      this.daisSlot.modelPivot.remove(this.daisSlot.modelRoot);
      this.daisSlot.modelRoot = null;
    }
    for (const mat of this.daisSlot.clonedMaterials) {
      mat.dispose();
    }
    this.daisSlot.clonedMaterials = [];
    this.daisSlot.sourceSlotIndices = [];
    this.daisSlot.sourceAssets = [];
    this.daisSlot.stats = null;
    this.daisSlot.sculptureName = null;
    this.daisSlot.isNamingSculpture = false;
    this.daisSlot.contactShadowMesh.visible = false;
    this.updateDaisLecternTexture();
  }

  private triggerDaisRevealEffect(): void {
    if (!this.daisSlot) return;
    this.daisSlot.revealTimer = REVEAL_DURATION;
    this.daisSlot.haloRippleMesh.visible = true;
    this.daisSlot.smokePuffPoints.visible = true;
    this.daisSlot.dustMotesPoints.visible = true;

    this.daisSlot.modelPivot.position.y = DAIS_HEIGHT - 0.22;
    this.daisSlot.modelPivot.scale.setScalar(0.74);

    if (this.mode === 'showroom') {
      this.audio.updateListener(this.camera);
      this.audio.playPlinthReveal(0);
    }
  }

  /**
   * Lazily initializes 4 seamless 512×512 procedural PBR material presets
   * (Gilded Cast Bronze, Oxidized Verdigris Copper, Brushed Damascus Chrome, and
   * Nero Marquina Gold-Veined Marble) so untextured / plain-white models in the
   * combined Dais sculpture receive rich metallic or stone surface patinas.
   */
  private getProceduralPresets(): Record<'bronze' | 'verdigris' | 'chrome' | 'marble', ProceduralPreset> {
    if (this.proceduralPresets) return this.proceduralPresets;

    const makeCanvasPair = (
      drawFn: (ctx: CanvasRenderingContext2D, bumpCtx: CanvasRenderingContext2D, w: number, h: number) => void
    ): { map: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture } => {
      const w = 512;
      const h = 512;
      const colCanvas = document.createElement('canvas');
      colCanvas.width = w;
      colCanvas.height = h;
      const bumpCanvas = document.createElement('canvas');
      bumpCanvas.width = w;
      bumpCanvas.height = h;

      const ctx = colCanvas.getContext('2d')!;
      const bumpCtx = bumpCanvas.getContext('2d')!;
      drawFn(ctx, bumpCtx, w, h);

      const map = new THREE.CanvasTexture(colCanvas);
      map.colorSpace = THREE.SRGBColorSpace;
      map.wrapS = THREE.RepeatWrapping;
      map.wrapT = THREE.RepeatWrapping;
      map.userData.isPersistentProcedural = true;

      const bumpMap = new THREE.CanvasTexture(bumpCanvas);
      bumpMap.wrapS = THREE.RepeatWrapping;
      bumpMap.wrapT = THREE.RepeatWrapping;
      bumpMap.userData.isPersistentProcedural = true;

      return { map, bumpMap };
    };

    // 1. Gilded Cast Bronze (warm metallic bronze with brushed grain & dark umber crevices)
    const bronzePair = makeCanvasPair((ctx, bumpCtx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#a3743d');
      grad.addColorStop(0.5, '#c89654');
      grad.addColorStop(1, '#7d562a');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      bumpCtx.fillStyle = '#808080';
      bumpCtx.fillRect(0, 0, w, h);

      for (let i = 0; i < 1600; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const len = 18 + Math.random() * 65;
        ctx.strokeStyle =
          i % 5 === 0
            ? 'rgba(45, 95, 82, 0.14)'
            : i % 2 === 0
              ? 'rgba(255, 218, 158, 0.14)'
              : 'rgba(42, 26, 12, 0.16)';
        ctx.lineWidth = 1 + Math.random() * 2.2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + len, y + (Math.random() - 0.5) * 8);
        ctx.stroke();

        bumpCtx.strokeStyle = i % 2 === 0 ? 'rgba(220,220,220,0.18)' : 'rgba(40,40,40,0.18)';
        bumpCtx.lineWidth = 1.2;
        bumpCtx.beginPath();
        bumpCtx.moveTo(x, y);
        bumpCtx.lineTo(x + len, y + (Math.random() - 0.5) * 8);
        bumpCtx.stroke();
      }
    });

    // 2. Oxidized Verdigris Copper (turquoise mineral patina with burnished copper veins)
    const verdigrisPair = makeCanvasPair((ctx, bumpCtx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#2c6e63');
      grad.addColorStop(0.55, '#3b8276');
      grad.addColorStop(1, '#7c4d34');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      bumpCtx.fillStyle = '#808080';
      bumpCtx.fillRect(0, 0, w, h);

      for (let i = 0; i < 420; i++) {
        const cx = Math.random() * w;
        const cy = Math.random() * h;
        const r = 6 + Math.random() * 28;
        ctx.fillStyle =
          i % 3 === 0
            ? 'rgba(184, 115, 51, 0.22)'
            : i % 2 === 0
              ? 'rgba(78, 182, 162, 0.16)'
              : 'rgba(22, 48, 44, 0.18)';
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();

        bumpCtx.fillStyle = i % 2 === 0 ? 'rgba(210,210,210,0.16)' : 'rgba(45,45,45,0.16)';
        bumpCtx.beginPath();
        bumpCtx.arc(cx, cy, r * 0.8, 0, Math.PI * 2);
        bumpCtx.fill();
      }
    });

    // 3. Brushed Damascus Chrome / Titanium (high-polish silver-iridium with swirling lathe grain)
    const chromePair = makeCanvasPair((ctx, bumpCtx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#cbd5e1');
      grad.addColorStop(0.5, '#f1f5f9');
      grad.addColorStop(1, '#94a3b8');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      bumpCtx.fillStyle = '#808080';
      bumpCtx.fillRect(0, 0, w, h);

      for (let i = 0; i < 1800; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const len = 24 + Math.random() * 80;
        ctx.strokeStyle =
          i % 2 === 0 ? 'rgba(255, 255, 255, 0.22)' : 'rgba(51, 65, 85, 0.16)';
        ctx.lineWidth = 0.9 + Math.random() * 1.5;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + len, y + Math.sin(y * 0.04) * 5);
        ctx.stroke();

        bumpCtx.strokeStyle = i % 2 === 0 ? 'rgba(225,225,225,0.16)' : 'rgba(50,50,50,0.16)';
        bumpCtx.lineWidth = 1;
        bumpCtx.beginPath();
        bumpCtx.moveTo(x, y);
        bumpCtx.lineTo(x + len, y + Math.sin(y * 0.04) * 5);
        bumpCtx.stroke();
      }
    });

    // 4. Nero Marquina Gold-Veined Marble (charcoal obsidian stone with antique gold & ivory veins)
    const marblePair = makeCanvasPair((ctx, bumpCtx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#1e222b');
      grad.addColorStop(0.5, '#2b303c');
      grad.addColorStop(1, '#151820');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      bumpCtx.fillStyle = '#808080';
      bumpCtx.fillRect(0, 0, w, h);

      // Subtle stone mottling
      for (let i = 0; i < 300; i++) {
        ctx.fillStyle =
          i % 2 === 0 ? 'rgba(255, 255, 255, 0.03)' : 'rgba(10, 12, 16, 0.06)';
        ctx.beginPath();
        ctx.arc(Math.random() * w, Math.random() * h, 10 + Math.random() * 35, 0, Math.PI * 2);
        ctx.fill();
      }

      // Branching gold and warm ivory mineral veins
      for (let v = 0; v < 26; v++) {
        let vx = Math.random() * w;
        let vy = 0;
        const isGold = v % 2 === 0;
        ctx.strokeStyle = isGold
          ? 'rgba(218, 168, 86, 0.62)'
          : 'rgba(241, 236, 225, 0.45)';
        ctx.lineWidth = isGold ? 1.8 + Math.random() * 2.2 : 1.1 + Math.random() * 1.6;
        ctx.beginPath();
        ctx.moveTo(vx, vy);

        bumpCtx.strokeStyle = 'rgba(225, 225, 225, 0.35)';
        bumpCtx.lineWidth = 2;
        bumpCtx.beginPath();
        bumpCtx.moveTo(vx, vy);

        while (vy < h) {
          vy += 14 + Math.random() * 18;
          vx += (Math.random() - 0.48) * 28;
          ctx.lineTo(vx, vy);
          bumpCtx.lineTo(vx, vy);
        }
        ctx.stroke();
        bumpCtx.stroke();
      }
    });

    this.proceduralPresets = {
      bronze: {
        ...bronzePair,
        color: 0xffedd5,
        metalness: 0.84,
        roughness: 0.27,
        bumpScale: 0.016,
      },
      verdigris: {
        ...verdigrisPair,
        color: 0xe6fffa,
        metalness: 0.68,
        roughness: 0.35,
        bumpScale: 0.018,
      },
      chrome: {
        ...chromePair,
        color: 0xf8fafc,
        metalness: 0.92,
        roughness: 0.16,
        bumpScale: 0.012,
      },
      marble: {
        ...marblePair,
        color: 0xffffff,
        metalness: 0.18,
        roughness: 0.24,
        bumpScale: 0.014,
      },
    };

    return this.proceduralPresets;
  }

  /**
   * Generates tri-planar / box-projected UV coordinates on any geometry that lacks
   * a `uv` attribute (such as raw `.stl` scans or untextured CAD `.glb`s) so procedural
   * textures and bump maps wrap cleanly without stretching or solid-color artifacts.
   */
  private ensureGeometryHasUVs(geometry: THREE.BufferGeometry): void {
    if (geometry.attributes.uv) return;
    const pos = geometry.attributes.position;
    if (!pos) return;

    if (!geometry.attributes.normal) {
      geometry.computeVertexNormals();
    }
    const norm = geometry.attributes.normal;
    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const minX = bb ? bb.min.x : -1;
    const minY = bb ? bb.min.y : -1;
    const minZ = bb ? bb.min.z : -1;
    const spanX = Math.max((bb ? bb.max.x - bb.min.x : 2), 0.001);
    const spanY = Math.max((bb ? bb.max.y - bb.min.y : 2), 0.001);
    const spanZ = Math.max((bb ? bb.max.z - bb.min.z : 2), 0.001);

    const tileScale = 2.0;
    const uvs = new Float32Array(pos.count * 2);

    for (let i = 0; i < pos.count; i++) {
      const px = (pos.getX(i) - minX) / spanX;
      const py = (pos.getY(i) - minY) / spanY;
      const pz = (pos.getZ(i) - minZ) / spanZ;

      const nx = norm ? Math.abs(norm.getX(i)) : 0;
      const ny = norm ? Math.abs(norm.getY(i)) : 1;
      const nz = norm ? Math.abs(norm.getZ(i)) : 0;

      if (nx >= ny && nx >= nz) {
        uvs[i * 2] = pz * tileScale;
        uvs[i * 2 + 1] = py * tileScale;
      } else if (ny >= nx && ny >= nz) {
        uvs[i * 2] = px * tileScale;
        uvs[i * 2 + 1] = pz * tileScale;
      } else {
        uvs[i * 2] = px * tileScale;
        uvs[i * 2 + 1] = py * tileScale;
      }
    }

    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  }

  /**
   * Upgrades untextured/white materials on the Thought Piece Dais (in `auto-patina` mode)
   * with a random patina or applies a unified procedural material across all 3 pieces (`bronze`, `chrome`, `marble`).
   */
  private upgradeDaisPieceMaterial(
    cMat: THREE.Material,
    geometry: THREE.BufferGeometry,
    randomPatinaKey: 'bronze' | 'verdigris' | 'chrome' | 'marble'
  ): THREE.Material {
    const std = cMat as THREE.MeshStandardMaterial;
    const presets = this.getProceduralPresets();

    // Determine whether this mesh is untextured / plain white or gray (or already tagged as untextured patina)
    const wasUntexturedPatina = Boolean(
      cMat.userData?.isFallbackStudioMaterial || cMat.userData?.isUntexturedPatina
    );
    const hasTextureMap =
      !wasUntexturedPatina && Boolean(std.map || std.emissiveMap || std.normalMap);
    const hasVertexColors = Boolean(geometry.attributes.color);
    const hsl = { h: 0, s: 0, l: 1 };
    if (std.color && typeof std.color.getHSL === 'function') {
      std.color.getHSL(hsl);
    }
    const isPlainWhiteOrGray =
      wasUntexturedPatina || (!hasTextureMap && !hasVertexColors && hsl.s < 0.2 && hsl.l > 0.42);

    const isUntexturedSolidColor = !hasTextureMap && !hasVertexColors && !isPlainWhiteOrGray;

    if (this.daisFinishMode === 'auto-patina') {
      if (isPlainWhiteOrGray) {
        // Replace untextured parts with the piece's random procedural museum patina
        this.ensureGeometryHasUVs(geometry);
        const preset = presets[randomPatinaKey];

        std.map = preset.map;
        std.bumpMap = preset.bumpMap;
        std.bumpScale = preset.bumpScale;
        std.color?.setHex(preset.color);
        std.metalness = preset.metalness;
        std.roughness = preset.roughness;
        std.userData.isUntexturedPatina = true;
        std.needsUpdate = true;
      } else if (isUntexturedSolidColor) {
        // Give flat-colored untextured parts subtle metallic sheen and brushed bump grain
        this.ensureGeometryHasUVs(geometry);
        std.bumpMap = std.bumpMap || presets.bronze.bumpMap;
        std.bumpScale = 0.012;
        std.metalness = Math.max(std.metalness ?? 0, 0.52);
        std.roughness = Math.min(std.roughness ?? 0.5, 0.32);
        std.needsUpdate = true;
      }
      return cMat;
    }

    // Unified material finish override ('bronze' | 'chrome' | 'marble')
    this.ensureGeometryHasUVs(geometry);
    const preset = presets[this.daisFinishMode];
    std.map = preset.map;
    std.bumpMap = std.bumpMap || preset.bumpMap;
    std.bumpScale = preset.bumpScale;
    std.color?.setHex(preset.color);
    std.metalness = preset.metalness;
    std.roughness = preset.roughness;
    std.vertexColors = false;
    std.needsUpdate = true;
    return cMat;
  }
  /**
   * Computes a 10-band vertical cross-section profile (dense medoid centers, effective radii,
   * solid-core shoulder/pelvis heights, bottom vs. top mass, and sampled 3D vertices) so the
   * Totem Stacker guarantees deep physical 3D mesh intersections rather than floating bounding boxes:
   * 1. Sorts the 3 pieces with the heaviest/broadest mass at the bottom
   * 2. Orients pieces so heavy bases sit down and skinny necks point toward the seam
   * 3. Finds the true solid-core shoulder (`coreTopSlice`) and pelvis (`coreBottomSlice`) bypassing thin legs/antennae
   * 4. Uses Dense Vertex Medoids (`denseX`, `denseZ`) instead of empty-space arithmetic means (e.g. between two legs)
   * 5. Runs a 3D Vertex Penetration Pull-In check so every totem joint physically intersects in 3D space!
   */
  private computeVerticalSliceProfile(
    root: THREE.Object3D,
    referenceParent: THREE.Group,
    numSlices = 10
  ): {
    bounds: { min: THREE.Vector3; max: THREE.Vector3; size: THREE.Vector3; center: THREE.Vector3 };
    sliceCx: Float32Array;
    sliceCz: Float32Array;
    sliceRadius: Float32Array;
    coreBottomSlice: number;
    coreTopSlice: number;
    coreBottomY: number;
    coreTopY: number;
    bulkMass: number;
    bottomMass: number;
    topMass: number;
    midRadius: number;
    sampledPts: Array<{ x: number; y: number; z: number; slice: number }>;
  } | null {
    const bounds = this.computeRobustLocalBounds(root, referenceParent);
    if (!bounds) return null;

    const minY = bounds.min.y;
    const spanY = Math.max(bounds.size.y, 0.001);
    const fallbackRadius = Math.max(bounds.size.x, bounds.size.z) * 0.35;

    const sumX = new Float64Array(numSlices);
    const sumZ = new Float64Array(numSlices);
    const counts = new Int32Array(numSlices);

    referenceParent.updateWorldMatrix(true, true);
    const invParentMatrix = referenceParent.matrixWorld.clone().invert();
    const localMatrix = new THREE.Matrix4();
    const v = new THREE.Vector3();

    const sampledPts: Array<{ x: number; y: number; z: number; slice: number }> = [];
    const sliceBuckets: Array<Array<{ x: number; y: number; z: number }>> = Array.from(
      { length: numSlices },
      () => []
    );

    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const pos = mesh.geometry.attributes.position;
      if (!pos || pos.count === 0) return;

      localMatrix.multiplyMatrices(invParentMatrix, mesh.matrixWorld);
      const step = Math.max(1, Math.floor(pos.count / 750));

      for (let i = 0; i < pos.count; i += step) {
        const vx = pos.getX(i);
        const vy = pos.getY(i);
        const vz = pos.getZ(i);
        if (!Number.isFinite(vx) || !Number.isFinite(vy) || !Number.isFinite(vz)) continue;
        v.set(vx, vy, vz).applyMatrix4(localMatrix);

        const normY = THREE.MathUtils.clamp((v.y - minY) / spanY, 0, 0.9999);
        const s = Math.floor(normY * numSlices);
        sumX[s] += v.x;
        sumZ[s] += v.z;
        counts[s]++;
        const pt = { x: v.x, y: v.y, z: v.z, slice: s };
        sampledPts.push(pt);
        if (sliceBuckets[s].length < 80) {
          sliceBuckets[s].push(pt);
        }
      }
    });

    const sliceCx = new Float32Array(numSlices);
    const sliceCz = new Float32Array(numSlices);
    const sumR2 = new Float64Array(numSlices);

    // Compute Dense Vertex Medoid per slice so legged/arched/hollow models anchor inside actual mesh geometry
    // rather than in the empty space between legs!
    for (let s = 0; s < numSlices; s++) {
      const bucket = sliceBuckets[s];
      if (bucket.length > 0) {
        const meanX = sumX[s] / counts[s];
        const meanZ = sumZ[s] / counts[s];
        let bestPt = bucket[0];
        let bestScore = -Infinity;

        for (let i = 0; i < bucket.length; i++) {
          const cand = bucket[i];
          let neighbors = 0;
          for (let j = 0; j < bucket.length; j++) {
            const other = bucket[j];
            const d2 = (cand.x - other.x) ** 2 + (cand.z - other.z) ** 2;
            if (d2 < 0.045) neighbors++;
          }
          // Prefer high local vertex density near the central trunk
          const distToMean = Math.hypot(cand.x - meanX, cand.z - meanZ);
          const score = neighbors * 2.5 - distToMean * 4.0;
          if (score > bestScore) {
            bestScore = score;
            bestPt = cand;
          }
        }
        // Blend 70% dense medoid + 30% mean so it stays inside real geometry while centered on the trunk
        sliceCx[s] = bestPt.x * 0.7 + meanX * 0.3;
        sliceCz[s] = bestPt.z * 0.7 + meanZ * 0.3;
      } else {
        sliceCx[s] = bounds.center.x;
        sliceCz[s] = bounds.center.z;
      }
    }

    for (const pt of sampledPts) {
      const dx = pt.x - sliceCx[pt.slice];
      const dz = pt.z - sliceCz[pt.slice];
      sumR2[pt.slice] += dx * dx + dz * dz;
    }

    const sliceRadius = new Float32Array(numSlices);
    for (let s = 0; s < numSlices; s++) {
      if (counts[s] > 2) {
        sliceRadius[s] = Math.max(0.04, Math.sqrt(sumR2[s] / counts[s]) * 1.35);
      } else {
        sliceRadius[s] = fallbackRadius;
      }
    }

    for (let s = 0; s < numSlices; s++) {
      if (counts[s] === 0) {
        const prev = s > 0 ? s - 1 : Math.min(numSlices - 1, s + 1);
        sliceCx[s] = sliceCx[prev];
        sliceCz[s] = sliceCz[prev];
        sliceRadius[s] = sliceRadius[prev];
      }
    }

    let bottomMass = 0;
    let topMass = 0;
    let midRadius = 0;
    let areaSum = 0;
    let maxCount = 1;

    for (let s = 0; s < numSlices; s++) {
      const r2 = sliceRadius[s] * sliceRadius[s];
      areaSum += r2;
      if (counts[s] > maxCount) maxCount = counts[s];
      if (s <= 3) bottomMass += r2;
      if (s >= numSlices - 4) topMass += r2;
      if (s >= 2 && s <= 7) midRadius = Math.max(midRadius, sliceRadius[s]);
    }
    midRadius = Math.max(midRadius, fallbackRadius * 0.6, 0.08);

    // Find solid-core bottom slice (pelvis) and solid-core top slice (shoulder),
    // skipping spindly extremities (thin legs, stands, antennae, sword tips)
    let coreBottomSlice = 1;
    for (let s = 0; s <= 4; s++) {
      if (counts[s] >= maxCount * 0.28 && sliceRadius[s] >= midRadius * 0.48) {
        coreBottomSlice = s;
        break;
      }
    }

    let coreTopSlice = numSlices - 2;
    for (let s = numSlices - 1; s >= 5; s--) {
      if (counts[s] >= maxCount * 0.28 && sliceRadius[s] >= midRadius * 0.48) {
        coreTopSlice = s;
        break;
      }
    }

    const coreBottomY = minY + ((coreBottomSlice + 0.5) / numSlices) * spanY;
    const coreTopY = minY + ((coreTopSlice + 0.5) / numSlices) * spanY;

    const footprintArea = bounds.size.x * bounds.size.z;
    const bulkMass = (areaSum / numSlices) * 1.6 + footprintArea * 0.45 + bottomMass * 0.25;

    return {
      bounds,
      sliceCx,
      sliceCz,
      sliceRadius,
      coreBottomSlice,
      coreTopSlice,
      coreBottomY,
      coreTopY,
      bulkMass,
      bottomMass,
      topMass,
      midRadius,
      sampledPts,
    };
  }

  public rebuildDaisCollision(
    triggerReveal = true,
    callGemini = true,
    keepCurrentCombo = false
  ): boolean {
    if (!this.daisSlot) return false;
    const prevSlotIndices = [...this.daisSlot.sourceSlotIndices];
    const prevSculptureName = this.daisSlot.sculptureName;
    const prevSculptureNameIsGemini = this.daisSlot.sculptureNameIsGemini;
    this.clearDaisModels();

    const availableSlots = this.slots.filter((s) => s.modelRoot !== null && s.asset !== null);
    if (availableSlots.length === 0) {
      this.daisSlot.sourceSlotIndices = [];
      this.daisSlot.sourceAssets = [];
      this.daisSlot.stats = null;
      this.updateDaisLecternTexture();
      return false;
    }

    // Choose up to 3 distinct pedestals to collide
    let chosenSlots: InternalPedestalSlot[];
    if (keepCurrentCombo && prevSlotIndices.length >= 2) {
      chosenSlots = prevSlotIndices
        .map((i) => this.slots[i])
        .filter((s) => s && s.modelRoot && s.asset);
      if (chosenSlots.length < 2) {
        chosenSlots = availableSlots.slice(0, 3);
      }
    } else if (availableSlots.length >= 4) {
      const combos = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
      ];
      const combo = combos[this.daisSlot.combinationCursor % combos.length];
      this.daisSlot.combinationCursor++;
      chosenSlots = combo.map((i) => this.slots[i]).filter((s) => s && s.modelRoot && s.asset);
    } else {
      chosenSlots = availableSlots.slice(0, 3);
    }

    const daisMasterGroup = new THREE.Group();
    daisMasterGroup.userData.slotIndex = DAIS_SLOT_INDEX;

    let totalVerts = 0;
    let totalTris = 0;
    let totalMeshes = 0;
    const baseTheta = Math.random() * Math.PI * 2;

    interface PreparedDaisPiece {
      slotIndex: number;
      asset: Discovered3DAsset;
      unitGroup: THREE.Group;
      colliderGroup: THREE.Group;
      bulkMass: number;
      bottomMass: number;
      topMass: number;
      rawSize: THREE.Vector3;
    }

    const preparedPieces: PreparedDaisPiece[] = [];

    for (let idx = 0; idx < chosenSlots.length; idx++) {
      const slot = chosenSlots[idx];
      if (!slot.modelRoot || !slot.asset) continue;

      // 1. Temporarily restore full-resolution original materials on the source pedestal
      //    so our clone always gets the native PBR textures even if the pedestal is far away
      slot.modelRoot.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.isMesh) {
          const orig = slot.originalMaterials.get(mesh);
          if (orig) mesh.material = orig;
        }
      });

      const sourceInner = slot.modelRoot.children[0] || slot.modelRoot;
      let clonedPiece: THREE.Object3D;
      try {
        clonedPiece = cloneSkeleton(sourceInner);
      } catch {
        clonedPiece = sourceInner.clone(true);
      }

      // Restore the source pedestal's current LOD state immediately
      this.applySlotLodTier(slot, slot.lodTier);

      // Reset root transforms on the cloned piece so pedestal normalization doesn't double-apply
      clonedPiece.position.set(0, 0, 0);
      clonedPiece.rotation.set(0, 0, 0);
      clonedPiece.scale.set(1, 1, 1);

      const patinaChoices: Array<'bronze' | 'verdigris' | 'chrome' | 'marble'> = [
        'bronze',
        'verdigris',
        'chrome',
        'marble',
      ];
      const randomPiecePatina =
        patinaChoices[Math.floor(Math.random() * patinaChoices.length)];

      // Clone material instances and apply procedural patina to any untextured/white meshes
      clonedPiece.traverse((obj) => {
        obj.userData.slotIndex = DAIS_SLOT_INDEX;
        const mesh = obj as THREE.Mesh;
        if (mesh.isMesh && mesh.geometry) {
          mesh.castShadow = true;
          mesh.receiveShadow = true;

          if (Array.isArray(mesh.material)) {
            mesh.material = mesh.material.map((m) => {
              const cMat = this.upgradeDaisPieceMaterial(
                m.clone(),
                mesh.geometry,
                randomPiecePatina
              );
              if ('wireframe' in cMat) {
                (cMat as THREE.MeshStandardMaterial).wireframe = this.wireframeEnabled;
              }
              this.daisSlot.clonedMaterials.push(cMat);
              return cMat;
            });
          } else if (mesh.material) {
            const cMat = this.upgradeDaisPieceMaterial(
              mesh.material.clone(),
              mesh.geometry,
              randomPiecePatina
            );
            if ('wireframe' in cMat) {
              (cMat as THREE.MeshStandardMaterial).wireframe = this.wireframeEnabled;
            }
            this.daisSlot.clonedMaterials.push(cMat);
            mesh.material = cMat;
          }
        }
      });

      const pieceStats = this.computeStatsAndPrepareMeshes(clonedPiece);
      totalVerts += pieceStats.vertices;
      totalTris += pieceStats.triangles;
      totalMeshes += pieceStats.meshes;

      // 2. Center and normalize each individual piece inside its own unitGroup (~1.18m span)
      //    so 3D rotations spin around the object's true center of mass!
      const unitGroup = new THREE.Group();
      unitGroup.add(clonedPiece);
      let rawSize = new THREE.Vector3(1, 1, 1);
      const rawBounds = this.computeRobustLocalBounds(clonedPiece, unitGroup);
      if (rawBounds) {
        const maxDim = Math.max(rawBounds.size.x, rawBounds.size.y, rawBounds.size.z, 0.001);
        const unitScale = 1.18 / maxDim;
        clonedPiece.scale.setScalar(unitScale);
        const centeredBounds = this.computeRobustLocalBounds(clonedPiece, unitGroup);
        if (centeredBounds) {
          clonedPiece.position.sub(centeredBounds.center);
          rawSize = centeredBounds.size.clone();
        }
      }

      const initProfile = this.computeVerticalSliceProfile(unitGroup, unitGroup);
      const colliderGroup = new THREE.Group();
      colliderGroup.add(unitGroup);

      preparedPieces.push({
        slotIndex: slot.index,
        asset: slot.asset,
        unitGroup,
        colliderGroup,
        bulkMass: initProfile ? initProfile.bulkMass : 1,
        bottomMass: initProfile ? initProfile.bottomMass : 1,
        topMass: initProfile ? initProfile.topMass : 1,
        rawSize,
      });
    }

    if (this.daisAssemblyMode === 'totem') {
      // HEURISTIC 1: Mass & Footprint Sorting — place the heaviest, broadest mass at the bottom
      //              and the lightest/slenderest finial at the crown!
      preparedPieces.sort((a, b) => b.bulkMass - a.bulkMass);

      let prevProfile: ReturnType<typeof this.computeVerticalSliceProfile> = null;
      const tierScales = [1.14, 0.96, 0.84];

      for (let idx = 0; idx < preparedPieces.length; idx++) {
        const piece = preparedPieces[idx];
        const { unitGroup, colliderGroup } = piece;

        // HEURISTIC 2: Taper & Upright Orientation
        // - For Tier 0 (Base): if the object is top-heavy (top half > 1.28× wider than bottom half),
        //   invert it so the broad heavy mass anchors the base and the skinny neck points up!
        // - For Tier 1 & 2: if the object's top is much narrower than its bottom, flip it 50% of the time
        //   or orient its narrower waist toward the lower piece's neck for a clean hourglass/socket joint.
        if (idx === 0 && piece.topMass > piece.bottomMass * 1.28) {
          unitGroup.rotation.z = Math.PI;
        } else if (idx > 0 && piece.topMass < piece.bottomMass * 0.72) {
          unitGroup.rotation.z = Math.PI;
        }

        colliderGroup.scale.setScalar(tierScales[idx] ?? 0.85);
        colliderGroup.rotation.set(
          (Math.random() - 0.5) * 0.09,
          Math.random() * Math.PI * 2,
          (Math.random() - 0.5) * 0.09
        );
        daisMasterGroup.add(colliderGroup);

        const curProfile = this.computeVerticalSliceProfile(colliderGroup, daisMasterGroup);
        if (!curProfile) continue;

        if (idx === 0 || !prevProfile) {
          // Ground Tier 0 cleanly at (0, 0, 0) using its bottom-core medoid
          const baseSlice = curProfile.coreBottomSlice;
          const baseCx = curProfile.sliceCx[baseSlice];
          const baseCz = curProfile.sliceCz[baseSlice];
          colliderGroup.position.x -= baseCx;
          colliderGroup.position.z -= baseCz;
          colliderGroup.position.y -= curProfile.bounds.min.y;
        } else {
          // HEURISTIC 3: Solid-Core Shoulder-to-Pelvis Sinking + Dynamic Skinny-Neck Overlap
          // Instead of touching extreme bounding-box tips (`max.y` to `min.y`), we weld the upper piece's
          // solid-core pelvis (`coreBottomY`) directly into the lower piece's solid-core shoulder (`coreTopY`),
          // PLUS an additional 24%–42% deep socket penetration so skinny parts deeply interpenetrate!
          const lowerTipR =
            (prevProfile.sliceRadius[7] + prevProfile.sliceRadius[8] + prevProfile.sliceRadius[9]) / 3;
          const upperTipR =
            (curProfile.sliceRadius[0] + curProfile.sliceRadius[1] + curProfile.sliceRadius[2]) / 3;
          const combinedMidR = Math.max(prevProfile.midRadius + curProfile.midRadius, 0.15);
          const taperRatio = (lowerTipR + upperTipR) / combinedMidR;

          const extraCorePenetrationFrac = THREE.MathUtils.clamp(
            0.38 - taperRatio * 0.18,
            0.22,
            0.42
          );
          const segHeight = Math.max(curProfile.bounds.size.y, 0.15);
          const lowerHeight = Math.max(prevProfile.bounds.size.y, 0.15);
          const extraDepth = Math.min(
            segHeight * extraCorePenetrationFrac,
            lowerHeight * 0.42
          );

          // HEURISTIC 4: Dense Medoid Contact-Slice Alignment
          const lowerWeldSlice = Math.max(4, prevProfile.coreTopSlice - 1);
          const upperWeldSlice = Math.min(5, curProfile.coreBottomSlice + 1);

          const targetSeamX = prevProfile.sliceCx[lowerWeldSlice];
          const targetSeamZ = prevProfile.sliceCz[lowerWeldSlice];
          const upperSeamX = curProfile.sliceCx[upperWeldSlice];
          const upperSeamZ = curProfile.sliceCz[upperWeldSlice];

          const dx = targetSeamX - upperSeamX;
          const dz = targetSeamZ - upperSeamZ;
          const dy = prevProfile.coreTopY - curProfile.coreBottomY - extraDepth;

          colliderGroup.position.x += dx;
          colliderGroup.position.z += dz;
          colliderGroup.position.y += dy;

          // HEURISTIC 5: Direct 3D Vertex Cloud Pull-In Verification
          // Check the actual 3D Euclidean distance between the lower piece's upper-half vertices
          // and the upper piece's lower-half vertices after shifting. If they aren't deeply
          // intersecting yet, pull the upper piece straight into the nearest dense vertex of the lower piece!
          let bestDistSq = Infinity;
          let bestLowerPt: { x: number; y: number; z: number } | null = null;
          let bestUpperPt: { x: number; y: number; z: number } | null = null;

          const lowerCandidates = prevProfile.sampledPts.filter((p) => p.slice >= 4);
          const upperCandidates = curProfile.sampledPts.filter((p) => p.slice <= 5);
          const lStep = Math.max(1, Math.floor(lowerCandidates.length / 120));
          const uStep = Math.max(1, Math.floor(upperCandidates.length / 120));

          for (let li = 0; li < lowerCandidates.length; li += lStep) {
            const lp = lowerCandidates[li];
            for (let ui = 0; ui < upperCandidates.length; ui += uStep) {
              const up = upperCandidates[ui];
              const ux = up.x + dx;
              const uy = up.y + dy;
              const uz = up.z + dz;
              const dSq = (lp.x - ux) ** 2 + (lp.y - uy) ** 2 + (lp.z - uz) ** 2;
              if (dSq < bestDistSq) {
                bestDistSq = dSq;
                bestLowerPt = lp;
                bestUpperPt = { x: ux, y: uy, z: uz };
              }
            }
          }

          if (bestLowerPt && bestUpperPt) {
            // Pull the upper piece's closest vertex cluster 0.14m inside the lower piece's vertex cluster!
            const pullX = bestLowerPt.x - bestUpperPt.x;
            const pullY = bestLowerPt.y - bestUpperPt.y - 0.12;
            const pullZ = bestLowerPt.z - bestUpperPt.z;
            colliderGroup.position.x += pullX * 0.85;
            colliderGroup.position.z += pullZ * 0.85;
            if (pullY < 0) {
              colliderGroup.position.y += pullY * 0.75;
            }
          }
        }

        // Re-measure profile in its final world-stacked pose so the next tier stacks onto its true coordinates
        prevProfile = this.computeVerticalSliceProfile(colliderGroup, daisMasterGroup);
      }
    } else {
      // 3B. Chaotic 3D Intersecting Collision Transforms
      for (let idx = 0; idx < preparedPieces.length; idx++) {
        const { colliderGroup } = preparedPieces[idx];
        if (idx === 0) {
          colliderGroup.scale.setScalar(0.96 + Math.random() * 0.22);
          colliderGroup.rotation.set(
            (Math.random() - 0.5) * 0.85,
            Math.random() * Math.PI * 2,
            (Math.random() - 0.5) * 0.85
          );
          colliderGroup.position.set(0, 0, 0);
        } else if (idx === 1) {
          colliderGroup.scale.setScalar(0.84 + Math.random() * 0.28);
          colliderGroup.rotation.set(
            Math.random() * Math.PI * 2,
            Math.random() * Math.PI * 2,
            Math.random() * Math.PI * 2
          );
          const r1 = 0.14 + Math.random() * 0.14;
          colliderGroup.position.set(
            Math.cos(baseTheta) * r1,
            (Math.random() - 0.35) * 0.26,
            Math.sin(baseTheta) * r1
          );
        } else {
          colliderGroup.scale.setScalar(0.8 + Math.random() * 0.32);
          colliderGroup.rotation.set(
            Math.random() * Math.PI * 2,
            Math.random() * Math.PI * 2,
            Math.random() * Math.PI * 2
          );
          const theta2 = baseTheta + Math.PI * (0.65 + Math.random() * 0.7);
          const r2 = 0.14 + Math.random() * 0.15;
          colliderGroup.position.set(
            Math.cos(theta2) * r2,
            (Math.random() - 0.3) * 0.3,
            Math.sin(theta2) * r2
          );
        }
        daisMasterGroup.add(colliderGroup);
      }
    }

    this.daisSlot.modelRoot = daisMasterGroup;
    this.daisSlot.sourceSlotIndices = preparedPieces.map((p) => p.slotIndex);
    this.daisSlot.sourceAssets = preparedPieces.map((p) => p.asset);
    this.daisSlot.collisionCount++;
    this.newShowroomModelsSinceDaisRebuild = 0;
    this.daisSlot.modelPivot.rotation.set(0, 0, 0);
    this.daisSlot.modelPivot.add(daisMasterGroup);

    // 4. Frame and ground the combined 3-object sculpture onto the Central Dais
    const normResult = this.normalizeObjectOnPivot(
      daisMasterGroup,
      this.daisSlot.modelPivot,
      this.daisAssemblyMode === 'totem'
        ? { maxHorizontalDiameter: 1.72, maxHeight: 2.12 }
        : { maxHorizontalDiameter: 1.88, maxHeight: 1.78 }
    );

    this.daisSlot.stats = {
      vertices: totalVerts,
      triangles: totalTris,
      meshes: totalMeshes,
      scaleFactor: normResult.scaleFactor,
      scaleLabel: normResult.scaleLabel,
    };

    const shadowSpan = THREE.MathUtils.clamp(
      Math.max(normResult.footprintX, normResult.footprintZ) * 1.08,
      0.85,
      1.85
    );
    this.daisSlot.contactShadowMesh.scale.set(shadowSpan, shadowSpan, 1);
    this.daisSlot.contactShadowMesh.visible = true;

    this.daisSlot.sculptureName = keepCurrentCombo ? prevSculptureName : null;
    this.daisSlot.sculptureNameIsGemini = keepCurrentCombo ? prevSculptureNameIsGemini : false;
    this.daisSlot.namingRequestId++;
    const canName =
      callGemini && this.mode === 'showroom' && this.daisSlot.sourceAssets.length >= 2;
    this.daisSlot.isNamingSculpture = canName;

    this.updateDaisLecternTexture();
    if (triggerReveal) {
      this.triggerDaisRevealEffect();
    }

    if (canName) {
      void this.requestGeminiSculptureName(this.daisSlot.namingRequestId);
    }

    return true;
  }

  /**
   * Captures 3 multi-angle 256×256 JPEG snapshots of the Central Dais sculpture
   * using the primary WebGLRenderer (so all resident PBR textures & lighting are preserved)
   * and immediately restores the user camera frame so no flicker occurs.
   */
  public captureDaisSculptureSnapshots(
    angles = [-0.68, 0.68, 2.38],
    thumbSize = 256
  ): string[] {
    const dais = this.daisSlot;
    const modelRoot = dais?.modelRoot;
    if (!dais || !modelRoot) return [];
    const prevShowroomVis = this.showroomGroup.visible;
    const prevStudioVis = this.studioPivot.visible;
    const prevPivotY = dais.modelPivot.position.y;
    const prevPivotScale = dais.modelPivot.scale.x;
    const prevPivotRotY = dais.modelPivot.rotation.y;
    const prevLecternVis = dais.lecternGroup.visible;
    const prevHaloVis = dais.haloRippleMesh.visible;
    const prevSmokeVis = dais.smokePuffPoints.visible;
    const prevDustVis = dais.dustMotesPoints.visible;
    const prevConeVis = dais.lightCone.visible;
    const prevLightIntensity = dais.daisLight.intensity;

    // Position the sculpture cleanly at its resting height on the Dais for crisp photography
    this.showroomGroup.visible = true;
    this.studioPivot.visible = false;
    dais.modelPivot.position.y = DAIS_HEIGHT + 0.054;
    dais.modelPivot.scale.setScalar(1.0);
    dais.modelPivot.rotation.y = 0;
    dais.lecternGroup.visible = false;
    dais.haloRippleMesh.visible = false;
    dais.smokePuffPoints.visible = false;
    dais.dustMotesPoints.visible = false;
    dais.lightCone.visible = false;
    dais.daisLight.intensity = 4.2;

    this.scene.updateMatrixWorld(true);

    const bounds = new THREE.Box3().setFromObject(modelRoot);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bounds.getCenter(center);
    bounds.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 0.65);
    const camDist = maxDim * 1.62;

    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = thumbSize;
    thumbCanvas.height = thumbSize;
    const ctx = thumbCanvas.getContext('2d');
    const snapshots: string[] = [];

    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const cropSpan = Math.min(cw, ch);
    const sx = (cw - cropSpan) * 0.5;
    const sy = (ch - cropSpan) * 0.5;

    this.daisSnapshotCamera.aspect = cw / Math.max(1, ch);
    this.daisSnapshotCamera.updateProjectionMatrix();

    for (const angle of angles) {
      this.daisSnapshotCamera.position.set(
        center.x + Math.sin(angle) * camDist,
        center.y + maxDim * 0.26,
        center.z + Math.cos(angle) * camDist
      );
      this.daisSnapshotCamera.lookAt(center);
      this.daisSnapshotCamera.updateMatrixWorld(true);

      this.renderer.render(this.scene, this.daisSnapshotCamera);

      if (ctx && cropSpan > 0) {
        ctx.clearRect(0, 0, thumbSize, thumbSize);
        ctx.drawImage(this.canvas, sx, sy, cropSpan, cropSpan, 0, 0, thumbSize, thumbSize);
        const dataUrl = thumbCanvas.toDataURL('image/jpeg', 0.8);
        snapshots.push(dataUrl.replace(/^data:image\/jpeg;base64,/, ''));
      }
    }

    // Restore scene & Dais reveal state and immediately re-render the user's active view
    this.showroomGroup.visible = prevShowroomVis;
    this.studioPivot.visible = prevStudioVis;
    dais.modelPivot.position.y = prevPivotY;
    dais.modelPivot.scale.setScalar(prevPivotScale);
    dais.modelPivot.rotation.y = prevPivotRotY;
    dais.lecternGroup.visible = prevLecternVis;
    dais.haloRippleMesh.visible = prevHaloVis;
    dais.smokePuffPoints.visible = prevSmokeVis;
    dais.dustMotesPoints.visible = prevDustVis;
    dais.lightCone.visible = prevConeVis;
    dais.daisLight.intensity = prevLightIntensity;

    this.scene.updateMatrixWorld(true);
    this.renderer.render(this.scene, this.camera);

    return snapshots;
  }

  private async requestGeminiSculptureName(requestId: number): Promise<void> {
    // Wait one animation frame so all textures and buffers are settled on the GPU
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    if (!this.daisSlot || this.daisSlot.namingRequestId !== requestId) return;

    const snapshots = this.captureDaisSculptureSnapshots();
    if (snapshots.length === 0) {
      const fallback = pickFallbackSculptureName();
      this.daisSlot.isNamingSculpture = false;
      this.daisSlot.sculptureName = fallback;
      this.daisSlot.sculptureNameIsGemini = false;
      this.updateDaisLecternTexture();
      this.onDaisSculptureNamed?.(fallback);
      return;
    }

    const result = await generateSculptureNameForDais({
      imagesBase64Jpeg: snapshots,
      assemblyMode: this.daisAssemblyMode,
    });

    if (!this.daisSlot || this.daisSlot.namingRequestId !== requestId) return;

    this.daisSlot.isNamingSculpture = false;
    this.daisSlot.sculptureName = result.name;
    this.daisSlot.sculptureNameIsGemini = result.fromGemini;
    this.updateDaisLecternTexture();
    this.onDaisSculptureNamed?.(result.name);
  }

  /**
   * Renders open-source paintings & specimen plates from Hugging Face image datasets
   * onto the framed wall canvases across the museum gallery (Front bay wall, side wings,
   * and rear entrance wall).
   *
   * Zero-repeat cycling & resilient loading:
   * - Keeps track of currently displayed art IDs in `activeArtIds`.
   * - Maintains a rolling FIFO history (`recentArtHistory`) of up to 100 items so the
   *   same artwork is not seen again for a long cycle across all frames.
   * - If an image candidate fails to download or times out (4.5s), automatically retries
   *   with subsequent candidate artworks so NO frame is ever left blank.
   * - Falls back to a bespoke procedural museum exhibition plate if offline.
   */
  public async applyWallArtworks(
    artworks: WallArtworkItem[],
    singleFrameUpdate = false
  ): Promise<void> {
    if (artworks.length === 0 || this.wallPaintingMeshes.length === 0) return;

    const indicesToUpdate = singleFrameUpdate
      ? [this.nextWallFrameIndex++ % this.wallPaintingMeshes.length]
      : this.wallPaintingMeshes.map((_, idx) => idx);

    await Promise.all(
      indicesToUpdate.map(async (frameIdx) => {
        const mesh = this.wallPaintingMeshes[frameIdx];
        if (!mesh) return;

        let loaded = false;
        let attempts = 0;
        const maxAttempts = Math.min(12, artworks.length);

        while (!loaded && attempts < maxAttempts) {
          attempts++;

          // 1. Pick candidate artwork avoiding activeArtIds and recentArtHistory
          let chosenArt: WallArtworkItem | null = null;
          const total = artworks.length;

          for (let i = 0; i < total; i++) {
            const idx = (this.artCursor + i) % total;
            const candidate = artworks[idx];
            if (!candidate) continue;

            if (!this.activeArtIds.has(candidate.id) && !this.recentArtHistory.includes(candidate.id)) {
              chosenArt = candidate;
              this.artCursor = (idx + 1) % total;
              break;
            }
          }

          // Relax recent history if necessary
          if (!chosenArt) {
            for (let i = 0; i < total; i++) {
              const idx = (this.artCursor + i) % total;
              const candidate = artworks[idx];
              if (candidate && !this.activeArtIds.has(candidate.id)) {
                chosenArt = candidate;
                this.artCursor = (idx + 1) % total;
                break;
              }
            }
          }

          // Final fallback candidate
          if (!chosenArt) {
            chosenArt = artworks[this.artCursor % total];
            this.artCursor = (this.artCursor + 1) % total;
          }

          if (!chosenArt) break;

          const art = chosenArt;

          // Attempt loading candidate artwork image with timeout
          try {
            const img = new Image();
            img.crossOrigin = 'anonymous';

            await new Promise<void>((resolve, reject) => {
              const timer = setTimeout(() => {
                img.onload = null;
                img.onerror = null;
                reject(new Error(`Timeout loading image ${art.imageUrl}`));
              }, 4500);

              img.onload = () => {
                clearTimeout(timer);
                resolve();
              };
              img.onerror = () => {
                clearTimeout(timer);
                reject(new Error(`Failed to load image ${art.imageUrl}`));
              };
              img.src = art.imageUrl;
            });

            // Render artwork onto canvas texture
            const c = document.createElement('canvas');
            c.width = 512;
            c.height = 320;
            const ctx = c.getContext('2d')!;

            // Center-crop artwork into the 512×282 upper canvas region
            const targetW = 512;
            const targetH = 282;
            const imgAspect = img.width / Math.max(1, img.height);
            const boxAspect = targetW / targetH;
            let sx = 0;
            let sy = 0;
            let sw = img.width;
            let sh = img.height;
            if (imgAspect > boxAspect) {
              sw = img.height * boxAspect;
              sx = (img.width - sw) * 0.5;
            } else {
              sh = img.width / boxAspect;
              sy = (img.height - sh) * 0.35;
            }
            ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetW, targetH);

            // Subtle museum brass/espresso nameplate strip at bottom (38px tall)
            ctx.fillStyle = '#181e29';
            ctx.fillRect(0, 282, 512, 38);
            ctx.fillStyle = '#d4a359';
            ctx.fillRect(0, 282, 512, 2);

            ctx.fillStyle = '#f3f4f6';
            ctx.font = 'bold 14px system-ui, -apple-system, sans-serif';
            const label = `${art.title.slice(0, 38)} — ${art.subtitle.slice(0, 32)}`;
            ctx.fillText(label, 14, 306);

            const newTex = new THREE.CanvasTexture(c);
            newTex.colorSpace = THREE.SRGBColorSpace;

            const mat = mesh.material as THREE.MeshStandardMaterial;
            if (mat.map) {
              this.disposeTexture(mat.map);
            }
            mat.map = newTex;
            mat.needsUpdate = true;
            mesh.visible = true;

            // Update active and recent art history tracking
            const oldArtId = mesh.userData.artId as string | undefined;
            if (oldArtId) {
              this.activeArtIds.delete(oldArtId);
            }
            mesh.userData.artId = art.id;
            this.activeArtIds.add(art.id);

            this.recentArtHistory.push(art.id);
            const maxHistory = Math.min(100, Math.floor(artworks.length * 0.7));
            if (this.recentArtHistory.length > maxHistory) {
              this.recentArtHistory.shift();
            }

            loaded = true;
          } catch {
            // Put candidate into recent history so we do not immediately loop back to it
            this.recentArtHistory.push(art.id);
            // Loop automatically continues to next candidate!
          }
        }

        // If all candidates failed (e.g. offline), ensure procedural fallback is rendered
        if (!loaded && (!mesh.userData.artId || !(mesh.material as THREE.MeshStandardMaterial).map)) {
          this.applyProceduralArtworkFallback(mesh, frameIdx);
        }
      })
    );
  }

  /**
   * Generates a museum-grade archival exhibition plate texture so no wall frame
   * is ever untextured, white, or blank in the showroom.
   */
  private applyProceduralArtworkFallback(mesh: THREE.Mesh, frameIdx: number): void {
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 320;
    const ctx = c.getContext('2d')!;

    // Rich dark archival museum palettes
    const palettes = [
      ['#14213d', '#080d19'],
      ['#2d1822', '#12070d'],
      ['#16291e', '#09130d'],
      ['#2d2416', '#140f08'],
      ['#21162d', '#0d0814'],
      ['#1a2a30', '#0a1215'],
    ];
    const [c1, c2] = palettes[frameIdx % palettes.length];

    const grad = ctx.createLinearGradient(0, 0, 512, 282);
    grad.addColorStop(0, c1);
    grad.addColorStop(1, c2);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 512, 282);

    // Architectural gallery framing in antique brass
    ctx.strokeStyle = '#d4a359';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(36, 36, 440, 210);
    ctx.strokeStyle = 'rgba(212, 163, 89, 0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(44, 44, 424, 194);

    // Central museum typography
    ctx.textAlign = 'center';
    ctx.fillStyle = '#f3f4f6';
    ctx.font = 'italic 20px Georgia, serif';
    ctx.fillText('Permanent Gallery Collection', 256, 126);

    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.fillText('Hugging Face Curated Fine Art & Open Specimens', 256, 160);

    // Nameplate at bottom
    ctx.textAlign = 'left';
    ctx.fillStyle = '#181e29';
    ctx.fillRect(0, 282, 512, 38);
    ctx.fillStyle = '#d4a359';
    ctx.fillRect(0, 282, 512, 2);

    ctx.fillStyle = '#f3f4f6';
    ctx.font = 'bold 14px system-ui, -apple-system, sans-serif';
    ctx.fillText('Gallery Exhibition Archive · Study Room', 14, 306);

    const newTex = new THREE.CanvasTexture(c);
    newTex.colorSpace = THREE.SRGBColorSpace;

    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (mat.map) {
      this.disposeTexture(mat.map);
    }
    mat.map = newTex;
    mat.needsUpdate = true;
    mesh.visible = true;
  }

  private setFocusedSlotHighlight(slotIndex: number, notifyHUD = false): void {
    if (this.focusedSlotIndex === slotIndex && !notifyHUD) return;
    this.focusedSlotIndex = slotIndex;

    for (const s of this.slots) {
      const mat = s.ringMesh.material as THREE.MeshBasicMaterial;
      mat.color.setHex(s.index === slotIndex ? 0xfbbf24 : 0x38bdf8);
      this.updateSlotLecternTexture(s);
    }

    if (this.daisSlot) {
      const daisMat = this.daisSlot.ringMesh.material as THREE.MeshBasicMaterial;
      daisMat.color.setHex(slotIndex === DAIS_SLOT_INDEX ? 0xfbbf24 : 0xc084fc);
      this.updateDaisLecternTexture();
    }

    if (notifyHUD) {
      this.onPedestalSelect?.(slotIndex);
    }
  }

  /**
   * Automatically detects which plinth (or the Central Dais) the visitor is standing in front of
   * while walking around the gallery, and updates the active plinth ring, lectern plaque,
   * and top HUD bar (#1–#4 + ⚡ Dais).
   */
  private lastDaisProximity = false;

  public isInDaisProximity(): boolean {
    if (this.focusedSlotIndex !== DAIS_SLOT_INDEX) return false;
    if (this.camGlideActive) return true;
    const dx = this.camera.position.x - DAIS_POSITION.x;
    const dz = this.camera.position.z - DAIS_POSITION.z;
    return Math.hypot(dx, dz) <= 5.2;
  }

  private updateActivePedestalByProximity(): void {
    if (this.camGlideActive) return;

    const camX = this.camera.position.x;
    const camZ = this.camera.position.z;
    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    forward.y = 0;
    if (forward.lengthSq() > 0.0001) {
      forward.normalize();
    }

    let bestSlot = -1;
    let bestScore = -Infinity;

    const candidates: Array<{ index: number; position: THREE.Vector3 }> = [
      ...this.slots.map((s) => ({ index: s.index, position: s.position })),
    ];
    if (this.daisSlot) {
      candidates.push({ index: DAIS_SLOT_INDEX, position: this.daisSlot.position });
    }

    for (const slot of candidates) {
      const dx = slot.position.x - camX;
      const dz = slot.position.z - camZ;
      const horizDist = Math.hypot(dx, dz);
      if (horizDist > 10.5) continue;

      const dot =
        horizDist > 0.001 ? (dx * forward.x + dz * forward.z) / horizDist : 1.0;

      // Require the plinth to be generally in front of our gaze (or right next to us)
      if (dot < 0.2 && horizDist > 2.4) continue;

      const lateralOffset = Math.abs(camX - slot.position.x);
      const inBayBonus = lateralOffset < 2.2 ? 0.65 : -lateralOffset * 0.12;
      const hysteresisBonus = slot.index === this.focusedSlotIndex ? 0.18 : 0;

      const score = dot * 2.4 - horizDist * 0.34 + inBayBonus + hysteresisBonus;
      if (score > bestScore) {
        bestScore = score;
        bestSlot = slot.index;
      }
    }

    if (bestSlot !== -1 && bestSlot !== this.focusedSlotIndex) {
      this.setFocusedSlotHighlight(bestSlot, true);
    }

    const nowNearDais = this.isInDaisProximity();
    if (nowNearDais !== this.lastDaisProximity) {
      this.lastDaisProximity = nowNearDais;
      this.onPedestalSelect?.(this.focusedSlotIndex);
    }
  }

  /**
   * Periodically triggers gallery remarks with organic randomized timing (every 28s–48s):
   * - If the visitor is in front of / looking at a plinth (even if moving around or rotating it),
   *   the remark plays directly over their shoulder as if an observer is right behind them.
   * - Otherwise, a remark can trigger from any of the other 3 exhibits across the gallery,
   *   spatially anchored at that pedestal and naturally diminishing with distance.
   */
  private updateGalleryWhispers(delta: number): void {
    if (this.camGlideActive) return;

    this.galleryWhisperTimer += delta;
    if (this.galleryWhisperTimer < this.nextGalleryWhisperInterval) return;

    // Reset timer with new randomized interval (30s to 50s)
    this.galleryWhisperTimer = 0;
    this.nextGalleryWhisperInterval = 30 + Math.random() * 20;

    const camX = this.camera.position.x;
    const camZ = this.camera.position.z;

    // Eligible slots that have a loaded 3D model
    const eligibleSlots = this.slots.filter((s) => s.asset && s.modelRoot);
    if (eligibleSlots.length === 0) return;

    const activeSlot = this.slots[this.focusedSlotIndex];
    const distToActive = activeSlot
      ? Math.hypot(activeSlot.position.x - camX, activeSlot.position.z - camZ)
      : Infinity;

    // Visitor counts as "at the exhibit" if within 5.8m of the focused plinth
    // (even if pacing back and forth, circling, or rotating the model)
    const isAtActiveExhibit = Boolean(activeSlot?.asset && activeSlot?.modelRoot && distToActive <= 5.8);

    let chosenSlot: InternalPedestalSlot;
    let isShoulder: boolean;

    // If standing in front of an exhibit, 70% chance to hear a shoulder remark about this exhibit,
    // and 30% chance to hear a distant whisper drifting from another pedestal across the room
    if (isAtActiveExhibit && (Math.random() < 0.7 || eligibleSlots.length === 1)) {
      chosenSlot = activeSlot;
      isShoulder = true;
    } else {
      // Pick another exhibit across the gallery
      const otherSlots = eligibleSlots.filter((s) => s.index !== this.focusedSlotIndex);
      chosenSlot =
        otherSlots.length > 0
          ? otherSlots[Math.floor(Math.random() * otherSlots.length)]
          : eligibleSlots[0];
      isShoulder = false;
    }

    if (!chosenSlot.asset) return;

    let remark: string | undefined;
    if (chosenSlot.customVisitorComments.length > 0) {
      const idx = chosenSlot.commentPlayIndex % chosenSlot.customVisitorComments.length;
      remark = chosenSlot.customVisitorComments[idx];
      chosenSlot.commentPlayIndex++;
    }

    this.audio.playOverShoulderWhisper(
      remark,
      {
        repoId: chosenSlot.asset.repoId,
        triangles: chosenSlot.stats?.triangles,
        scaleFactor: chosenSlot.stats?.scaleFactor,
        wireframe: this.wireframeEnabled,
      },
      this.camera,
      { isShoulder, slotIndex: chosenSlot.index }
    );
  }

  public triggerVisitorWhisper(customText?: string): string | null {
    const activeSlot = this.slots[this.focusedSlotIndex];
    return this.audio.playOverShoulderWhisper(
      customText,
      {
        repoId: activeSlot?.asset?.repoId,
        triangles: activeSlot?.stats?.triangles,
        scaleFactor: activeSlot?.stats?.scaleFactor,
        wireframe: this.wireframeEnabled,
      },
      this.camera
    );
  }

  public focusPedestal(slotIndex: number, animate = true): void {
    if (slotIndex === DAIS_SLOT_INDEX && this.daisSlot) {
      const targetCenter = new THREE.Vector3(
        DAIS_POSITION.x,
        DAIS_HEIGHT + 0.65,
        DAIS_POSITION.z
      );
      const desiredCamPos = new THREE.Vector3(
        DAIS_POSITION.x,
        DAIS_HEIGHT + 1.24,
        DAIS_POSITION.z + 3.55
      );
      if (animate) {
        this.camGlideGoalTarget.copy(targetCenter);
        this.camGlideGoalPos.copy(desiredCamPos);
        this.camGlideActive = true;
      } else {
        this.controls.target.copy(targetCenter);
        this.camera.position.copy(desiredCamPos);
        this.controls.update();
      }
      this.setFocusedSlotHighlight(DAIS_SLOT_INDEX, false);
      this.lastDaisProximity = this.isInDaisProximity();
      return;
    }

    const slot = this.slots[slotIndex];
    if (!slot) return;

    const targetCenter = new THREE.Vector3(
      slot.position.x,
      PLINTH_HEIGHT + 0.55,
      slot.position.z
    );
    const desiredCamPos = new THREE.Vector3(
      slot.position.x,
      PLINTH_HEIGHT + 1.02,
      slot.position.z + 3.05
    );

    if (animate) {
      this.camGlideGoalTarget.copy(targetCenter);
      this.camGlideGoalPos.copy(desiredCamPos);
      this.camGlideActive = true;
    } else {
      this.controls.target.copy(targetCenter);
      this.camera.position.copy(desiredCamPos);
      this.controls.update();
    }
    this.setFocusedSlotHighlight(slotIndex, false);
    this.lastDaisProximity = this.isInDaisProximity();
  }

  private updateSize(): void {
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  /**
   * Triggers the theatrical 5-part Plinth Reveal Combo Effect whenever a new 3D model
   * replaces an existing item on a pedestal:
   * 1) Overhead Spotlight Flare & Volumetric Beam Swell
   * 2) Plinth Elevator Rise (gliding up from inside the marble cap)
   * 3) Low-Lying Marble-Cap Velvet Smoke Puff rolling outward
   * 4) Rising Golden Dust Motes spiraling up the light beam
   * 5) Expanding Brass Halo Ripple across the plinth rim
   */
  private triggerSlotRevealEffect(slot: InternalPedestalSlot): void {
    slot.revealTimer = REVEAL_DURATION;
    slot.haloRippleMesh.visible = true;
    slot.smokePuffPoints.visible = true;
    slot.dustMotesPoints.visible = true;

    // Start the model slightly recessed inside the plinth cap so it rises up smoothly
    slot.modelPivot.position.y = PLINTH_HEIGHT - 0.22;
    slot.modelPivot.scale.setScalar(0.76);

    if (this.mode === 'showroom') {
      this.audio.updateListener(this.camera);
      this.audio.playPlinthReveal(slot.index);
    }
  }

  private updateSlotRevealEffects(delta: number): void {
    for (const slot of this.slots) {
      if (slot.revealTimer <= 0) continue;

      slot.revealTimer = Math.max(0, slot.revealTimer - delta);
      const t = 1 - slot.revealTimer / REVEAL_DURATION; // 0 -> 1 normalized progress
      const easeOutCubic = 1 - Math.pow(1 - t, 3);
      const easeOutQuart = 1 - Math.pow(1 - t, 4);

      // 1. Plinth Elevator Rise & Scale-in
      slot.modelPivot.position.y = THREE.MathUtils.lerp(
        PLINTH_HEIGHT - 0.22,
        PLINTH_HEIGHT + 0.054,
        easeOutQuart
      );
      slot.modelPivot.scale.setScalar(THREE.MathUtils.lerp(0.76, 1.0, easeOutCubic));

      // 2. Overhead Spotlight Flare & Volumetric Light Cone Swell
      const flareEnvelope = Math.sin(Math.pow(t, 0.58) * Math.PI); // Quick bright swell, smooth settle
      slot.bayLight.intensity = 2.2 + flareEnvelope * 6.8;
      slot.lightConeMat.opacity = 0.045 + flareEnvelope * 0.28;

      // 3. Expanding Brass Halo Ripple across the marble cap
      const rippleScale = 0.55 + easeOutCubic * 0.92;
      slot.haloRippleMesh.scale.set(rippleScale, rippleScale, 1);
      slot.haloRippleMat.opacity = (1 - easeOutCubic) * 0.88;

      // 4. Low-Lying Marble-Cap Velvet Smoke Puff (36 particles rolling outward & curling slightly)
      const smokePosAttr = slot.smokePuffPoints.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      const smokeArr = smokePosAttr.array as Float32Array;
      const smokeCount = smokeArr.length / 3;
      for (let s = 0; s < smokeCount; s++) {
        const baseAngle = slot.smokeSeeds[s * 4];
        const speed = slot.smokeSeeds[s * 4 + 1];
        const heightDrift = slot.smokeSeeds[s * 4 + 2];
        const phase = slot.smokeSeeds[s * 4 + 3];

        const radius = 0.14 + easeOutCubic * 0.82 * speed;
        const swirlAngle = baseAngle + easeOutCubic * 0.45;
        smokeArr[s * 3] = Math.cos(swirlAngle) * radius;
        smokeArr[s * 3 + 1] =
          0.02 + easeOutCubic * heightDrift + Math.sin(t * 5 + phase) * 0.025;
        smokeArr[s * 3 + 2] = Math.sin(swirlAngle) * radius;
      }
      smokePosAttr.needsUpdate = true;
      slot.smokePuffMat.opacity = Math.sin(Math.pow(t, 0.65) * Math.PI) * 0.52;

      // 5. Rising Golden Dust Motes spiraling up into the spotlight cone (48 particles)
      const dustPosAttr = slot.dustMotesPoints.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      const dustArr = dustPosAttr.array as Float32Array;
      const dustCount = dustArr.length / 3;
      for (let d = 0; d < dustCount; d++) {
        const baseAngle = slot.dustSeeds[d * 4];
        const baseRadius = slot.dustSeeds[d * 4 + 1];
        const baseHeight = slot.dustSeeds[d * 4 + 2];
        const upSpeed = slot.dustSeeds[d * 4 + 3];

        const spiralAngle = baseAngle + t * 2.6 * upSpeed;
        const radius = baseRadius * (1 - t * 0.22);
        dustArr[d * 3] = Math.cos(spiralAngle) * radius;
        dustArr[d * 3 + 1] = baseHeight + easeOutCubic * 1.45 * upSpeed;
        dustArr[d * 3 + 2] = Math.sin(spiralAngle) * radius;
      }
      dustPosAttr.needsUpdate = true;
      slot.dustMotesMat.opacity = Math.sin(Math.pow(t, 0.55) * Math.PI) * 0.92;

      // Clean reset when animation completes
      if (slot.revealTimer === 0) {
        slot.modelPivot.position.y = PLINTH_HEIGHT + 0.054;
        slot.modelPivot.scale.setScalar(1.0);
        slot.bayLight.intensity = 2.2;
        slot.lightConeMat.opacity = 0.045;
        slot.haloRippleMesh.visible = false;
        slot.smokePuffPoints.visible = false;
        slot.dustMotesPoints.visible = false;
      }
    }

    // Update Central Dais reveal animation if active
    if (this.daisSlot && this.daisSlot.revealTimer > 0) {
      const dais = this.daisSlot;
      dais.revealTimer = Math.max(0, dais.revealTimer - delta);
      const t = 1 - dais.revealTimer / REVEAL_DURATION;
      const easeOutCubic = 1 - Math.pow(1 - t, 3);
      const easeOutQuart = 1 - Math.pow(1 - t, 4);

      dais.modelPivot.position.y = THREE.MathUtils.lerp(
        DAIS_HEIGHT - 0.22,
        DAIS_HEIGHT + 0.054,
        easeOutQuart
      );
      dais.modelPivot.scale.setScalar(THREE.MathUtils.lerp(0.74, 1.0, easeOutCubic));

      const flareEnvelope = Math.sin(Math.pow(t, 0.58) * Math.PI);
      dais.daisLight.intensity = 2.5 + flareEnvelope * 7.5;
      dais.lightConeMat.opacity = 0.05 + flareEnvelope * 0.3;

      const rippleScale = 0.55 + easeOutCubic * 0.95;
      dais.haloRippleMesh.scale.set(rippleScale, rippleScale, 1);
      dais.haloRippleMat.opacity = (1 - easeOutCubic) * 0.9;

      const smokePosAttr = dais.smokePuffPoints.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      const smokeArr = smokePosAttr.array as Float32Array;
      const smokeCount = smokeArr.length / 3;
      for (let s = 0; s < smokeCount; s++) {
        const baseAngle = dais.smokeSeeds[s * 4];
        const speed = dais.smokeSeeds[s * 4 + 1];
        const heightDrift = dais.smokeSeeds[s * 4 + 2];
        const phase = dais.smokeSeeds[s * 4 + 3];
        const radius = 0.2 + easeOutCubic * 1.15 * speed;
        const swirlAngle = baseAngle + easeOutCubic * 0.45;
        smokeArr[s * 3] = Math.cos(swirlAngle) * radius;
        smokeArr[s * 3 + 1] =
          0.02 + easeOutCubic * heightDrift + Math.sin(t * 5 + phase) * 0.025;
        smokeArr[s * 3 + 2] = Math.sin(swirlAngle) * radius;
      }
      smokePosAttr.needsUpdate = true;
      dais.smokePuffMat.opacity = Math.sin(Math.pow(t, 0.65) * Math.PI) * 0.55;

      const dustPosAttr = dais.dustMotesPoints.geometry.getAttribute(
        'position'
      ) as THREE.BufferAttribute;
      const dustArr = dustPosAttr.array as Float32Array;
      const dustCount = dustArr.length / 3;
      for (let d = 0; d < dustCount; d++) {
        const baseAngle = dais.dustSeeds[d * 4];
        const baseRadius = dais.dustSeeds[d * 4 + 1];
        const baseHeight = dais.dustSeeds[d * 4 + 2];
        const upSpeed = dais.dustSeeds[d * 4 + 3];
        const spiralAngle = baseAngle + t * 2.6 * upSpeed;
        const radius = baseRadius * (1 - t * 0.2);
        dustArr[d * 3] = Math.cos(spiralAngle) * radius;
        dustArr[d * 3 + 1] = baseHeight + easeOutCubic * 1.65 * upSpeed;
        dustArr[d * 3 + 2] = Math.sin(spiralAngle) * radius;
      }
      dustPosAttr.needsUpdate = true;
      dais.dustMotesMat.opacity = Math.sin(Math.pow(t, 0.55) * Math.PI) * 0.92;

      if (dais.revealTimer === 0) {
        dais.modelPivot.position.y = DAIS_HEIGHT + 0.054;
        dais.modelPivot.scale.setScalar(1.0);
        dais.daisLight.intensity = 2.5;
        dais.lightConeMat.opacity = 0.05;
        dais.haloRippleMesh.visible = false;
        dais.smokePuffPoints.visible = false;
        dais.dustMotesPoints.visible = false;
      }
    }
  }

  private renderTick(): void {
    if (document.hidden) return;
    const delta = Math.min(this.clock.getDelta(), 0.1);

    if (this.mode === 'studio') {
      if (this.studioMixer) {
        this.studioMixer.update(delta);
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // SHOWROOM MODE TICK:
    this.updateShowroomWalking(delta);
    this.updateSlotRevealEffects(delta);

    if (this.camGlideActive) {
      const prevX = this.camera.position.x;
      const prevZ = this.camera.position.z;
      const alpha = 1 - Math.exp(-6.5 * delta);
      this.camera.position.lerp(this.camGlideGoalPos, alpha);
      this.controls.target.lerp(this.camGlideGoalTarget, alpha);

      const glideStepDist = Math.hypot(
        this.camera.position.x - prevX,
        this.camera.position.z - prevZ
      );
      if (glideStepDist > 0.001) {
        this.stepDistanceAccum += glideStepDist;
        if (this.stepDistanceAccum >= 0.72) {
          this.stepDistanceAccum %= 0.72;
          this.audio.playFootstep(this.camera.position.x, this.camera.position.z);
        }
      }

      if (
        this.camera.position.distanceToSquared(this.camGlideGoalPos) < 0.002 &&
        this.controls.target.distanceToSquared(this.camGlideGoalTarget) < 0.002
      ) {
        this.camGlideActive = false;
      }
    }

    this.controls.update();
    this.audio.updateListener(this.camera);
    this.updateActivePedestalByProximity();
    this.updateGalleryWhispers(delta);

    const states: PedestalSlotState[] = [];
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    const projVec = new THREE.Vector3();

    for (const slot of this.slots) {
      const dist = this.camera.position.distanceTo(slot.position);
      const nextTier: LodTier = dist < 5.8 ? 'close' : dist < 11.2 ? 'medium' : 'far';

      if (slot.lodTier !== nextTier && slot.modelRoot) {
        this.applySlotLodTier(slot, nextTier);
      }

      if (slot.modelRoot && slot.lodTier === 'close') {
        if (slot.mixer) {
          slot.mixer.update(delta);
        }
        if (this.autoRotateEnabled) {
          slot.modelPivot.rotation.y += delta * 0.45;
        }
      }

      slot.placardAnchor.getWorldPosition(projVec);
      projVec.project(this.camera);

      const inFront = projVec.z >= -1 && projVec.z <= 1;
      const screenX = (projVec.x * 0.5 + 0.5) * width;
      const screenY = (-(projVec.y * 0.5) + 0.5) * height;
      const visibleOnScreen =
        Boolean(slot.asset) &&
        inFront &&
        screenX >= -120 &&
        screenX <= width + 120 &&
        screenY >= -80 &&
        screenY <= height + 80;

      states.push({
        slotIndex: slot.index,
        asset: slot.asset,
        datasetMeta: slot.datasetMeta,
        stats: slot.stats,
        lodTier: slot.lodTier,
        distance: Math.round(dist * 10) / 10,
        screenX,
        screenY,
        visibleOnScreen,
        isFocused: slot.index === this.focusedSlotIndex,
      });
    }

    if (this.daisSlot?.modelRoot && this.autoRotateEnabled) {
      this.daisSlot.modelPivot.rotation.y += delta * 0.36;
    }

    this.onShowroomProjectionUpdate?.(states);
    this.renderer.render(this.scene, this.camera);
  }

  private updateShowroomWalking(delta: number): void {
    const { forward, backward, left, right } = this.moveKeys;
    if (!forward && !backward && !left && !right) {
      this.wasWalkingLastFrame = false;
      return;
    }

    if (!this.wasWalkingLastFrame) {
      this.wasWalkingLastFrame = true;
      // Prime the stride accumulator so the first step plays promptly when starting to walk
      this.stepDistanceAccum = 0.44;
    }

    const speed = 5.6 * delta;
    const forwardVec = new THREE.Vector3();
    this.camera.getWorldDirection(forwardVec);
    forwardVec.y = 0;
    if (forwardVec.lengthSq() > 0.0001) {
      forwardVec.normalize();
    }

    const rightVec = new THREE.Vector3()
      .crossVectors(forwardVec, THREE.Object3D.DEFAULT_UP)
      .normalize();
    const moveDelta = new THREE.Vector3();

    if (forward) moveDelta.add(forwardVec);
    if (backward) moveDelta.sub(forwardVec);
    if (right) moveDelta.add(rightVec);
    if (left) moveDelta.sub(rightVec);

    if (moveDelta.lengthSq() > 0) {
      moveDelta.normalize().multiplyScalar(speed);
      const nextCam = this.camera.position.clone().add(moveDelta);
      // Keep viewer comfortably inside the 4-bay museum hall walls
      nextCam.x = THREE.MathUtils.clamp(nextCam.x, -9.4, 9.4);
      nextCam.z = THREE.MathUtils.clamp(nextCam.z, -4.6, 6.6);
      nextCam.y = THREE.MathUtils.clamp(nextCam.y, 0.85, 4.2);

      const appliedDelta = nextCam.sub(this.camera.position);
      this.camera.position.add(appliedDelta);
      this.controls.target.add(appliedDelta);

      const horizStep = Math.hypot(appliedDelta.x, appliedDelta.z);
      if (horizStep > 0.0005) {
        this.stepDistanceAccum += horizStep;
        if (this.stepDistanceAccum >= 0.56) {
          this.stepDistanceAccum -= 0.56;
          this.audio.playFootstep(this.camera.position.x, this.camera.position.z);
        }
      }
    }
  }

  private applySlotLodTier(slot: InternalPedestalSlot, tier: LodTier): void {
    slot.lodTier = tier;
    this.updateSlotLecternTexture(slot);
    if (!slot.modelRoot) return;

    slot.modelRoot.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;

      if (tier === 'close') {
        const orig = slot.originalMaterials.get(mesh);
        if (orig) mesh.material = orig;
        mesh.castShadow = true;
      } else if (tier === 'medium') {
        mesh.material = this.mediumProxyMaterial;
        mesh.castShadow = false;
      } else {
        mesh.material = this.farProxyMaterial;
        mesh.castShadow = false;
      }
    });
  }

  private clearSlot(slot: InternalPedestalSlot): void {
    if (slot.mixer) {
      slot.mixer.stopAllAction();
      if (slot.modelRoot) {
        slot.mixer.uncacheRoot(slot.modelRoot);
      }
      slot.mixer = null;
    }

    if (slot.modelRoot) {
      for (const [mesh, origMat] of slot.originalMaterials.entries()) {
        mesh.material = origMat;
      }
      slot.originalMaterials.clear();
      slot.modelPivot.remove(slot.modelRoot);
      this.disposeObject3D(slot.modelRoot);
      slot.modelRoot = null;
    }

    slot.revealTimer = 0;
    slot.whisperedForCurrentAsset = false;
    slot.customVisitorComments = [];
    slot.commentPlayIndex = 0;
    slot.modelPivot.position.y = PLINTH_HEIGHT + 0.054;
    slot.modelPivot.scale.setScalar(1.0);
    slot.bayLight.intensity = 2.2;
    slot.lightConeMat.opacity = 0.045;
    slot.haloRippleMesh.visible = false;
    slot.smokePuffPoints.visible = false;
    slot.dustMotesPoints.visible = false;
    slot.contactShadowMesh.visible = false;
    slot.asset = null;
    slot.datasetMeta = null;
    slot.stats = null;
    this.renderer.renderLists.dispose();
    THREE.Cache.clear();
  }

  public clearCurrentModel(): void {
    if (this.studioMixer) {
      this.studioMixer.stopAllAction();
      if (this.currentModelRoot) {
        this.studioMixer.uncacheRoot(this.currentModelRoot);
      }
      this.studioMixer = null;
    }

    if (this.currentModelRoot) {
      this.studioPivot.remove(this.currentModelRoot);
      this.disposeObject3D(this.currentModelRoot);
      this.currentModelRoot = null;
    }
    this.studioContactShadow.visible = false;

    this.renderer.renderLists.dispose();
    THREE.Cache.clear();
  }

  private disposeTexture(tex: THREE.Texture): void {
    if (tex.userData?.isPersistentProcedural) return;
    const sourceData = tex.source?.data as { close?: () => void } | undefined;
    if (sourceData && typeof sourceData.close === 'function') {
      try {
        sourceData.close();
      } catch {
        // Ignore if already closed
      }
    }
    tex.dispose();
  }

  private disposeMaterial(mat: THREE.Material): void {
    if (mat === this.mediumProxyMaterial || mat === this.farProxyMaterial) return;
    for (const value of Object.values(mat)) {
      if (value && typeof value === 'object' && 'isTexture' in value) {
        this.disposeTexture(value as THREE.Texture);
      }
    }
    const shaderMat = mat as THREE.ShaderMaterial;
    if (shaderMat.uniforms && typeof shaderMat.uniforms === 'object') {
      for (const u of Object.values(shaderMat.uniforms)) {
        const val = u?.value;
        if (val && typeof val === 'object' && 'isTexture' in val) {
          this.disposeTexture(val as THREE.Texture);
        }
      }
    }
    mat.dispose();
  }

  private disposeObject3D(root: THREE.Object3D): void {
    root.traverse((obj) => {
      const anyObj = obj as THREE.Mesh & THREE.SkinnedMesh & THREE.Line & THREE.Points;

      if (anyObj.geometry && typeof anyObj.geometry.dispose === 'function') {
        anyObj.geometry.dispose();
      }

      if (anyObj.isSkinnedMesh && anyObj.skeleton) {
        anyObj.skeleton.dispose();
      }

      if (anyObj.material) {
        const materials = Array.isArray(anyObj.material) ? anyObj.material : [anyObj.material];
        for (const mat of materials) {
          if (mat) this.disposeMaterial(mat);
        }
      }
    });
  }

  private async fetchArrayBufferWithProgress(
    url: string,
    signal: AbortSignal,
    onProgress?: (percent: number | null) => void
  ): Promise<ArrayBuffer> {
    const timeoutController = new AbortController();
    const timeoutId = window.setTimeout(() => timeoutController.abort(), 9000);
    const onParentAbort = () => timeoutController.abort();
    signal.addEventListener('abort', onParentAbort, { once: true });

    try {
      const response = await fetch(url, { signal: timeoutController.signal });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching ${url}`);
      }

      const contentLength = Number(response.headers.get('content-length') || 0);
      if (contentLength > 20 * 1024 * 1024) {
        throw new Error(`Model file too large (${Math.round(contentLength / 1048576)}MB)`);
      }
      if (!response.body || contentLength <= 0) {
        onProgress?.(null);
        return await response.arrayBuffer();
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let receivedLength = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          receivedLength += value.byteLength;
          onProgress?.(Math.min(100, Math.round((receivedLength / contentLength) * 100)));
        }
      }

      const combined = new Uint8Array(receivedLength);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return combined.buffer;
    } finally {
      window.clearTimeout(timeoutId);
      signal.removeEventListener('abort', onParentAbort);
    }
  }

  private async parseAssetGeometry(
    asset: Discovered3DAsset,
    signal: AbortSignal,
    onProgress?: (percent: number | null) => void
  ): Promise<{
    rootObject: THREE.Object3D;
    animations: THREE.AnimationClip[];
    stats: LoadedModelStats;
  }> {
    let rootObject: THREE.Object3D;
    let animations: THREE.AnimationClip[] = [];

    if (asset.format === 'glb' || asset.format === 'gltf') {
      const buffer = await this.fetchArrayBufferWithProgress(
        asset.downloadUrl,
        signal,
        onProgress
      );
      if (signal.aborted) throw new Error('Load aborted');
      const gltf = await new Promise<{ scene: THREE.Group; animations: THREE.AnimationClip[] }>(
        (resolve, reject) => {
          this.gltfLoader.parse(buffer, asset.resourceBaseUrl, resolve, reject);
        }
      );
      rootObject = gltf.scene || new THREE.Group();
      animations = gltf.animations || [];
    } else if (asset.format === 'obj') {
      const buffer = await this.fetchArrayBufferWithProgress(
        asset.downloadUrl,
        signal,
        onProgress
      );
      if (signal.aborted) throw new Error('Load aborted');
      const text = new TextDecoder().decode(buffer);
      const objGroup = this.objLoader.parse(text);
      this.applyStudioMaterialIfMissing(objGroup);
      rootObject = objGroup;
    } else if (asset.format === 'stl') {
      const buffer = await this.fetchArrayBufferWithProgress(
        asset.downloadUrl,
        signal,
        onProgress
      );
      if (signal.aborted) throw new Error('Load aborted');
      const geometry = this.stlLoader.parse(buffer);
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({
        color: 0xd1d5db,
        roughness: 0.38,
        metalness: 0.18,
      });
      material.userData.isFallbackStudioMaterial = true;
      const mesh = new THREE.Mesh(geometry, material);
      const group = new THREE.Group();
      group.add(mesh);
      rootObject = group;
    } else {
      throw new Error(`Unsupported 3D format: ${asset.format}`);
    }

    if (signal.aborted) {
      this.disposeObject3D(rootObject);
      throw new Error('Load aborted');
    }

    const stats = this.computeStatsAndPrepareMeshes(rootObject);
    if (stats.meshes === 0 || stats.vertices === 0) {
      this.disposeObject3D(rootObject);
      throw new Error('Loaded 3D file contained no valid mesh geometry.');
    }

    return { rootObject, animations, stats };
  }

  public async loadAsset(
    asset: Discovered3DAsset,
    onProgress?: (percent: number | null) => void,
    targetPedestalSlot?: number,
    focusOnLoad = true
  ): Promise<{ stats: LoadedModelStats; slotIndex: number }> {
    if (this.mode === 'showroom' || targetPedestalSlot !== undefined) {
      const existingSlot = this.findSlotWithAsset(asset);
      if (existingSlot !== -1) {
        throw new Error(`Object already displayed on Plinth #${existingSlot + 1}`);
      }
    }

    let resolvedSlotIdx: number | undefined = targetPedestalSlot;
    if (this.mode === 'showroom' && resolvedSlotIdx === undefined) {
      const emptySlot = this.slots.find((s) => s.modelRoot === null);
      if (emptySlot) {
        resolvedSlotIdx = emptySlot.index;
      } else {
        resolvedSlotIdx = this.nextReplaceSlotIndex;
        this.nextReplaceSlotIndex = (this.nextReplaceSlotIndex + 1) % 4;
      }
    }

    const abortController = new AbortController();
    if (resolvedSlotIdx !== undefined && resolvedSlotIdx >= 0 && resolvedSlotIdx < 4) {
      this.slotAbortControllers[resolvedSlotIdx]?.abort();
      this.slotAbortControllers[resolvedSlotIdx] = abortController;
    } else {
      this.activeAbortController?.abort();
      this.activeAbortController = abortController;
    }

    const { rootObject, animations, stats } = await this.parseAssetGeometry(
      asset,
      abortController.signal,
      onProgress
    );

    if (this.mode === 'showroom' || resolvedSlotIdx !== undefined) {
      const slotIdx = resolvedSlotIdx ?? 0;

      const slot = this.slots[slotIdx];
      const wasReplacingExistingPedestal = slot.modelRoot !== null;
      // Clear Dais clones before disposing a pedestal slot that might be referenced by the Dais
      if (this.daisSlot?.sourceSlotIndices.includes(slotIdx)) {
        this.clearDaisModels();
      }
      this.clearSlot(slot);

      rootObject.traverse((obj) => {
        obj.userData.slotIndex = slotIdx;
        const mesh = obj as THREE.Mesh;
        if (mesh.isMesh && mesh.material) {
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          slot.originalMaterials.set(mesh, mesh.material);
        }
      });

      // Wrap rootObject inside an animation-proof displayGroup so GLTF root keyframe tracks
      // never overwrite our pedestal maquette scaling or y = 0 grounding offset!
      const displayGroup = new THREE.Group();
      displayGroup.userData.slotIndex = slotIdx;
      displayGroup.add(rootObject);

      slot.modelRoot = displayGroup;
      slot.asset = asset;
      slot.stats = stats;
      slot.lodTier = 'close';
      slot.loadedAt = Date.now();
      slot.whisperedForCurrentAsset = false;
      slot.customVisitorComments = [];
      slot.commentPlayIndex = 0;
      slot.modelPivot.rotation.set(0, 0, 0);
      slot.modelPivot.add(displayGroup);

      if (animations.length > 0) {
        slot.mixer = new THREE.AnimationMixer(rootObject);
        slot.mixer.clipAction(animations[0]).play();
        // Evaluate frame 0 pose BEFORE measuring bounds so rigged/animated meshes are grounded accurately
        slot.mixer.update(0);
      }

      const normResult = this.normalizeObjectOnPivot(displayGroup, slot.modelPivot, {
        maxHorizontalDiameter: 1.32,
        maxHeight: 1.36,
      });
      stats.scaleFactor = normResult.scaleFactor;
      stats.scaleLabel = normResult.scaleLabel;

      const shadowSpan = THREE.MathUtils.clamp(
        Math.max(normResult.footprintX, normResult.footprintZ) * 1.05,
        0.55,
        1.22
      );
      slot.contactShadowMesh.scale.set(shadowSpan, shadowSpan, 1);
      slot.contactShadowMesh.visible = true;
      this.applyWireframeToRoot(displayGroup);
      this.updateSlotLecternTexture(slot);
      this.triggerSlotRevealEffect(slot);

      // Automatically rebuild the Central Dais combined sculpture after 2 new sculptures
      // have been loaded into the museum view (while always keeping manual R / T / F working!)
      if (this.mode === 'showroom' && wasReplacingExistingPedestal) {
        this.newShowroomModelsSinceDaisRebuild++;
        if (this.newShowroomModelsSinceDaisRebuild >= 2 && this.getFilledPedestalCount() >= 3) {
          this.rebuildDaisCollision(true);
        }
      }

      if (this.mode === 'showroom' && focusOnLoad) {
        this.focusPedestal(slotIdx, true);
      }

      return { stats, slotIndex: slotIdx };
    }

    // Studio Mode load
    this.clearCurrentModel();

    rootObject.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
    });

    const studioDisplayGroup = new THREE.Group();
    studioDisplayGroup.add(rootObject);

    this.currentModelRoot = studioDisplayGroup;
    this.studioPivot.rotation.set(0, 0, 0);
    this.studioPivot.add(studioDisplayGroup);

    if (animations.length > 0) {
      this.studioMixer = new THREE.AnimationMixer(rootObject);
      const action = this.studioMixer.clipAction(animations[0]);
      action.play();
      this.studioMixer.update(0);
    }

    const normResult = this.normalizeObjectOnPivot(studioDisplayGroup, this.studioPivot, {
      maxHorizontalDiameter: 2.35,
      maxHeight: 2.2,
    });
    stats.scaleFactor = normResult.scaleFactor;
    stats.scaleLabel = normResult.scaleLabel;

    const studioShadowSpan = THREE.MathUtils.clamp(
      Math.max(normResult.footprintX, normResult.footprintZ) * 1.1,
      0.85,
      2.35
    );
    this.studioContactShadow.scale.set(studioShadowSpan, studioShadowSpan, 1);
    this.studioContactShadow.visible = true;

    this.applyWireframeToRoot(studioDisplayGroup);

    this.resetCameraView();
    return { stats, slotIndex: 0 };
  }

  private applyStudioMaterialIfMissing(root: THREE.Object3D): void {
    root.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh) {
        if (!mesh.geometry.attributes.normal) {
          mesh.geometry.computeVertexNormals();
        }
        const hasVertexColors = Boolean(mesh.geometry.attributes.color);
        if (
          !mesh.material ||
          (mesh.material as THREE.Material).type === 'MeshBasicMaterial' ||
          (mesh.material as THREE.Material).type === 'MeshPhongMaterial'
        ) {
          const oldMat = mesh.material as THREE.Material | undefined;
          oldMat?.dispose();
          const fallbackMat = new THREE.MeshStandardMaterial({
            color: hasVertexColors ? 0xffffff : 0xe2e8f0,
            vertexColors: hasVertexColors,
            roughness: 0.42,
            metalness: 0.12,
            side: THREE.DoubleSide,
          });
          if (!hasVertexColors) {
            fallbackMat.userData.isFallbackStudioMaterial = true;
          }
          mesh.material = fallbackMat;
        }
      }
    });
  }

  private computeStatsAndPrepareMeshes(root: THREE.Object3D): LoadedModelStats {
    let vertices = 0;
    let triangles = 0;
    let meshes = 0;

    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh && mesh.geometry) {
        meshes++;
        const pos = mesh.geometry.attributes.position;
        if (pos) {
          vertices += pos.count;
          if (mesh.geometry.index) {
            triangles += Math.floor(mesh.geometry.index.count / 3);
          } else {
            triangles += Math.floor(pos.count / 3);
          }
        }
      }
    });

    return { vertices, triangles, meshes };
  }

  /**
   * Computes an outlier-resistant bounding box in `pivot`'s local space across visible Meshes:
   * - Preserves 100% of low-poly architectural/building walls, roofs, and foundations (`posCount <= 1500`).
   * - Trims the 0.3% outlier tail on dense photogrammetry scan meshes (`posCount > 1500`) so floating dust specks don't lift the model.
   * - Automatically excludes giant skybox/ground-plane environment backdrops if a multi-mesh scene includes one.
   */
  private computeRobustLocalBounds(
    modelRoot: THREE.Object3D,
    pivot: THREE.Group
  ): { min: THREE.Vector3; max: THREE.Vector3; size: THREE.Vector3; center: THREE.Vector3 } | null {
    pivot.updateMatrixWorld(true);
    const invPivotWorld = new THREE.Matrix4().copy(pivot.matrixWorld).invert();
    const meshToPivot = new THREE.Matrix4();
    const v = new THREE.Vector3();

    interface MeshBoxEntry {
      box: THREE.Box3;
      diag: number;
      vertCount: number;
      isFlatPlane: boolean;
    }

    const meshEntries: MeshBoxEntry[] = [];

    modelRoot.traverseVisible((obj) => {
      const mesh = obj as THREE.Mesh & THREE.SkinnedMesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const posAttr = mesh.geometry.attributes.position;
      if (!posAttr || posAttr.count === 0) return;

      const lowerName = `${mesh.name || ''} ${mesh.parent?.name || ''}`.toLowerCase();
      if (
        lowerName.includes('shadow') ||
        lowerName.includes('collider') ||
        lowerName.includes('bounds') ||
        lowerName.includes('bounding') ||
        lowerName.includes('skybox') ||
        lowerName.includes('skydome') ||
        lowerName.includes('backdrop') ||
        lowerName.includes('env_sphere')
      ) {
        return;
      }

      meshToPivot.multiplyMatrices(invPivotWorld, mesh.matrixWorld);

      const mxs: number[] = [];
      const mys: number[] = [];
      const mzs: number[] = [];

      const step = Math.max(1, Math.floor(posAttr.count / 900));
      for (let i = 0; i < posAttr.count; i += step) {
        v.fromBufferAttribute(posAttr, i);
        if (mesh.isSkinnedMesh && typeof mesh.applyBoneTransform === 'function') {
          try {
            mesh.applyBoneTransform(i, v);
          } catch {
            // Fallback to bind pose if skeleton is incomplete
          }
        }
        v.applyMatrix4(meshToPivot);
        if (Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z)) {
          mxs.push(v.x);
          mys.push(v.y);
          mzs.push(v.z);
        }
      }

      if (mys.length === 0) return;

      mxs.sort((a, b) => a - b);
      mys.sort((a, b) => a - b);
      mzs.sort((a, b) => a - b);

      // Only trim 0.3% photogrammetry dust specks on dense scan meshes (>1500 verts);
      // keep exact bounds on low-poly architectural walls/roofs so buildings never clip!
      const isDenseScan = posAttr.count > 1500 && mys.length > 120;
      const lo = isDenseScan ? Math.floor(mys.length * 0.003) : 0;
      const hi = isDenseScan ? Math.min(mys.length - 1, Math.ceil(mys.length * 0.997)) : mys.length - 1;

      const bMin = new THREE.Vector3(mxs[lo], mys[lo], mzs[lo]);
      const bMax = new THREE.Vector3(mxs[hi], mys[hi], mzs[hi]);
      const box = new THREE.Box3(bMin, bMax);
      const sz = new THREE.Vector3();
      box.getSize(sz);
      const horiz = Math.hypot(sz.x, sz.z);
      const diag = sz.length();
      const isFlatPlane = horiz > 0 && sz.y < horiz * 0.015 && posAttr.count <= 64;

      meshEntries.push({
        box,
        diag,
        vertCount: posAttr.count,
        isFlatPlane,
      });
    });

    if (meshEntries.length === 0) return null;

    // If a multi-mesh model includes a giant environment shell / stage floor (>6x larger than all other meshes combined),
    // filter out that backdrop shell so the actual building or object isn't shrunk to a speck
    let activeEntries = meshEntries;
    if (meshEntries.length >= 2) {
      const sortedByDiag = [...meshEntries].sort((a, b) => b.diag - a.diag);
      const largest = sortedByDiag[0];
      const restBox = new THREE.Box3();
      for (let i = 1; i < sortedByDiag.length; i++) {
        restBox.union(sortedByDiag[i].box);
      }
      const restSize = new THREE.Vector3();
      restBox.getSize(restSize);
      const restDiag = restSize.length();

      if (
        restDiag > 0.0001 &&
        ((largest.isFlatPlane && largest.diag > restDiag * 2.5) ||
          (largest.diag > restDiag * 6.5 && largest.vertCount < 500))
      ) {
        activeEntries = sortedByDiag.slice(1);
      }
    }

    const combinedBox = new THREE.Box3();
    for (const entry of activeEntries) {
      combinedBox.union(entry.box);
    }

    const min = combinedBox.min.clone();
    const max = combinedBox.max.clone();
    const size = new THREE.Vector3().subVectors(max, min);
    const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);

    return { min, max, size, center };
  }

  /**
   * Scales any 3D object—from a tiny coin to a massive multi-building architectural block—to fit
   * cleanly inside the cylindrical pedestal maquette envelope (`maxHorizontalDiameter` × `maxHeight`),
   * centers it horizontally, and grounds its base flush at `y = 0`.
   */
  private normalizeObjectOnPivot(
    displayGroup: THREE.Object3D,
    pivot: THREE.Group,
    envelope: { maxHorizontalDiameter: number; maxHeight: number }
  ): { scaleFactor: number; scaleLabel: string; footprintX: number; footprintZ: number } {
    displayGroup.position.set(0, 0, 0);
    displayGroup.scale.set(1, 1, 1);

    const initialBounds = this.computeRobustLocalBounds(displayGroup, pivot);
    if (!initialBounds) {
      return { scaleFactor: 1, scaleLabel: '1:1', footprintX: 0.8, footprintZ: 0.8 };
    }

    // Use horizontal diagonal `Math.hypot(size.x, size.z)` so even rectangular buildings
    // fit 100% inside the circular marble pedestal top without overhanging corners!
    const horizDiameter = Math.hypot(initialBounds.size.x, initialBounds.size.z);
    const height = initialBounds.size.y;

    const scaleHoriz =
      horizDiameter > 0.00001 ? envelope.maxHorizontalDiameter / horizDiameter : Infinity;
    const scaleVert = height > 0.00001 ? envelope.maxHeight / height : Infinity;
    let scaleFactor = Math.min(scaleHoriz, scaleVert);
    if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
      scaleFactor = 1;
    }

    displayGroup.scale.setScalar(scaleFactor);

    const scaledBounds = this.computeRobustLocalBounds(displayGroup, pivot);
    if (!scaledBounds) {
      return { scaleFactor, scaleLabel: '1:1', footprintX: 0.8, footprintZ: 0.8 };
    }

    // Center horizontally on X/Z and place the bottom flush on the pedestal surface (y = 0)
    displayGroup.position.x -= scaledBounds.center.x;
    displayGroup.position.z -= scaledBounds.center.z;
    displayGroup.position.y -= scaledBounds.min.y;
    pivot.updateMatrixWorld(true);

    let scaleLabel = '1:1';
    if (scaleFactor < 0.75) {
      const ratio = Math.max(2, Math.round(1 / scaleFactor));
      scaleLabel = `1:${ratio.toLocaleString()} maquette`;
    } else if (scaleFactor > 1.35) {
      scaleLabel = `${Math.round(scaleFactor * 10) / 10}× scale`;
    }

    return {
      scaleFactor,
      scaleLabel,
      footprintX: scaledBounds.size.x,
      footprintZ: scaledBounds.size.z,
    };
  }

  public flipUpAxis(slotIndex?: number): void {
    if (this.mode === 'showroom') {
      const idx = slotIndex ?? this.focusedSlotIndex;
      const slot = this.slots[idx];
      if (!slot || !slot.modelRoot) return;
      const targetObj = slot.modelRoot.children[0] || slot.modelRoot;
      targetObj.rotateX(-Math.PI / 2);
      const normResult = this.normalizeObjectOnPivot(slot.modelRoot, slot.modelPivot, {
        maxHorizontalDiameter: 1.32,
        maxHeight: 1.36,
      });
      if (slot.stats) {
        slot.stats.scaleFactor = normResult.scaleFactor;
        slot.stats.scaleLabel = normResult.scaleLabel;
      }
      return;
    }

    if (!this.currentModelRoot) return;
    const targetObj = this.currentModelRoot.children[0] || this.currentModelRoot;
    targetObj.rotateX(-Math.PI / 2);
    this.normalizeObjectOnPivot(this.currentModelRoot, this.studioPivot, {
      maxHorizontalDiameter: 2.35,
      maxHeight: 2.2,
    });
    this.resetCameraView();
  }

  public resetCameraView(): void {
    if (this.mode === 'showroom') {
      this.focusPedestal(this.focusedSlotIndex, true);
      return;
    }

    if (!this.currentModelRoot) {
      this.camera.position.set(2.8, 2.1, 3.4);
      this.controls.target.set(0, 0.8, 0);
      this.controls.update();
      return;
    }

    const box = new THREE.Box3().setFromObject(this.studioPivot);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    const radius = Math.max(size.length() * 0.5, 0.5);
    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + radius * 1.45,
      center.y + radius * 0.95,
      center.z + radius * 1.65
    );
    this.controls.update();
  }

  public setAutoRotate(enabled: boolean): void {
    this.autoRotateEnabled = enabled;
    if (this.mode === 'studio') {
      this.controls.autoRotate = enabled;
    }
  }

  public getAutoRotate(): boolean {
    return this.autoRotateEnabled;
  }

  public setWireframe(enabled: boolean): void {
    this.wireframeEnabled = enabled;
    if (this.currentModelRoot) {
      this.applyWireframeToRoot(this.currentModelRoot);
    }
    for (const slot of this.slots) {
      if (slot.modelRoot) {
        this.applyWireframeToRoot(slot.modelRoot);
      }
    }
    if (this.daisSlot?.modelRoot) {
      this.applyWireframeToRoot(this.daisSlot.modelRoot);
    }
  }

  public getWireframe(): boolean {
    return this.wireframeEnabled;
  }

  private applyWireframeToRoot(root: THREE.Object3D): void {
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh && mesh.material) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const m of mats) {
          if (m && m !== this.farProxyMaterial && 'wireframe' in m) {
            (m as THREE.MeshStandardMaterial).wireframe = this.wireframeEnabled;
          }
        }
      }
    });
  }

  public setGridVisible(visible: boolean): void {
    this.studioGridPreferred = visible;
    if (this.mode === 'studio') {
      this.studioFloor.visible = visible;
    }
  }

  public getGridVisible(): boolean {
    return this.studioGridPreferred;
  }

  public setAudioEnabled(enabled: boolean): void {
    this.audio.setEnabled(enabled);
  }

  public getAudioEnabled(): boolean {
    return this.audio.isEnabled();
  }

  public unlockAudio(): void {
    this.audio.unlock();
  }

  public getMemoryStats(): ViewerMemoryStats {
    const mem = this.renderer.info.memory;
    const perfMem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    const jsHeapMB = perfMem
      ? Math.round((perfMem.usedJSHeapSize / (1024 * 1024)) * 10) / 10
      : null;

    return {
      gpuGeometries: mem.geometries,
      gpuTextures: mem.textures,
      jsHeapMB,
    };
  }
}
