import * as THREE from 'three';

export const GEMMA_VISITOR_COMMENTS = {
  category_1_wonder_awe: [
    'Wow, the detail on this texture is incredible.',
    'This is so much more immersive than just looking at a 2D image.',
    'The scale of this model is astonishing.',
    "It's amazing how much information is packed into this single file.",
    "This is like having a physical object in my hands, but it's digital.",
  ],
  category_2_technical_and_analytical: [
    'What modeling software did they use for this?',
    "I'm curious about the topology. How efficient is the mesh for rendering?",
    'The lighting setup in this rendering really highlights the material properties.',
    'This demonstrates a huge leap in digital prototyping.',
    'How did they handle the UV unwrapping for this surface?',
  ],
  category_3_reflective_and_philosophical: [
    'This changes how we think about physical presence. It blurs the line between digital and real.',
    'It’s incredible how much iteration is possible in a digital space.',
    'What are the ethical implications of creating hyper-realistic digital replicas?',
  ],
  category_4_curious_and_question_based: [
    'Can someone show me how this model was originally scanned or built?',
    'What kind of materials are represented here? Is it photorealistic rendering or a stylized look?',
    'How much computational power was needed to generate this level of detail?',
    'What was the workflow from concept to final render?',
  ],
  category_5_casual_and_conversational: [
    'That looks amazing! Where did you get that model?',
    "I can't believe this was all done on a computer.",
    'It looks so sharp!',
    'It’s like looking at a digital sculpture.',
  ],
} as const;

export interface VisitorCommentContext {
  repoId?: string;
  triangles?: number;
  scaleFactor?: number;
  wireframe?: boolean;
}

/**
 * 100% Procedural Web Audio 3D Spatial Sound Engine for HF 3D Roulette.
 * - Uses HRTF PannerNodes anchored at the 4 museum pedestals so plinth reveals pan
 *   and attenuate naturally based on where you stand and look in the 20.8m gallery.
 * - Synthesizes surface-aware footsteps (Polished Oak Parquet vs. Woven Promenade Rug)
 *   with alternating left/right stereo placement as you walk around the room.
 * - Synthesizes live Over-the-Shoulder Visitor Whispers upon approaching a plinth,
 *   combining hushed Web Speech synthesis with a 3D HRTF syllable-synced aspirated
 *   formant whisper stream positioned right behind your left or right shoulder.
 */
export class SpatialAudioManager {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private slotPanners: PannerNode[] = [];
  private shoulderPanner: PannerNode | null = null;
  private pedestalPositions: THREE.Vector3[];
  private enabled = true;
  private leftFootNext = true;
  private shoulderSide: -1 | 1 = -1;
  private recentComments: string[] = [];
  // Kokoro TTS is temporarily disabled pending remote model integration or removal
  // private kokoroWorker: Worker | null = null;
  // private pendingWhisperOrigin?: { isShoulder: boolean; slotIndex?: number };
  private activeWhisperSource: AudioBufferSourceNode | null = null;

  // Distinct warm Major-7th pentatonic roots for Pedestals #1 -> #4 (C4, E4, G4, B4)
  // so you can tell which bay updated both by 3D direction and by harmonic pitch.
  private static readonly SLOT_ROOT_FREQS = [261.63, 329.63, 392.0, 493.88];

  private readonly forwardVec = new THREE.Vector3();
  private readonly upVec = new THREE.Vector3();
  private readonly rightVec = new THREE.Vector3();

  constructor(pedestalPositions: THREE.Vector3[]) {
    this.pedestalPositions = pedestalPositions.map((p) => p.clone());
  }



  /**
   * Lazily spawns the background Kokoro-82M neural TTS Web Worker when entering Showroom Mode.
   * (TEMPORARILY DISABLED: Kokoro TTS commented out)
   */
  public warmUpKokoroWorker(): void {
    // Kokoro TTS disabled pending remote model integration or removal
  }

  /*
  private playKokoroSamples(samples: Float32Array, sampleRate: number): void {
    const ctx = this.ensureContext();
    if (!ctx || !this.enabled || samples.length === 0) return;

    this.cancelActiveWhisper();

    const isShoulder = this.pendingWhisperOrigin?.isShoulder ?? true;
    const slotIdx = this.pendingWhisperOrigin?.slotIndex;
    let targetPanner: PannerNode | null = null;

    if (isShoulder || slotIdx === undefined) {
      targetPanner = this.shoulderPanner;
      this.shoulderSide = this.shoulderSide === -1 ? 1 : -1;
    } else {
      targetPanner = this.slotPanners[slotIdx] || this.shoulderPanner;
    }

    if (!targetPanner) return;

    const audioBuffer = ctx.createBuffer(1, samples.length, sampleRate);
    audioBuffer.getChannelData(0).set(samples);

    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;

    // Hushed gallery voice shaping: trim low boom (< 145 Hz) and soften harsh top end (> 3.8 kHz)
    const highpass = ctx.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = 145;

    const lowpass = ctx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = isShoulder ? 3800 : 2800; // Distant whispers sound more atmospheric

    const gain = ctx.createGain();
    // Intimate shoulder whisper is clear (0.52); distant gallery whisper is subtly quieter (0.35)
    gain.gain.value = isShoulder ? 0.52 : 0.35;

    src.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(gain);
    gain.connect(targetPanner);

    this.activeWhisperSource = src;
    src.onended = () => {
      if (this.activeWhisperSource === src) {
        this.activeWhisperSource = null;
      }
      src.disconnect();
      highpass.disconnect();
      lowpass.disconnect();
      gain.disconnect();
    };

    src.start();
  }
  */

  public isEnabled(): boolean {
    return this.enabled;
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) {
      this.ensureContext();
    } else {
      this.cancelActiveWhisper();
    }
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(enabled ? 0.85 : 0, this.ctx.currentTime, 0.04);
    }
  }

  /**
   * Unlocks / resumes the Web Audio context on any user click or keydown.
   */
  public unlock(): void {
    if (!this.enabled) return;
    this.ensureContext();
  }

  private ensureContext(): AudioContext | null {
    if (!this.enabled) return null;

    if (!this.ctx) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return null;

      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.85;
      this.masterGain.connect(this.ctx.destination);

      // Pre-allocate a single reusable 1.8s pink-noise buffer (0 runtime allocations)
      const sampleRate = this.ctx.sampleRate;
      const length = Math.floor(sampleRate * 1.8);
      this.noiseBuffer = this.ctx.createBuffer(1, length, sampleRate);
      const data = this.noiseBuffer.getChannelData(0);
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      let b3 = 0;
      let b4 = 0;
      let b5 = 0;
      let b6 = 0;
      for (let i = 0; i < length; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        b3 = 0.86650 * b3 + white * 0.3104856;
        b4 = 0.55000 * b4 + white * 0.5329522;
        b5 = -0.7616 * b5 - white * 0.0168980;
        data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.045;
        b6 = white * 0.115926;
      }

      // Create 4 persistent HRTF 3D PannerNodes anchored at each pedestal
      this.slotPanners = this.pedestalPositions.map((pos) => {
        const panner = this.ctx!.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 2.4;
        panner.maxDistance = 24;
        panner.rolloffFactor = 1.35;
        panner.coneInnerAngle = 360;
        panner.coneOuterAngle = 360;
        this.setNodePosition(panner, pos.x, 1.65, pos.z);
        panner.connect(this.masterGain!);
        return panner;
      });

      // Create a dedicated close-field HRTF PannerNode positioned right behind the visitor's shoulder
      this.shoulderPanner = this.ctx.createPanner();
      this.shoulderPanner.panningModel = 'HRTF';
      this.shoulderPanner.distanceModel = 'inverse';
      this.shoulderPanner.refDistance = 0.35;
      this.shoulderPanner.maxDistance = 4.0;
      this.shoulderPanner.rolloffFactor = 1.6;
      this.shoulderPanner.connect(this.masterGain);
    }

    if (this.ctx.state === 'suspended') {
      void this.ctx.resume().catch(() => {
        // Ignore if browser still requires a direct user interaction
      });
    }

    return this.ctx;
  }

  private setNodePosition(panner: PannerNode, x: number, y: number, z: number): void {
    if (panner.positionX) {
      panner.positionX.value = x;
      panner.positionY.value = y;
      panner.positionZ.value = z;
    } else {
      panner.setPosition(x, y, z);
    }
  }

  /**
   * Syncs the 3D Web Audio listener with the active Three.js camera position and orientation,
   * and keeps the over-the-shoulder HRTF panner tucked right behind the visitor's left or right shoulder.
   */
  public updateListener(camera: THREE.Camera): void {
    if (!this.ctx || !this.enabled || this.ctx.state !== 'running') return;

    const listener = this.ctx.listener;
    const pos = camera.position;
    camera.getWorldDirection(this.forwardVec);
    this.upVec.copy(camera.up).applyQuaternion(camera.quaternion).normalize();
    this.rightVec.crossVectors(this.forwardVec, this.upVec).normalize();

    if (listener.positionX) {
      listener.positionX.value = pos.x;
      listener.positionY.value = pos.y;
      listener.positionZ.value = pos.z;
      listener.forwardX.value = this.forwardVec.x;
      listener.forwardY.value = this.forwardVec.y;
      listener.forwardZ.value = this.forwardVec.z;
      listener.upX.value = this.upVec.x;
      listener.upY.value = this.upVec.y;
      listener.upZ.value = this.upVec.z;
    } else {
      listener.setPosition(pos.x, pos.y, pos.z);
      listener.setOrientation(
        this.forwardVec.x,
        this.forwardVec.y,
        this.forwardVec.z,
        this.upVec.x,
        this.upVec.y,
        this.upVec.z
      );
    }

    // Position the whisper panner 0.44m behind and 0.38m to the left or right shoulder
    if (this.shoulderPanner) {
      const sx =
        pos.x - this.forwardVec.x * 0.44 + this.rightVec.x * (this.shoulderSide * 0.38);
      const sy = pos.y - 0.06;
      const sz =
        pos.z - this.forwardVec.z * 0.44 + this.rightVec.z * (this.shoulderSide * 0.38);
      this.setNodePosition(this.shoulderPanner, sx, sy, sz);
    }
  }

  /**
   * Picks a context-appropriate visitor remark from the 5 Gemma categories based on the
   * 3D model on the approached pedestal, avoiding recent repeats.
   */
  public pickVisitorComment(context?: VisitorCommentContext): string {
    const candidatePool: string[] = [];

    if (context?.wireframe) {
      candidatePool.push(...GEMMA_VISITOR_COMMENTS.category_2_technical_and_analytical);
    } else if (context?.scaleFactor !== undefined && context.scaleFactor < 0.25) {
      candidatePool.push(
        'The scale of this model is astonishing.',
        ...GEMMA_VISITOR_COMMENTS.category_1_wonder_awe,
        ...GEMMA_VISITOR_COMMENTS.category_3_reflective_and_philosophical
      );
    } else if (context?.triangles !== undefined && context.triangles > 50000) {
      candidatePool.push(
        "It's amazing how much information is packed into this single file.",
        'How much computational power was needed to generate this level of detail?',
        ...GEMMA_VISITOR_COMMENTS.category_1_wonder_awe,
        ...GEMMA_VISITOR_COMMENTS.category_2_technical_and_analytical
      );
    } else if (context?.repoId?.toLowerCase().includes('smithsonian')) {
      candidatePool.push(
        'Can someone show me how this model was originally scanned or built?',
        'This changes how we think about physical presence. It blurs the line between digital and real.',
        'What are the ethical implications of creating hyper-realistic digital replicas?',
        ...GEMMA_VISITOR_COMMENTS.category_1_wonder_awe
      );
    }

    // Always mix in the full set so every visit feels varied
    const allLines = [
      ...GEMMA_VISITOR_COMMENTS.category_1_wonder_awe,
      ...GEMMA_VISITOR_COMMENTS.category_2_technical_and_analytical,
      ...GEMMA_VISITOR_COMMENTS.category_3_reflective_and_philosophical,
      ...GEMMA_VISITOR_COMMENTS.category_4_curious_and_question_based,
      ...GEMMA_VISITOR_COMMENTS.category_5_casual_and_conversational,
    ];
    if (candidatePool.length === 0) {
      candidatePool.push(...allLines);
    }

    const fresh = candidatePool.filter((line) => !this.recentComments.includes(line));
    const poolToUse = fresh.length > 0 ? fresh : allLines;
    const chosen = poolToUse[Math.floor(Math.random() * poolToUse.length)];

    this.recentComments.push(chosen);
    if (this.recentComments.length > 8) {
      this.recentComments.shift();
    }
    return chosen;
  }

  public cancelActiveWhisper(): void {
    if (this.activeWhisperSource) {
      try {
        this.activeWhisperSource.stop();
        this.activeWhisperSource.disconnect();
      } catch {
        // Ignore if already stopped
      }
      this.activeWhisperSource = null;
    }
  }

  /**
   * Sends a context-matched visitor remark to the background Kokoro-82M neural TTS worker,
   * which synthesizes natural hushed speech (`af_nicole` / `bf_emma` / `af_heart`) and plays it
   * through the 3D HRTF `shoulderPanner` positioned behind the visitor's left or right shoulder.
   */
  public playOverShoulderWhisper(
    _customText?: string,
    _context?: VisitorCommentContext,
    _camera?: THREE.Camera,
    _spatialOrigin?: { isShoulder: boolean; slotIndex?: number }
  ): string | null {
    // Kokoro TTS is currently commented out / disabled
    return null;
    /*
    if (!this.enabled) return null;
    this.ensureContext();
    this.warmUpKokoroWorker();
    if (!this.kokoroWorker) return null;

    if (camera) {
      this.updateListener(camera);
    }

    this.pendingWhisperOrigin = spatialOrigin || { isShoulder: true };
    const text = customText || this.pickVisitorComment(context);
    this.kokoroWorker.postMessage({ type: 'speak', text });
    return text;
    */
  }

  /**
   * Plays a 3-layer directional Plinth Reveal sound from the specified pedestal's 3D position:
   * 1) Warm hydraulic/stone plinth elevator rise
   * 2) Airy velvet smoke puff whoosh
   * 3) Gentle harmonic brass/glass bell chime tuned to that pedestal's bay index (#1-#4)
   */
  public playPlinthReveal(slotIndex: number): void {
    const ctx = this.ensureContext();
    if (!ctx || ctx.state !== 'running' || !this.noiseBuffer) return;

    const panner = this.slotPanners[slotIndex];
    if (!panner) return;

    const now = ctx.currentTime;

    // 1. Stone / Brass Elevator Rise Hum (92 Hz -> 148 Hz over 1.35s)
    const liftOsc = ctx.createOscillator();
    const liftFilter = ctx.createBiquadFilter();
    const liftGain = ctx.createGain();

    liftOsc.type = 'triangle';
    liftOsc.frequency.setValueAtTime(92, now);
    liftOsc.frequency.exponentialRampToValueAtTime(148, now + 1.25);

    liftFilter.type = 'lowpass';
    liftFilter.frequency.setValueAtTime(240, now);

    liftGain.gain.setValueAtTime(0.0001, now);
    liftGain.gain.linearRampToValueAtTime(0.16, now + 0.18);
    liftGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.45);

    liftOsc.connect(liftFilter);
    liftFilter.connect(liftGain);
    liftGain.connect(panner);
    liftOsc.start(now);
    liftOsc.stop(now + 1.48);

    // 2. Velvet Smoke Puff Whoosh (Bandpass-filtered pink noise burst)
    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = this.noiseBuffer;

    const smokeFilter = ctx.createBiquadFilter();
    smokeFilter.type = 'bandpass';
    smokeFilter.Q.setValueAtTime(1.7, now);
    smokeFilter.frequency.setValueAtTime(260, now);
    smokeFilter.frequency.exponentialRampToValueAtTime(920, now + 0.32);
    smokeFilter.frequency.exponentialRampToValueAtTime(210, now + 1.25);

    const smokeGain = ctx.createGain();
    smokeGain.gain.setValueAtTime(0.0001, now);
    smokeGain.gain.linearRampToValueAtTime(0.28, now + 0.14);
    smokeGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.28);

    noiseSrc.connect(smokeFilter);
    smokeFilter.connect(smokeGain);
    smokeGain.connect(panner);
    noiseSrc.start(now);
    noiseSrc.stop(now + 1.3);

    // 3. Curatorial Harmonic Chime (Root + Fifth + Shimmer Octave)
    const rootFreq =
      SpatialAudioManager.SLOT_ROOT_FREQS[slotIndex % SpatialAudioManager.SLOT_ROOT_FREQS.length];
    const partials: Array<[number, number, number]> = [
      [rootFreq, 0.14, 1.55],
      [rootFreq * 1.498, 0.075, 1.35], // Warm fifth
      [rootFreq * 2.0, 0.045, 1.15], // Golden dust shimmer octave
    ];

    partials.forEach(([freq, peakGain, duration], idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const onset = now + 0.06 + idx * 0.045;

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, onset);

      gain.gain.setValueAtTime(0.0001, onset);
      gain.gain.linearRampToValueAtTime(peakGain, onset + 0.035);
      gain.gain.exponentialRampToValueAtTime(0.0001, onset + duration);

      osc.connect(gain);
      gain.connect(panner);
      osc.start(onset);
      osc.stop(onset + duration + 0.02);
    });
  }

  /**
   * Plays a realistic, surface-aware footstep sound while walking in the showroom:
   * - Crisp, warm wooden heel-toe tap on the Polished Oak Parquet floor
   * - Softer, muted fabric step when walking across the 14.6m × 3.8m Woven Promenade Rug
   */
  public playFootstep(camX: number, camZ: number): void {
    const ctx = this.ensureContext();
    if (!ctx || ctx.state !== 'running' || !this.noiseBuffer) return;

    // Check if the visitor is currently stepping on the central woven promenade rug
    // (centered at x: 0, z: 2.4 with width 14.6m -> [-7.3, 7.3] and depth 3.8m -> [0.5, 4.3])
    const onRug = Math.abs(camX) <= 7.3 && camZ >= 0.5 && camZ <= 4.3;

    const now = ctx.currentTime;
    const isLeft = this.leftFootNext;
    this.leftFootNext = !this.leftFootNext;

    // Subtle human pitch/timing variation per step
    const jitter = 0.94 + Math.random() * 0.12;

    const stereoPanner = ctx.createStereoPanner();
    stereoPanner.pan.setValueAtTime(isLeft ? -0.16 : 0.16, now);
    stereoPanner.connect(this.masterGain!);

    // 1. Low-end heel thump
    const thumpOsc = ctx.createOscillator();
    const thumpGain = ctx.createGain();
    thumpOsc.type = 'sine';

    if (onRug) {
      // Soft muffled carpet step
      thumpOsc.frequency.setValueAtTime(98 * jitter, now);
      thumpOsc.frequency.exponentialRampToValueAtTime(42 * jitter, now + 0.075);
      thumpGain.gain.setValueAtTime(0.0001, now);
      thumpGain.gain.linearRampToValueAtTime(0.11, now + 0.008);
      thumpGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);
    } else {
      // Warm oak parquet heel resonance
      thumpOsc.frequency.setValueAtTime(142 * jitter, now);
      thumpOsc.frequency.exponentialRampToValueAtTime(54 * jitter, now + 0.065);
      thumpGain.gain.setValueAtTime(0.0001, now);
      thumpGain.gain.linearRampToValueAtTime(0.15, now + 0.005);
      thumpGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.07);
    }

    thumpOsc.connect(thumpGain);
    thumpGain.connect(stereoPanner);
    thumpOsc.start(now);
    thumpOsc.stop(now + 0.085);

    // 2. Surface sole contact transient (filtered noise burst)
    const soleSrc = ctx.createBufferSource();
    soleSrc.buffer = this.noiseBuffer;

    const soleFilter = ctx.createBiquadFilter();
    const soleGain = ctx.createGain();

    if (onRug) {
      soleFilter.type = 'lowpass';
      soleFilter.frequency.setValueAtTime(420 * jitter, now);
      soleGain.gain.setValueAtTime(0.0001, now);
      soleGain.gain.linearRampToValueAtTime(0.07, now + 0.01);
      soleGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.065);
    } else {
      soleFilter.type = 'bandpass';
      soleFilter.Q.setValueAtTime(2.6, now);
      soleFilter.frequency.setValueAtTime(1050 * jitter, now);
      soleGain.gain.setValueAtTime(0.0001, now);
      soleGain.gain.linearRampToValueAtTime(0.13, now + 0.004);
      soleGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.048);
    }

    // Pick a random slice of the reusable noise buffer so every step has unique texture
    const offset = Math.random() * 1.2;
    soleSrc.connect(soleFilter);
    soleFilter.connect(soleGain);
    soleGain.connect(stereoPanner);
    soleSrc.start(now, offset, 0.09);
  }
}
