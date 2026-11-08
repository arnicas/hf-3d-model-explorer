import { KokoroTTS } from 'kokoro-js';

const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';

// Varied gallery visitor voices across American & British female & male personas.
// Highly expressive voices (af_bella, af_heart, af_sky, bf_emma, bf_isabella, am_adam)
// are prioritized for their natural inflection on questions and conversational contours.
const VISITOR_VOICES = [
  'af_bella',    // expressive, high pitch variety, great question inflection
  'am_adam',     // casual young adult / gamer, natural inflection
  'bf_emma',     // thoughtful British student, clear rising tones
  'af_heart',    // warm, dynamic pitch range
  'af_sky',      // light, airy, expressive intonation
  'bf_isabella', // analytical British art student
  'am_eric',     // energetic kid / gamer
  'af_nicole',   // soft, whispery
  'bm_george',   // mature curatorial guard
  'am_michael',  // relaxed observer
  'bm_lewis',    // quiet British patron
] as const;

let lastVoiceIndex = -1;

function pickRandomVisitorVoice(): string {
  let nextIdx: number;
  do {
    nextIdx = Math.floor(Math.random() * VISITOR_VOICES.length);
  } while (nextIdx === lastVoiceIndex && VISITOR_VOICES.length > 1);
  lastVoiceIndex = nextIdx;
  return VISITOR_VOICES[nextIdx];
}

let ttsInstance: any = null;
let initPromise: Promise<any> | null = null;
let isGenerating = false;

async function ensureKokoroLoaded(): Promise<any> {
  if (ttsInstance) return ttsInstance;
  if (initPromise) return initPromise;

  self.postMessage({ type: 'status', status: 'loading' });

  initPromise = (async () => {
    try {
      const tts = await KokoroTTS.from_pretrained(MODEL_ID, {
        dtype: 'q8',
        device: 'wasm',
      });
      ttsInstance = tts;
      self.postMessage({ type: 'status', status: 'ready' });
      return tts;
    } catch (err) {
      initPromise = null;
      self.postMessage({
        type: 'status',
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
      });
      throw err;
    }
  })();

  return initPromise;
}

self.addEventListener('message', async (event: MessageEvent) => {
  const data = event.data;
  if (!data || typeof data !== 'object') return;

  if (data.type === 'init') {
    try {
      await ensureKokoroLoaded();
    } catch {
      // Handled in ensureKokoroLoaded
    }
    return;
  }

  if (data.type === 'speak' && typeof data.text === 'string') {
    // Drop overlapping requests so visitors never talk over each other
    if (isGenerating) return;
    isGenerating = true;

    try {
      const tts = await ensureKokoroLoaded();
      const voice = data.voice || pickRandomVisitorVoice();
      const rawAudio = await tts.generate(data.text, {
        voice,
        speed: 0.95,
      });

      const pcm =
        rawAudio.audio instanceof Float32Array
          ? rawAudio.audio
          : new Float32Array(rawAudio.audio);
      const samples = new Float32Array(pcm);
      const sampleRate: number = rawAudio.sampling_rate || 24000;

      (self as unknown as Worker).postMessage(
        {
          type: 'audio',
          text: data.text,
          voice,
          samples,
          sampleRate,
        },
        [samples.buffer]
      );
    } catch (err) {
      self.postMessage({
        type: 'error',
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      isGenerating = false;
    }
  }
});
