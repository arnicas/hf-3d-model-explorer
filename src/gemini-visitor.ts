import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

// const CACHE_PREFIX = 'hf3d_gemini_whisper_v7:';

export interface VisitorCommentsResponse {
  comments: string[];
}

export interface SculptureNameResponse {
  name: string;
}

export interface SculptureNameResult {
  name: string;
  fromGemini: boolean;
}

export const FALLBACK_SCULPTURE_NAMES: string[] = [
  // User-curated titles
  'The Contemplation of Breakdown',
  'A Schism in the Void',
  'Repetition on Its Head',
  'A Broken Material',
  "Her Mother's Habit",
  'Infection',
  'Alien Gloss',
  'Rhyme in Time, Upright',
  // Additional obscure & deadpan sculpture titles
  'Unscheduled Monument',
  'Post-Domestic Residue',
  'The Weight of an Excuse',
  'An Agreement Between Strangers',
  'Leftover Gravity',
  'The Polite Intruder',
  'Study for a Forgotten Appliance',
  'Misplaced Anatomy',
  'Three Things Refusing to Part',
  'A Hesitation in the Mold',
  'Domestic Parallax',
  'Everything Must Go Upward',
  'Symptom of a Larger Room',
  'Synthetic Heirloom',
  'The Second Disappointment',
  'Accidental Reliquary',
  'False Equilibrium',
  'Borrowing Shape from Air',
  'Mild Structural Panic',
  'A Theory of Bad Posture',
  'Objects Caught Conspiring',
  'The Persistence of Clutter',
  'Quietly Load-Bearing',
  'Surrogate Ancestor',
  'Uncanny Inventory',
  'Ceremony Without Guests',
  'A Brief Misunderstanding of Mass',
  'Fossil of Next Tuesday',
  'Reluctant Verticality',
  'Some Assembly Regretted',
  'Phantom Counterweight',
  'Anatomy of a Rumor',
  'The Last Known Arrangement',
  'Too Heavy for Its Shadow',
  'Suspiciously Still',
  'An Argument in Three Volumes',
  'Provincial Ectoplasm',
  'Notes on a Collapsing Shelf',
  'The Object Looks Back',
  'Minor Tectonic Indiscretion',
  'Something Kept in a Drawer',
  'Vernacular Anomaly',
  'A Compromise with the Floor',
  'Half-Remembered Machinery',
  'The Wrong Kind of Permanence',
  'Soft Catastrophe in Brass',
  'Calcified Alibi',
  'Apology to the Plinth',
];

const recentFallbackIndices: number[] = [];

export function pickFallbackSculptureName(): string {
  const total = FALLBACK_SCULPTURE_NAMES.length;
  const maxRecent = Math.min(20, Math.floor(total / 2));
  const candidates: number[] = [];
  for (let i = 0; i < total; i++) {
    if (!recentFallbackIndices.includes(i)) {
      candidates.push(i);
    }
  }
  const pool = candidates.length > 0 ? candidates : FALLBACK_SCULPTURE_NAMES.map((_, i) => i);
  const chosenIdx = pool[Math.floor(Math.random() * pool.length)];
  recentFallbackIndices.push(chosenIdx);
  if (recentFallbackIndices.length > maxRecent) {
    recentFallbackIndices.shift();
  }
  return FALLBACK_SCULPTURE_NAMES[chosenIdx];
}

/**
 * Takes 2–4 multi-angle snapshots of the collided 3D sculpture on the Central Dais
 * and asks Gemini (`gemini-3.5-flash-lite`) to create an exhibit name for the sculpture.
 * Falls back to `FALLBACK_SCULPTURE_NAMES` if the API key is missing, rate-limited, or fails.
 */
export async function generateSculptureNameForDais(params: {
  imagesBase64Jpeg: string[];
  assemblyMode?: 'collision' | 'totem';
}): Promise<SculptureNameResult> {
  const apiKey = (import.meta.env.VITE_GEMINI_API_KEY as string | undefined)?.trim();
  if (!apiKey) {
    console.warn('[Gemini Sculpture Naming] No VITE_GEMINI_API_KEY configured. Using fallback title.');
    return { name: pickFallbackSculptureName(), fromGemini: false };
  }

  const validImages = params.imagesBase64Jpeg.filter((img) => img && img.length > 0);
  if (validImages.length === 0) {
    return { name: pickFallbackSculptureName(), fromGemini: false };
  }

  try {
    const ai = new GoogleGenAI({ apiKey });

    const promptText = `These are photos from different angles of a 3D sculpture created by colliding three 3D models together on a museum pedestal. Create a name for the sculpture (2 to 6 words, no quotation marks).`;

    const config = {
      thinkingConfig: {
        thinkingLevel: ThinkingLevel.MINIMAL,
      },
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        required: ['name'],
        properties: {
          name: {
            type: Type.STRING,
          },
        },
      },
    };

    const contents = [
      {
        role: 'user',
        parts: [
          ...validImages.map((base64Data) => ({
            inlineData: {
              mimeType: 'image/jpeg',
              data: base64Data,
            },
          })),
          {
            text: promptText,
          },
        ],
      },
    ];

    const response = await ai.models.generateContentStream({
      model: 'gemini-3.5-flash-lite',
      config,
      contents,
    });

    let fullJson = '';
    for await (const chunk of response) {
      if (chunk.text) {
        fullJson += chunk.text;
      }
    }

    const parsed = JSON.parse(fullJson) as SculptureNameResponse;
    const cleanName = parsed?.name?.replace(/^["'“”]+|["'“”]+$/g, '').trim();
    if (cleanName) {
      console.log(
        `%c[Central Dais] Gemini Sculpture Name (${validImages.length} angles): "${cleanName}"`,
        'color: #c084fc; font-weight: bold; font-size: 13px;'
      );
      return { name: cleanName, fromGemini: true };
    }
  } catch (err) {
    console.warn('[Gemini Sculpture Naming] Failed to generate sculpture name, using fallback:', err);
  }

  const fallback = pickFallbackSculptureName();
  console.log(
    `%c[Central Dais] Fallback Sculpture Name: "${fallback}"`,
    'color: #f59e0b; font-weight: bold; font-size: 13px;'
  );
  return { name: fallback, fromGemini: false };
}

export const VISITOR_PERSONAS = [
  'You are a tired and hungry kid dragged to this museum. What are short, annoyed things you might say? Provide 3.',
  'You are an art student with extreme pretensions examining this 3D piece with dry irony. What are short things you might say? Provide 3.',
  'You are an enthusiastic computer gamer inspecting this 3D model, roasting its polycount or hitbox. What are short things you might say? Provide 3.',
  'You are an exhausted teacher trying to get rowdy students to find deep meaning in this ridiculous object. What are short, educational things you might say? Provide 3.',
  'You are an eccentric collector conspiratorially convinced this plastic toy is an authentic ancient artifact. What are short, impressed things you might say? Provide 3.',
  'You are a completely lost museum patron looking for the cafeteria or the Romantic period. What are short, confused things you might say? Provide 3.',
  'You are an officious museum guard secretly judging the exhibit while telling people not to get too close. What are short things you might say? Provide 3.',
];

/**
 * Downscales an HTMLCanvasElement to a 256x256 JPEG base64 string
 * (~258 tokens per image in Gemini).
 */
export function captureCanvasThumbnailBase64(sourceCanvas: HTMLCanvasElement): string {
  const thumb = document.createElement('canvas');
  thumb.width = 256;
  thumb.height = 256;
  const ctx = thumb.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(sourceCanvas, 0, 0, thumb.width, thumb.height);
  const dataUrl = thumb.toDataURL('image/jpeg', 0.72);
  return dataUrl.replace(/^data:image\/jpeg;base64,/, '');
}

/**
 * Generates custom character-driven visitor remarks using Gemini 3.5 Flash Lite
 * with multimodal canvas thumbnail input, structured JSON output, and localStorage caching.
 *
 * Notice: We deliberately do NOT put raw object names, filenames, or repo identifiers
 * into the prompt since they are often machine slugs or hashes (e.g. "acropora-secale-cb2e1f4d").
 * Instead, Gemini examines the visual 3D snapshot purely on its physical appearance!
 */
export async function generateVisitorWhispersForExhibit(_params: {
  imageBase64Jpeg: string;
  assetId: string;
}): Promise<string[] | null> {
  // Gemini API calls temporarily disabled
  return null;
  /*
  const apiKey = (import.meta.env.VITE_GEMINI_API_KEY as string | undefined)?.trim();
  if (!apiKey) return null;

  const cacheKey = `${CACHE_PREFIX}${params.assetId}`;
  const cached = localStorage.getItem(cacheKey);
  if (cached) {
    try {
      const parsed = JSON.parse(cached) as VisitorCommentsResponse;
      if (Array.isArray(parsed.comments) && parsed.comments.length > 0) {
        return parsed.comments;
      }
    } catch {
      // Ignore cache parse errors
    }
  }

  try {
    const ai = new GoogleGenAI({ apiKey });
    const persona =
      VISITOR_PERSONAS[Math.floor(Math.random() * VISITOR_PERSONAS.length)];

    const promptText = [
      persona,
      `You are quietly murmuring or whispering softly behind someone's shoulder in an echoey museum gallery.`,
      `You are looking at this 3D exhibit resting on a marble pedestal.`,
      `CRITICAL KOKORO TTS INTONATION & QUESTION RULES:`,
      `Your lines are spoken aloud by a neural TTS model (Kokoro). How you write questions directly controls whether the voice actually rises or falls:`,
      `1. FOR RISING QUESTION INTONATION: Do NOT use "What", "Why", or "How" for rising pitch (the TTS drops pitch on Wh-words like a statement). INSTEAD, use auxiliary/inversion questions or conversational tag questions, e.g. "Is that really supposed to be a foot?", "Did someone seriously pay for this?", "Are those supposed to be wings, right?", "Could this actually be ancient?", "Wait, does it have teeth?". These force a dramatic rising pitch contour!`,
      `2. PAIRED OR CONFUSED QUESTIONS: Use a leading pause before the question, e.g. "Hmm... is that a snout?", "Wait — are you seeing what I'm seeing?".`,
      `3. PUNCTUATION & TIMING: Use ellipsis "..." for skeptical deadpan pauses, commas "," for quick breath pauses, and an occasional capitalized word for comedic stress (e.g. "That is DEFINITELY not standard pottery.").`,
      `4. FORMAT: Keep each line under 13 words. No quotation marks, no emojis, no stage directions like [sighs] or (whispering).`,
      `Look closely at the snapshot of the 3D model and write 3 short, punchy, sarcastic or funny single-sentence remarks in this style reacting directly to what you see.`,
    ].join(' ');

    const config = {
      thinkingConfig: {
        thinkingLevel: ThinkingLevel.MINIMAL,
      },
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        required: ['comments'],
        properties: {
          comments: {
            type: Type.ARRAY,
            items: {
              type: Type.STRING,
            },
          },
        },
      },
    };

    const contents = [
      {
        role: 'user',
        parts: [
          ...(params.imageBase64Jpeg
            ? [
                {
                  inlineData: {
                    mimeType: 'image/jpeg',
                    data: params.imageBase64Jpeg,
                  },
                },
              ]
            : []),
          {
            text: promptText,
          },
        ],
      },
    ];

    const response = await ai.models.generateContentStream({
      model: 'gemini-3.5-flash-lite',
      config,
      contents,
    });

    let fullJson = '';
    for await (const chunk of response) {
      if (chunk.text) {
        fullJson += chunk.text;
      }
    }

    const parsed = JSON.parse(fullJson) as VisitorCommentsResponse;
    if (Array.isArray(parsed?.comments) && parsed.comments.length > 0) {
      try {
        localStorage.setItem(cacheKey, JSON.stringify(parsed));
      } catch {
        // Ignore localStorage quota errors
      }
      return parsed.comments;
    }
  } catch (err) {
    console.warn('[Gemini Visitor] Falling back to default remarks:', err);
  }

  return null;
  */
}
