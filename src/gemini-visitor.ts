// import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

// const CACHE_PREFIX = 'hf3d_gemini_whisper_v7:';

export interface VisitorCommentsResponse {
  comments: string[];
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
