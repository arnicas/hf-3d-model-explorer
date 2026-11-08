# Client-Side Gemini Vision & Visitor Dialogue Guide (Serverless & Spend-Capped)

This document outlines how to call **`gemini-3.5-flash-lite`** via `@google/genai` directly from this 100% client-side browser app to inspect loaded 3D objects and generate funny, context-aware museum visitor remarks (`{ "comments": ["..."] }`)—**without running a backend server and without risking runaway API bills.**

---

## Option 1: Google AI Studio Free Tier + Domain-Restricted API Key (Hard $0 Cap)

The simplest way to guarantee **$0.00 spend** while deploying as a static site (GitHub Pages, Hugging Face Spaces Static, Netlify, Vercel) is to use an API key on a project with **no Cloud Billing account attached**.

### Step-by-Step Setup
1. **Create an Unbilled Project & Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Create a new API key in a **new Google Cloud project** that stays on the **Free Tier** (do *not* link a credit card / billing account to this project).
   - **Why this is 100% safe:** On the Free Tier, you get generous daily/minute quota on `gemini-3.5-flash-lite` at zero cost. If traffic spikes or someone tries to abuse the key, the API simply returns `429 Too Many Requests` and never bills you.
2. **Lock the API Key to Your Website's Domain (HTTP Referrer Restriction):**
   - Open [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials) and select your AI Studio project.
   - Click on your `Generative Language API Key`.
   - Under **Application restrictions**, select **Websites** and add your allowed origins:
     - `http://localhost:5173/*` (for local Vite dev)
     - `https://your-username.github.io/*` (or your production domain)
   - Under **API restrictions**, select **Restrict key** and check **only** `Generative Language API`.
   - Click **Save**.

---

## Option 2: Firebase AI Logic (`firebase/ai`) + App Check (Serverless Production SDK)

If you want to prevent bots/scrapers from using your quota even when spoofing `Referer` headers, Google's official client-side solution is **Firebase AI Logic** paired with **Firebase App Check**.

### How It Works
- You do **not** embed a raw Gemini API key in your frontend code.
- Instead, the Firebase Web SDK (`firebase/ai`) uses **reCAPTCHA Enterprise (App Check)** to attest that requests originate from a genuine browser running your web app before forwarding the multimodal prompt to the **Gemini Developer API**.

### Quick CLI Setup
```bash
npm install firebase@latest
npx -y firebase-tools@latest init ailogic
```
- In the Firebase Console, go to **Build → AI Logic** and enable **App Check** with **reCAPTCHA Enterprise**.
- For local development on `localhost:5173`, set `self.FIREBASE_APPCHECK_DEBUG_TOKEN = true;` before initializing App Check and register the console-printed debug UUID in **Firebase Console → App Check → Apps → Manage debug tokens**.

---

## Option 3: Hard API Quota Caps in Google Cloud (If Billing Is Enabled)

> **Important:** Google Cloud **Billing Budgets & Alerts** only send email notifications—they do **not** automatically disable the API when your budget is reached.

To enforce a **hard daily spend ceiling** on a billed project without a backend server, cap the API quota directly:

1. Open [Google Cloud Console → IAM & Admin → Quotas & System Limits](https://console.cloud.google.com/iam-admin/quotas).
2. In the filter bar, search for:
   - **Service:** `Generative Language API` (or `Vertex AI API`)
   - **Limit Name:** `GenerateContent requests per day` (and `GenerateContent requests per minute per user`)
3. Select the quota, click **Edit Quotas**, and lower the daily limit to a fixed ceiling—for example, **`500` requests/day**.
4. **Cost Math for `gemini-3.5-flash-lite`:**
   - A `256×256` thumbnail image (`~258` tokens) + short prompt (`~80` tokens) + 3–5 generated remarks (`~80` output tokens) with minimal thinking costs fractions of a cent per 100 calls.
   - At a hard cap of `500` calls/day, your maximum possible monthly spend is **under $1/month**.
   - Once the daily quota of `500` is reached, Google Cloud immediately rejects further requests with `HTTP 429` until midnight Pacific Time.

---

## Architecture for This App: Connecting `gemini-3.5-flash-lite` to Kokoro-82M Whispers

To keep token usage tiny and latency unnoticeable, **do not call Gemini on every frame or every footstep**. Instead:

1. **Snapshot Once on Plinth Load:**
   - After `viewer.loadAsset()` places a 3D model onto a plinth and renders a frame, capture a small `256×256` JPEG data URL from the WebGL canvas (`canvas.toDataURL('image/jpeg', 0.7)`).
2. **Cache by `asset.id` in `localStorage`:**
   - Before calling Gemini, check `localStorage.getItem('hf3d_gemini_whisper_v1:' + asset.id)`. If cached `{ comments: [...] }` exist, reuse them at **0 API cost**.
3. **Character Personas & Dynamic Prompt Mixing:**
   - To make the museum feel alive and hilarious, mix up prompts using rotating **character persona prefixes**.
   - **Direct Art/Object Observers:**
     - *"You are a tired and hungry kid who sees this in the museum. What are 3 things you might say?"*
     - *"You are an art student with pretensions examining this object."*
     - *"You are an enthusiastic lover of computer games inspecting this 3D model."*
     - *"You are a teacher trying to get your rowdy students to enjoy what they see when you find this."*
   - **Ambient / Wandering Gallery Characters (Great for empty spots or background whispers):**
     - *"You are a lost museum patron trying to find the Romantic period in the museum."*
     - *"You are a museum guard trying to keep people from getting too close to the plinth."*
4. **Request Structured `{ comments: string[] }` Output:**
   - Send the `256×256` JPEG + character persona prompt to `gemini-3.5-flash-lite`.
   - Ask for `3–5` brief single-sentence remarks (under 14 words each) so a single API call supplies multiple lines in that character's voice.
5. **Feed Directly into `SpatialAudioManager.playOverShoulderWhisper()`:**
   - Store the returned `comments` array on the `InternalPedestalSlot` (`slot.customVisitorComments: string[]`).
   - When `updateApproachWhisper()` in [`src/viewer.ts`](./src/viewer.ts) triggers after the `2.2s` approach dwell, pass one of the Gemini-generated lines into `this.audio.playOverShoulderWhisper(customLine, ..., this.camera)`.
   - **Kokoro-82M** (`src/kokoro.worker.ts`) will synthesize the custom Gemini remark in `af_nicole`'s (or `bf_emma` / `af_heart`) hushed voice and play it over your left or right shoulder via the 3D HRTF `shoulderPanner`!
   - If the user is offline, hasn't configured an API key, or hits `429` quota, `playOverShoulderWhisper()` automatically falls back to `GEMMA_VISITOR_COMMENTS` in [`src/audio.ts`](./src/audio.ts).

---

## Drop-In Implementation (`src/gemini-visitor.ts`)

Install the official SDK:
```bash
bun add @google/genai
```

Add your domain-restricted key to `.env.local`:
```env
VITE_GEMINI_API_KEY=your_api_key_here
```

### 1. Your AI Studio Test Snippet Turned Into Multimodal (Vision Input + Structured JSON Output)

In your copied snippet, `parts` only had `{ text: 'INSERT_INPUT_HERE' }` because no image was attached when exporting from AI Studio.

**Good news:** **`gemini-3.5-flash-lite` is natively multimodal on input** (it accepts **Text, Image, Video, Audio, and PDF** inputs and returns **Text/JSON** output). To make your snippet see the 3D object, you simply add an `inlineData` part (`mimeType: 'image/jpeg'`, `data: base64String`) alongside `{ text }` inside `contents[0].parts`:

> **Note on Audio Output:** `gemini-3.5-flash-lite` only outputs **Text** (which is ideal here since **Kokoro-82M** synthesizes the whisper locally in-browser for free). If you ever wanted Gemini itself to return synthesized audio bytes instead of using Kokoro, you would use `gemini-3.8-flash-lite-tts` or `gemini-3.8-flash-tts`.

```ts
// src/gemini-visitor.ts
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const CACHE_PREFIX = 'hf3d_gemini_whisper_v1:';

export interface VisitorCommentsResponse {
  comments: string[];
}

export const VISITOR_PERSONAS = [
  'You are a tired and hungry kid who sees this in the museum. What are short, funny things you might say? Provide 3.',
  'You are an art student with pretensions examining this 3D piece. What are short, funny things you might say? Provide 3.',
  'You are an enthusiastic lover of computer games inspecting this 3D model. What are short, funny things you might say? Provide 3.',
  'You are a teacher trying to get your rowdy students to appreciate what they see when you find this. What are short, funny things you might say? Provide 3.',
  'You are an eccentric collector convinced this is an authentic ancient artifact. What are short, funny things you might say? Provide 3.',
  'You are a lost museum patron trying to find the Romantic period in the museum. What are short, funny things you might say? Provide 3.',
  'You are a museum guard keeping people from touching or leaning over the pedestal. What are short, funny things you might say? Provide 3.',
];

/**
 * Downscales the current Three.js WebGL canvas to a compact 256×256 JPEG base64 string
 * so multimodal token usage stays minimal (~258 tokens per image).
 */
export function captureCanvasThumbnailBase64(sourceCanvas: HTMLCanvasElement): string {
  const thumb = document.createElement('canvas');
  thumb.width = 256;
  thumb.height = 256;
  const ctx = thumb.getContext('2d')!;
  ctx.drawImage(sourceCanvas, 0, 0, thumb.width, thumb.height);
  const dataUrl = thumb.toDataURL('image/jpeg', 0.72);
  return dataUrl.replace(/^data:image\/jpeg;base64,/, '');
}

/**
 * Calls `gemini-3.5-flash-lite` with `ThinkingLevel.MINIMAL` and a structured
 * `{ comments: string[] }` JSON schema to generate funny/observant visitor remarks
 * about the 3D exhibit currently displayed on a pedestal.
 */
export async function generateVisitorWhispersForExhibit(params: {
  imageBase64Jpeg: string;
  assetId: string;
  title: string;
  repoId: string;
  scaleLabel?: string;
}): Promise<string[] | null> {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
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
      // Ignore corrupt cache entry
    }
  }

  try {
    const ai = new GoogleGenAI({ apiKey });

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

    const persona =
      VISITOR_PERSONAS[Math.floor(Math.random() * VISITOR_PERSONAS.length)];

    const promptText = [
      persona,
      `You are whispering softly behind someone's shoulder in a museum gallery.`,
      `You are looking at a 3D exhibit on a marble pedestal titled "${params.title}" (from dataset "${params.repoId}", scale: ${params.scaleLabel || '1:1'}).`,
      `Look at the snapshot of the 3D object and write 4 short, in-character, single-sentence remarks (under 14 words each) reacting specifically to what you see.`,
    ].join(' ');

    const contents = [
      {
        role: 'user',
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: params.imageBase64Jpeg,
            },
          },
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
        // Ignore storage quota errors
      }
      return parsed.comments;
    }
  } catch (err) {
    // Gracefully fall back to curated GEMMA_VISITOR_COMMENTS on 429 quota limits or network errors
    console.warn('[Gemini Visitor] Falling back to built-in comments:', err);
  }

  return null;
}
```

### 2. Alternative Using the Modern `ai.interactions.create` API (`@google/genai`)

```ts
import { GoogleGenAI } from '@google/genai';

export async function generateVisitorWhispersViaInteractions(params: {
  imageBase64Jpeg: string;
  title: string;
  repoId: string;
}): Promise<string[] | null> {
  const ai = new GoogleGenAI({
    apiKey: import.meta.env.VITE_GEMINI_API_KEY,
  });

  const interaction = await ai.interactions.create({
    model: 'gemini-3.5-flash-lite',
    store: false,
    input: [
      {
        type: 'image',
        data: params.imageBase64Jpeg,
        mime_type: 'image/jpeg',
      },
      {
        type: 'text',
        text: `Write 4 short, witty, hushed museum visitor remarks (under 14 words each) reacting to this 3D exhibit ("${params.title}" from ${params.repoId}).`,
      },
    ],
    response_format: {
      type: 'object',
      required: ['comments'],
      properties: {
        comments: {
          type: 'array',
          items: { type: 'string' },
        },
      },
    },
  });

  if (!interaction.output_text) return null;
  const parsed = JSON.parse(interaction.output_text) as { comments: string[] };
  return parsed.comments;
}
```
