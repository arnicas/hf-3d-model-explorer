import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const geminiKey = env.VITE_GEMINI_API_KEY || env.GEMINI_API_KEY || '';

  return {
    // Relative so the build works from a subfolder (ghostweather.com/apps/hf-3d-roulette/)
    base: './',
    define: {
      'import.meta.env.VITE_GEMINI_API_KEY': JSON.stringify(geminiKey),
    },
    optimizeDeps: {
      exclude: ['kokoro-js', '@huggingface/transformers', 'onnxruntime-web'],
    },
    server: {
      proxy: {
        '/hf-dataset-assets': {
          target: 'https://datasets-server.huggingface.co',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/hf-dataset-assets/, '/assets'),
        },
      },
    },
  };
});
