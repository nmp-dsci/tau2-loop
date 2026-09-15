import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: Vite serves the SPA and proxies the API to FastAPI. In the image FastAPI
// serves the built SPA from the same origin, so the proxy does not exist there.
const API = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8081';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': { target: API, changeOrigin: false },
      '/healthz': { target: API, changeOrigin: false },
    },
  },
});
