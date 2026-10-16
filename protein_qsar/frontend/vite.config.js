import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/targets': 'http://127.0.0.1:8000',
      '/research-targets': 'http://127.0.0.1:8000',
      '/resolve-target': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/predict-batch': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/curate-dataset': 'http://127.0.0.1:8000',
    },
  },
})
