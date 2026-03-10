import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api/semanticscholar': {
        target: 'https://api.semanticscholar.org',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/semanticscholar/, ''),
      },
    },
  },
})
