import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  build: {
    // viteSingleFile inlines all CSS/JS into a single index.html
    // report.py reads this and injects __FORGE_DATA__ before writing the final dashboard
    cssCodeSplit: false,
  },
});
