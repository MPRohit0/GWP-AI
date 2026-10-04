import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // IMPORTANT:
  // For https://USERNAME.github.io/REPO-NAME/
  // change this to: base: "/REPO-NAME/"
  // For https://USERNAME.github.io/ use: base: "/"
  base: "/greenbuild-ai/"
});
