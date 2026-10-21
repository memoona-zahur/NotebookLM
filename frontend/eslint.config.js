// Lint, not format. The rules are the ones that find bugs in a React app that
// has no TypeScript to catch them: undefined and unused names, and the two
// hooks rules (a hook called conditionally, or an effect missing a dependency,
// are the classic silent stale-closure bugs).
import js from "@eslint/js";
import globals from "globals";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";

export default [
  { ignores: ["dist/**", "node_modules/**", "../static/**"] },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}", "*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { react, "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Without these two, a component used only as <Thing /> is reported as an
      // unused import: core ESLint does not know JSX is a use.
      "react/jsx-uses-vars": "error",
      "react/jsx-uses-react": "error",
      "no-unused-vars": ["error", { varsIgnorePattern: "^React$|^_", argsIgnorePattern: "^_", caughtErrors: "none" }],
    },
  },
  {
    files: ["src/test/**/*.{js,jsx}"],
    languageOptions: { globals: { ...globals.browser, ...globals.node, ...globals.vitest } },
  },
];
