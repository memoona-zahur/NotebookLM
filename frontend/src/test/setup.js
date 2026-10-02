// Extra matchers (toBeInTheDocument, toHaveTextContent, ...) and the cleanup
// Testing Library requires between tests.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);

// jsdom implements neither of these, and both are used by the citation
// scrolling and by focus-visible styles respectively.
if (!globalThis.CSS?.escape) {
  globalThis.CSS = {
    escape: (value) => String(value).replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`),
  };
}

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}