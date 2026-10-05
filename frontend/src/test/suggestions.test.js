import { describe, expect, it } from "vitest";
import { subjectFromName, suggestionsFor } from "../suggestions.js";

describe("subjectFromName", () => {
  it("strips the extension and separators so it reads inside a sentence", () => {
    expect(subjectFromName("q3-returns-policy_v2.pdf")).toBe("q3 returns policy v2");
    expect(subjectFromName("kepler.pdf")).toBe("kepler");
  });

  it("is safe on a missing name, which the retriever cannot help with", () => {
    expect(subjectFromName(undefined)).toBe("");
    expect(subjectFromName("")).toBe("");
  });
});

describe("suggestionsFor", () => {
  it("offers nothing for an empty notebook, because every answer would be a refusal", () => {
    expect(suggestionsFor([])).toEqual([]);
    expect(suggestionsFor(null)).toEqual([]);
  });

  it("asks about the document rather than in general", () => {
    const out = suggestionsFor([{ name: "returns-policy.pdf", kind: "pdf", numeric: 0 }]);
    expect(out).toContain("What are the key points about returns policy?");
  });

  it("only offers a figures question when there is reason to think there are figures", () => {
    // Prose that mentions numbers is not a figures notebook.
    const prose = suggestionsFor([{ name: "manifesto.txt", kind: "txt", numeric: 0 }]);
    expect(prose.some((q) => /figures and dates/.test(q))).toBe(false);

    // A genuine numeric dump sets the flag.
    const dump = suggestionsFor([{ name: "metrics.log", kind: "txt", numeric: 40 }]);
    expect(dump.some((q) => /figures and dates/.test(q))).toBe(true);

    // A spreadsheet does not, because the CSV parser reshapes tables into
    // `Label: Value` lines that never look numeric-dense. The kind covers it.
    const sheet = suggestionsFor([{ name: "revenue.csv", kind: "csv", numeric: 0 }]);
    expect(sheet.some((q) => /figures and dates/.test(q))).toBe(true);
  });

  it("treats a missing kind or numeric field as no signal rather than throwing", () => {
    const out = suggestionsFor([{ name: "old.txt" }]);
    expect(out.some((q) => /figures and dates/.test(q))).toBe(false);
    expect(out.length).toBeGreaterThan(0);
  });

  it("compares two documents only when there are two", () => {
    const one = suggestionsFor([{ name: "a.txt", kind: "txt", numeric: 0 }]);
    expect(one.some((q) => /relate to/.test(q))).toBe(false);

    const two = suggestionsFor([
      { name: "returns-policy.pdf", kind: "pdf", numeric: 0 },
      { name: "refund-form.docx", kind: "docx", numeric: 0 },
    ]);
    expect(two.some((q) => /refund form/.test(q))).toBe(true);
  });

  it("skips a filename too long to read inside a sentence", () => {
    const out = suggestionsFor([
      { name: `${"a".repeat(80)}.txt`, kind: "txt", numeric: 0 },
      { name: "notes.txt", kind: "txt", numeric: 0 },
    ]);
    expect(out.every((q) => !q.includes("a".repeat(80)))).toBe(true);
    expect(out.some((q) => /notes/.test(q))).toBe(true);
  });

  it("never returns duplicates, which would render as a repeated React key", () => {
    const out = suggestionsFor([
      { name: "report.txt", kind: "txt", numeric: 0 },
      { name: "report.txt", kind: "txt", numeric: 0 },
    ]);
    expect(new Set(out).size).toBe(out.length);
  });

  it("still offers something when every filename is unusable", () => {
    // "r2" is a real case: stripping the extension left a two-character name
    // that would read as "the key points about r2?". Dropping the name is right;
    // dropping every question except the catch-all was not, so the fallback has
    // to carry the empty state on its own.
    const out = suggestionsFor([{ name: ".pdf", kind: "pdf", numeric: 0 }]);
    expect(out).toContain("Summarise these sources in a few sentences");
    expect(out.every((q) => !q.includes("about ?"))).toBe(true);

    const short = suggestionsFor([{ name: "r2.md", kind: "md", numeric: 0 }]);
    expect(short.length).toBeGreaterThan(1);
    expect(short.some((q) => /\br2\b/.test(q))).toBe(false);
  });

  it("never emits a dangling preposition when a name is dropped", () => {
    const out = suggestionsFor([{ name: "x.txt", kind: "txt", numeric: 0 }]);
    for (const q of out) {
      expect(q).not.toMatch(/\s\s/);
      expect(q).not.toMatch(/\s[?.]$/);
    }
  });
});