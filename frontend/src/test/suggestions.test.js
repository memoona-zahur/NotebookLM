import { describe, expect, it } from "vitest";
import {
  EXPLORE,
  subjectFromName,
  suggestionsFor,
  suggestionsWithExplore,
} from "../suggestions.js";

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
describe("machine-generated filenames", () => {
  // Regression: uploads are stored under a UUID (store.py writes
  // "<uuid>.<ext>"), so the app offered "What are the key points about
  // 2a7062e3acce425790117aba9bdc1167?" and then refused to answer it. Retrieval
  // finds nothing for a UUID, so the suggestion was a question the app had
  // invented and could not answer.
  it("never builds a question out of a generated filename", () => {
    const generated = [
      "2a7062e3acce425790117aba9bdc1167.pdf",
      "00f622add57a4e839c2911439981d242.pdf",
      "deadbeefcafe.pdf",
      "a1b2c3d4e5f6.pdf",
      "0123456789abcdef0123456789abcdef.pdf",
      "DEADBEEF0123.pdf",
    ];
    for (const name of generated) {
      expect(subjectFromName(name), name).toBe("");
      const out = suggestionsFor([{ name, kind: "pdf", numeric: 0 }]).join(" ");
      expect(out, name).not.toMatch(/[0-9a-f]{8}/i);
      // It must still suggest something, just not the UUID.
      expect(out.length, name).toBeGreaterThan(0);
    }
  });

  it("still uses real filenames that happen to contain digits", () => {
    // The generated-name check must not eat legitimate subjects.
    for (const name of [
      "q3-returns-policy_v2.pdf",
      "2024-financial-summary.csv",
      "iso-27001-controls.pdf",
    ]) {
      const subject = subjectFromName(name);
      expect(subject, name).not.toBe("");
      expect(subject, name).toMatch(/[aeiouy]/i);
    }
  });
});

describe("questions the app can actually answer", () => {
  // Regression: the list ended with "What questions should I be asking about
  // this?", which is a question about the app rather than the corpus. Clicking
  // it sent that text to the model, which found no passage about how to use the
  // app and refused - so the app answered its own suggestion with "I could not
  // find anything relevant in the indexed sources".
  it("offers no question about the app itself", () => {
    const sets = [
      [{ name: "brew-guide.pdf", kind: "pdf", numeric: 0 }],
      [{ name: "notes.txt", kind: "txt", numeric: 0 }],
      [
        { name: "a.pdf", kind: "pdf", numeric: 0 },
        { name: "b.md", kind: "md", numeric: 0 },
      ],
      [{ name: "sales.csv", kind: "csv", numeric: 0 }],
      [],
    ];
    for (const sources of sets) {
      for (const q of suggestionsFor(sources)) {
        expect(q, JSON.stringify(sources)).not.toMatch(
          /what questions (should|can|could) i/i
        );
        expect(q, JSON.stringify(sources)).not.toMatch(/how do i (use|ask)/i);
      }
    }
  });

  it("keeps the explore control out of the question list", () => {
    const sources = [{ name: "notes.txt", kind: "txt", numeric: 0 }];
    const withControl = suggestionsWithExplore(sources);
    expect(withControl).toContain(EXPLORE);

    const questions = withControl.filter((x) => x !== EXPLORE);
    expect(questions).toEqual(suggestionsFor(sources));
    for (const q of questions) {
      expect(typeof q).toBe("string");
      expect(q).not.toBe(EXPLORE);
    }

    // Nothing to reveal for an empty notebook, and no stray control.
    expect(suggestionsWithExplore([])).toEqual([]);
  });
});
