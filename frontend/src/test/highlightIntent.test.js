import { describe, expect, it } from "vitest";
import { highlightTerm, readIntent } from "../highlightIntent.js";

// The whole point of this module is restraint. Every case below that starts
// with "leaves" is a real question that must reach the model, not be quietly
// turned into a document search.
describe("highlightTerm", () => {
  it("takes the word from the plain command", () => {
    expect(highlightTerm("highlight velocity")).toBe("velocity");
    expect(highlightTerm("Highlight velocity")).toBe("velocity");
    expect(highlightTerm("mark velocity")).toBe("velocity");
    expect(highlightTerm("  highlight   velocity  ")).toBe("velocity");
  });

  it("looks past the filler people put around the word", () => {
    expect(highlightTerm("highlight all occurrences of velocity")).toBe("velocity");
    expect(highlightTerm("highlight all of velocity")).toBe("velocity");
    expect(highlightTerm("highlight every mention of velocity")).toBe("velocity");
    expect(highlightTerm("highlight all the instances of velocity")).toBe("velocity");
    expect(highlightTerm("highlight the word velocity")).toBe("velocity");
    expect(highlightTerm("highlight velocity")).toBe("velocity");
  });

  it("takes a multi-word phrase", () => {
    expect(highlightTerm("highlight escape velocity")).toBe("escape velocity");
    expect(highlightTerm("highlight all occurrences of escape velocity")).toBe("escape velocity");
  });

  it("honours quotes so a phrase can contain punctuation", () => {
    expect(highlightTerm('highlight the word "escape velocity"')).toBe("escape velocity");
    expect(highlightTerm("highlight 'photosynthesis rate'")).toBe("photosynthesis rate");
  });

  it("looks past politeness", () => {
    expect(highlightTerm("please highlight velocity")).toBe("velocity");
    expect(highlightTerm("can you highlight velocity")).toBe("velocity");
    expect(highlightTerm("Could you please highlight all occurrences of velocity?")).toBe("velocity");
  });

  it("understands the where-is phrasing", () => {
    expect(highlightTerm("where does velocity appear")).toBe("velocity");
    expect(highlightTerm("where does velocity appear in the documents")).toBe("velocity");
    expect(highlightTerm("show me where velocity shows up")).toBe("velocity");
    expect(highlightTerm("where is velocity mentioned")).toBe("velocity");
    expect(highlightTerm("find all occurrences of velocity")).toBe("velocity");
    expect(highlightTerm("search for all occurrences of velocity")).toBe("velocity");
  });

  it("ignores a trailing scope, since every search is session-scoped anyway", () => {
    expect(highlightTerm("highlight velocity in my notes")).toBe("velocity");
    expect(highlightTerm("highlight velocity across the sources")).toBe("velocity");
  });

  it("returns the user's own casing", () => {
    expect(highlightTerm("highlight Photosynthesis")).toBe("Photosynthesis");
    expect(highlightTerm("Highlight ESCAPE VELOCITY")).toBe("ESCAPE VELOCITY");
  });

  // The dangerous cases.
  it("leaves ordinary questions to the model", () => {
    expect(highlightTerm("what is escape velocity?")).toBeNull();
    expect(highlightTerm("how does the photometer work?")).toBeNull();
    expect(highlightTerm("why is the sky blue?")).toBeNull();
    expect(highlightTerm("compare the two papers")).toBeNull();
    expect(highlightTerm("explain the second law")).toBeNull();
    expect(highlightTerm("summarize chapter 2")).toBeNull();
  });

  // "where is X" is only a highlight request when something says the word is
  // expected to appear. Without that, it is a question.
  it("does not read a location question as a highlight request", () => {
    expect(highlightTerm("where is the capital of France")).toBeNull();
    expect(highlightTerm("where are the CCDs described")).toBeNull();
    expect(highlightTerm("where is the photometer calibration discussed")).toBeNull();
  });

  it("does not read a comparison request as a phrase to search for", () => {
    expect(highlightTerm("highlight the difference between the two papers")).toBeNull();
    expect(highlightTerm("show the difference")).toBeNull();
  });

  it("refuses a sentence that is too long to be a search term", () => {
    expect(highlightTerm("highlight what the second paper says about spectral lines")).toBeNull();
    expect(highlightTerm("mark the section that explains how the photometer was calibrated")).toBeNull();
  });

  it("refuses a term that is not searchable", () => {
    expect(highlightTerm("highlight ???")).toBeNull();
    expect(highlightTerm("highlight ---")).toBeNull();
    expect(highlightTerm("highlight")).toBeNull();
    expect(highlightTerm("")).toBeNull();
    expect(highlightTerm("   ")).toBeNull();
    expect(highlightTerm("highlight 12345")).toBeNull();
  });

  it("refuses a term longer than the server accepts", () => {
    expect(highlightTerm(`highlight ${"a".repeat(61)}`)).toBeNull();
    expect(highlightTerm(`highlight ${"a".repeat(60)}`)).toBe(`a${"a".repeat(59)}`);
  });

  it("does not get excited about punctuation or emoji", () => {
    expect(highlightTerm("highlight ✨")).toBeNull();
    expect(highlightTerm("highlight velocity!")).toBe("velocity");
    expect(highlightTerm("highlight velocity?")).toBe("velocity");
    // A term full of punctuation is not something anyone is searching for.
    expect(highlightTerm("highlight a.b.c.d")).toBeNull();
  });

  it("keeps digits and hyphens, which real terms contain", () => {
    expect(highlightTerm("highlight co2")).toBe("co2");
    expect(highlightTerm("highlight kuiper")).toBe("kuiper");
    expect(highlightTerm("highlight Kepler-22b")).toBe("Kepler-22b");
  });

  it("survives input that is not a string", () => {
    expect(highlightTerm(undefined)).toBeNull();
    expect(highlightTerm(null)).toBeNull();
    expect(highlightTerm(42)).toBeNull();
  });
});

// The reason `readIntent` exists. Before it, a message carrying two
// instructions went to the model whole, so retrieval was searching for the words
// "highlight 30 days" as well as the question - noise in the dense query and in
// BM25, which nothing downstream can recover from.
describe("readIntent", () => {
  it("splits a question from a highlight asked in the same message", () => {
    expect(readIntent("what is the return window? highlight 30 days")).toEqual({
      question: "what is the return window",
      term: "30 days",
    });
    expect(readIntent("what is the fee, highlight late payment")).toEqual({
      question: "what is the fee",
      term: "late payment",
    });
    expect(readIntent("What are the payment terms, and highlight the penalty clause")).toEqual({
      question: "What are the payment terms",
      term: "the penalty clause",
    });
  });

  // The polite forms are the common ones, and losing them would make the feature
  // feel broken for the way people actually type.
  it("splits through politeness", () => {
    expect(readIntent("what is the fee? please highlight late payment")).toEqual({
      question: "what is the fee",
      term: "late payment",
    });
  });

  it("accepts every verb the standalone command accepts", () => {
    expect(readIntent("what is the fee? mark the deadline").term).toBe("the deadline");
    expect(readIntent("what is the fee? underline the heading").term).toBe("the heading");
  });

  it("reports a bare command as a lookup with no question", () => {
    expect(readIntent("highlight velocity")).toEqual({ question: null, term: "velocity" });
    expect(readIntent("where does velocity appear")).toEqual({
      question: null,
      term: "velocity",
    });
  });

  it("passes an ordinary question through untouched", () => {
    expect(readIntent("what is escape velocity")).toEqual({
      question: "what is escape velocity",
      term: null,
    });
    expect(readIntent("what is the fee?")).toEqual({ question: "what is the fee?", term: null });
  });

  // These are the failures that matter. A question quietly turned into a word
  // search is a wrong answer, not a missing feature.
  it("refuses to split a question that mentions highlighting", () => {
    expect(readIntent("what does the contract say about highlighting the deadline")).toEqual({
      question: "what does the contract say about highlighting the deadline",
      term: null,
    });
    expect(readIntent("how do I highlight a word")).toEqual({
      question: "how do I highlight a word",
      term: null,
    });
    expect(readIntent("why would you mark the heading")).toEqual({
      question: "why would you mark the heading",
      term: null,
    });
  });

  it("refuses a trailing verb with no term to look for", () => {
    expect(readIntent("what is the fee? highlight")).toEqual({
      question: "what is the fee? highlight",
      term: null,
    });
    expect(readIntent("what is the fee, highlight")).toEqual({
      question: "what is the fee, highlight",
      term: null,
    });
  });

  // The standalone parser's refusals still apply inside a longer message, so the
  // two paths cannot disagree about what counts as a term.
  it("applies the standalone rules to the trailing clause", () => {
    // "the difference between them" is a refusal in the standalone parser, and
    // it must stay one here: the two paths cannot disagree about what a term is.
    expect(readIntent("compare the two, highlight the difference between them").term).toBeNull();
    expect(readIntent("what is it? highlight the difference").term).toBeNull();
  });

  // Too little text in front of the verb to be a question. Asking the model to
  // answer "ok" would be worse than just running the lookup.
  it("does not treat a stray word in front of the verb as a question", () => {
    expect(readIntent("ok highlight velocity")).toEqual({
      question: "ok highlight velocity",
      term: null,
    });
    expect(readIntent("now highlight velocity")).toEqual({
      question: "now highlight velocity",
      term: null,
    });
  });

  // A documented limit rather than an accident: "30 days and 2%" contains the
  // same conjunction the question splitter uses, so accepting it would mean
  // guessing where the term ends and the question resumes. Guessing wrong sends
  // the wrong half to the model, which is worse than not splitting at all.
  it("refuses a highlight term that ends in a conjunction", () => {
    // The standalone parser refuses it too, so a bare command is asked as a
    // whole rather than looking up the wrong span. Both paths agree, which is
    // the property the other rule in this file is protecting.
    expect(readIntent("what is the fee? highlight 30 days and 2%")).toEqual({
      question: "what is the fee? highlight 30 days and 2%",
      term: null,
    });
    expect(readIntent("highlight 30 days and 2%")).toEqual({
      question: "highlight 30 days and 2%",
      term: null,
    });
  });

  it("survives input that is not a string", () => {
    expect(readIntent(undefined)).toEqual({ question: null, term: null });
    expect(readIntent(null)).toEqual({ question: null, term: null });
    expect(readIntent(42)).toEqual({ question: null, term: null });
    expect(readIntent("   ")).toEqual({ question: null, term: null });
  });
});