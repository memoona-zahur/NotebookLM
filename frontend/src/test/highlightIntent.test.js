import { describe, expect, it } from "vitest";
import { highlightTerm } from "../highlightIntent.js";

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