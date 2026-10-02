import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Answer, Citation, Evidence } from "../Answer.jsx";

describe("Answer", () => {
  it("turns citation markers into clickable chips", () => {
    render(<Answer text="Orbit speed is 11.2 km/s [1] and period 90 min [2]." cited={[1, 2]} />);
    const refs = screen.getAllByRole("link");
    expect(refs).toHaveLength(2);
    expect(refs[0]).toHaveAttribute("href", "#cite-1");
    expect(refs[1]).toHaveAttribute("href", "#cite-2");
    expect(screen.getByText(/Orbit speed is 11.2 km\/s/)).toBeInTheDocument();
  });

  it("keeps text either side of a marker", () => {
    render(<Answer text="before [1] after" cited={[1]} />);
    expect(screen.getByText(/before/)).toBeInTheDocument();
    expect(screen.getByText(/after/)).toBeInTheDocument();
  });

  it("renders plain text when there are no markers", () => {
    render(<Answer text="No citations here at all." cited={[]} />);
    expect(screen.queryAllByRole("link")).toHaveLength(0);
    expect(screen.getByText("No citations here at all.")).toBeInTheDocument();
  });

  // The server strips out-of-range markers, but if one ever reached the client a
  // dead link would be worse than showing the text unlinked.
  it("does not link a marker pointing past the evidence", () => {
    render(<Answer text="claim [1] and a second [5]" cited={[1]} />);
    const refs = screen.getAllByRole("link");
    expect(refs).toHaveLength(1);
    expect(refs[0]).toHaveAttribute("href", "#cite-1");
  });

  it("drops a zero or negative index rather than throwing", () => {
    render(<Answer text="bad [0] and [1]" cited={[1]} />);
    expect(screen.getAllByRole("link")).toHaveLength(1);
  });
});

describe("Citation", () => {
  const citation = {
    source: "kepler.pdf",
    page: 3,
    score: 0.42,
    text: "The photometer has 42 CCDs.",
  };

  it("collapses the passage until clicked", () => {
    render(
      <div className="thread">
        <Citation citation={citation} index={0} cited={[1]} />
      </div>,
    );
    // The card is a button in the markup; the snippet is CSS-hidden, so
    // behaviour is asserted through the open class instead of visibility.
    expect(document.querySelector(".cite")).not.toHaveClass("open");
  });

  it("is marked uncited when the model did not use it", () => {
    render(<Citation citation={citation} index={1} cited={[1]} />);
    expect(document.querySelector(".cite")).toHaveClass("uncited");
  });

  it("treats a passage as cited when the model cited nothing at all", () => {
    render(<Citation citation={citation} index={0} cited={[]} />);
    expect(document.querySelector(".cite")).not.toHaveClass("uncited");
  });

  it("shows the source, page and score", () => {
    render(<Citation citation={citation} index={0} cited={[1]} />);
    expect(screen.getByText(/kepler\.pdf/)).toBeInTheDocument();
    expect(screen.getByText(/p\.3/)).toBeInTheDocument();
    expect(screen.getByText("42%")).toBeInTheDocument();
  });
});

describe("Evidence", () => {
  it("reports the floor and that the model was not asked", () => {
    render(
      <Evidence
        evidence={{
          verdict: "no_match",
          best_score: 0.11,
          min_score: 0.25,
        }}
      />,
    );
    expect(screen.getByText(/11%/)).toBeInTheDocument();
    expect(screen.getByText(/25%/)).toBeInTheDocument();
    expect(screen.getByText(/was not asked/)).toBeInTheDocument();
  });

  it("shows the confidence chip and candidate accounting", () => {
    render(
      <Evidence
        evidence={{
          verdict: "answered",
          confidence: "high",
          best_score: 0.5114,
          considered: 40,
          returned: 6,
          invalid: [],
        }}
      />,
    );
    expect(screen.getByText("high confidence")).toBeInTheDocument();
    expect(screen.getByText(/6 of 40 candidates used/)).toBeInTheDocument();
  });

  it("names the citations the server removed", () => {
    render(
      <Evidence
        evidence={{
          verdict: "answered",
          confidence: "medium",
          best_score: 0.3,
          considered: 10,
          returned: 3,
          invalid: [2, 4],
        }}
      />,
    );
    expect(screen.getByText(/removed citation \[2\], \[4\]/)).toBeInTheDocument();
  });

  it("renders nothing when there is no evidence", () => {
    const { container } = render(<Evidence evidence={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});