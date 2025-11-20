import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# ----------------- CONFIG -----------------

load_dotenv()  # load .env locally
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4.1"  # or "gpt-4o" if you prefer


# ------- PROMPTS (PASTE YOURS, WITH PLACEHOLDERS) -------

# Prompt 1: your 1–2 page memo prompt.
# Replace the original “you have already identified 1 high level idea…” line
# with the exact placeholder {{THEME}} where the OZ text should go.
PROMPT_1_TEMPLATE = """
[PASTE YOUR FULL PROMPT 1 TEXT HERE]

You have already identified 1 high level idea which is {{THEME}}.
"""

# Prompt 2: your ~10-page concept generator prompt.
# Somewhere in here, reference {{EXPANDED_MEMO}} and tell the model to return JSON only.
PROMPT_2_TEMPLATE = """
[PASTE YOUR FULL PROMPT 2 TEXT HERE]

Use the following expanded memo as input:
{{EXPANDED_MEMO}}

Respond ONLY with valid JSON in this exact format:
{
  "concepts": [
    {
      "name": "",
      "problem": "",
      "solution": "",
      "user": "",
      "why_now": "",
      "comparables": "",
      "differentiation": "",
      "risks": ""
    }
  ]
}
"""


# ----------------- DATA STRUCTURES -----------------


@dataclass
class Concept:
    name: str
    problem: str
    solution: str
    user: str
    why_now: str
    comparables: str
    differentiation: str
    risks: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concept":
        return cls(
            name=d.get("name", ""),
            problem=d.get("problem", ""),
            solution=d.get("solution", ""),
            user=d.get("user", ""),
            why_now=d.get("why_now", ""),
            comparables=d.get("comparables", ""),
            differentiation=d.get("differentiation", ""),
            risks=d.get("risks", ""),
        )


@dataclass
class ResearchResult:
    oz_text: str
    expanded_memo: str
    concepts: List[Concept]


# ----------------- CORE CALLS (WITH WEB SEARCH) -----------------


def run_prompt_1(oz_text: str) -> str:
    """
    Runs Prompt 1 with web search to produce the expanded 1–2 page memo.
    """
    prompt = PROMPT_1_TEMPLATE.replace("{{THEME}}", oz_text)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=[{"type": "web_search_preview"}],
    )

    # openai-python v1 exposes a merged text helper:
    expanded_memo = response.output_text
    return expanded_memo


def run_prompt_2(expanded_memo: str) -> List[Concept]:
    """
    Runs Prompt 2 with web search to produce JSON concepts.
    """
    prompt = PROMPT_2_TEMPLATE.replace("{{EXPANDED_MEMO}}", expanded_memo)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=[{"type": "web_search_preview"}],
    )

    raw = response.output_text.strip()

    # Strip ```json ... ``` if the model wraps the JSON
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        # For debugging if JSON parsing fails
        raise RuntimeError(f"Failed to parse JSON from Prompt 2: {e}\nRaw output (truncated):\n{raw[:1000]}")

    concepts_raw = data.get("concepts", [])
    if not isinstance(concepts_raw, list):
        raise RuntimeError(f"Prompt 2 JSON did not contain a 'concepts' list. Got keys: {list(data.keys())}")

    return [Concept.from_dict(c) for c in concepts_raw]


def run_research_for_oz(oz_text: str) -> ResearchResult:
    """
    Run Prompt 1 and Prompt 2 for a single OZ theme string.
    """
    print(f"Running Prompt 1 (expand theme) for OZ:\n{oz_text}\n")
    expanded_memo = run_prompt_1(oz_text)
    print("Prompt 1 done.\n")

    print("Running Prompt 2 (concept generation)...\n")
    concepts = run_prompt_2(expanded_memo)
    print(f"Prompt 2 done. Generated {len(concepts)} concepts.\n")

    return ResearchResult(
        oz_text=oz_text,
        expanded_memo=expanded_memo,
        concepts=concepts,
    )


# ----------------- CLI ENTRYPOINT (for Step 1) -----------------


def main():
    # For Step 1, we just hardcode a sample OZ or accept from input.
    oz_text = input("Enter OZ / Taxonomy theme: ").strip()
    if not oz_text:
        print("No OZ text provided, exiting.")
        return

    result = run_research_for_oz(oz_text)

    # Print a concise summary to console
    print("\n========== EXPANDED MEMO ==========\n")
    print(result.expanded_memo)

    print("\n========== CONCEPTS (SUMMARY) ==========\n")
    for i, c in enumerate(result.concepts, start=1):
        print(f"Concept {i}: {c.name}")
        print(f"  Problem      : {c.problem[:200]}{'...' if len(c.problem) > 200 else ''}")
        print(f"  Solution     : {c.solution[:200]}{'...' if len(c.solution) > 200 else ''}")
        print(f"  Why now      : {c.why_now[:200]}{'...' if len(c.why_now) > 200 else ''}")
        print()

    # Also dump full structured result to JSON file (we'll use this in later steps)
    out = {
        "oz_text": result.oz_text,
        "expanded_memo": result.expanded_memo,
        "concepts": [c.__dict__ for c in result.concepts],
    }

    with open("last_run_output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved full structured output to last_run_output.json")


if __name__ == "__main__":
    main()

