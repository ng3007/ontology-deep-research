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
[You are a healthcare venture investor tasked with identifying opportunities for 0-1 company creation, this is creating NEW startups, not investing in existing ones. You are healthcare exclusive and focused AI-Innate businesses. These are thematic areas, concepts, and businesses where artificial intelligence is not an add-on, but a foundational enabler of value creation, scalability, differentiation, margin/revenue potential, and defensibility. You have already identified 1 high level idea which is -{{THEME}} The goal for this phase of research is to get to a more detailed level of analysis for each opportunity as well as prioritize within these areas those of greatest interest.

You should develop a more detailed ~1 -page research memo. **Do not just synthesize existing information provided but go and conduct additional research to refine and advance the thesis**. These memos will then be read by the investment team as part of a prioritization exercise for putting the best ideas into our company creation funnel. This memo should be broad,focus on large emergin themes and can potentially inform multiple concepts under the umbrella of this problem statement.
 ## Research Memo Output Format For each opportunity, your research focus and ~1 page memo contents should be:

### 1. Problem & Market Analysis - Problem scope and quantification (time, dollars, outcomes) -key jobs to be done.  Root causes vs. symptoms analysis - Market tailwinds and "why now" factors - Regulatory or industry shifts enabling opportunity
 ### 2. AI & Technical Innovation - Specific AI capabilities creating transformation (LLMs, computer vision, etc.) 

This 1-2 pager should be able to highlight and give context on the specific problem statement, opportunity zone etc. It should detail out the various JTBDs, workflows and stakeholders involved included status quo of the space/function]

You have already identified 1 high level idea which is {{THEME}}.
"""

# Prompt 2: your ~10-page concept generator prompt.
# Somewhere in here, reference {{EXPANDED_MEMO}} and tell the model to return JSON only.
PROMPT_2_TEMPLATE = """
Use the attached/ supplemented memo below: {{EXANDED MEMO}}and blow this up do research on new concept ideas. Generate 5-7 business concept ideas each that exploit opportunities in this broader insight. This should be a list of venture-scale $B company ideas for launching a new startup for a specific customer group. These should be concepts that address priority pain points and produce a high ROI / upside to target stakeholders Each of these concepts should be AI Innate businesses. These concepts should also be differentiated in the market. The definition of AI innate is attached under [AI Investment Framework] 




And “Respond ONLY with JSON in this structure: { "concepts": [ { ... } ] }”.


##Output

For each concept, follow the format below. Keep the output restricted to less than one page at best. 

1. Problem statement -  what is the problem and who feels it the most? Highlight using data wherever possible. What is the status quo and why does it need a solution?
Eg. We are solving the problem of escalating cybersecurity threats and unaffordable cyber insurance, which disproportionately affects administrators, compliance teams, and IT staff at U.S. healthcare SMBs (<$2B Revenue). Today, these SMBs face cybersecurity budgets (~$23B annually, about 30% of their total IT spend of ~$76B) that are insufficient to purchase and maintain effective enterprise-grade cybersecurity solutions. The status quo, characterized by fragmented tools and expensive, limited cyber insurance coverage, leaves SMBs vulnerable to frequent breaches, operational downtime, regulatory fines, legal action, reputational damage, and increased patient safety risks.

2. Core value proposition/ solution - 2-3 sentences describing what an AI innate solution would solve. Why would anyone use it? What is the key JTBD

3.GTM - Who is the champion/buyer? 
4. Why now? - what makes now the right timing to build this? Confluence of regulatory,policy, economic, technological factors

4. Comparable companies/ competitive commentary-  who are the legacy players serving this directly or in close adjacency? Who are the newly established AI focused startups/ players (<3 years old). Add 1-2 lines on how this new concept could differentiate

5. Ai innate (2-3 lines)- what makes this company AI innate? 

### AI Investment Framework 
Healthcare stands at the precipice of an AI revolution—one that promises to transform care delivery, operational efficiency, and patient outcomes at unprecedented scale. As Redesign positions itself at the forefront of this transformation, we must distinguish between genuine innovation and superficial AI adoption. This memo provides a structured, technical investment framework for identifying and building AI-driven healthcare ventures with sustainable competitive advantages and outsized return potential. 

The fundamentals remain unchanged: exceptional businesses must generate returns that significantly exceed their cost of capital over extended periods. In healthcare, this durability is paramount—bureaucratic inertia, entrenched incumbents, and lengthy sales cycles create natural barriers to the rapid scaling afforded to traditional consumer and enterprise software businesses. 

When properly leveraged, AI introduces new opportunities for cheap market entry, margin expansion, internal efficiency, and data-driven competitive moats. However, it also engenders unique complexity: intense competitive cycles, evolving regulatory scrutiny, and a paradigm shift in what makes a good business, and the very value of human capital. 

The framework that follows outlines our investment thesis for AI in healthcare, focusing on: Production factors (technology and founders) Strategy (defensible business models, scalable unit economics, and resilience to market shifts). 

Some anti-goals of this memo to note: This document explicitly strives to be evergreen and therefore avoids investment tenets that may shift over the short-to-medium term, for example specific healthcare areas or concepts; or particular enabling technologies on which we’re bullish. This document does not cover function-specific frameworks or guidelines. Instead we expect this “constitutional” content to inform many downstream artifacts, for example a full accounting of our ideal founder profile, or the ways in which this investment memo intersects our global strategy. 

A Good AI Business is AI-Innate 
It’s fashionable in 2025 for investors and CEOs to talk about companies being AI-native or AI-enabled in terms of their core business line, with these aligning roughly to product and service offerings. Our philosophy is that outstanding AI leverage is more a question of the company’s culture and talent than its business model. Put otherwise, there are great investment opportunities in both product and service companies that will be exceptional by dint of their use of technology. 

However, Redesign will not invest in companies without outstanding AI leverage. We therefore propose an orthogonal view of the AI-native/AI-enabled distinction, building and investing accordingly: Team: AI-innate vs. AI-adopted Solution: Product vs. Service An AI-innate venture relies on advanced AI capabilities to create and deliver its core value proposition–such that it could not exist in the same form without AI. 

Team: We Back AI-Innate Founders 
AI-innate businesses are those where artificial intelligence is deeply embedded in the very DNA of the company—not merely as a product feature or service enhancement, but as the basic way the business operates, competes, and creates value. AI-innate businesses could not exist without AI; AI-adopted businesses could exist but may not compete successfully nor attract early-stage venture capital. AI-innateness at the formation stage of a venture is a characteristic of the founding team rather than one of the business concept. 

AI-innate ventures have until now been, and will for some time continue to be, founded by entrepreneurs with strong technical spikes who recognize the opportunity to build companies in fundamentally new ways, leveraging AI across all aspects from engineering to sales to customer success. The startup market has thus far demonstrated a strong (but not perfect) correlation between technical acumen and the ability to build an AI-innate venture. 

AI-innate founders build and scale with lean, tech-native teams, hiring domain experts only when explicitly accretive. They often invest in small, tightly-coupled groups of bright, tech-native, adaptable generalists to “figure out” how to build scaled systems in historically nontechnical operations. For example, companies like Cursor achieved impressive early success without hiring a traditional set of functional leaders in marketing and sales. Likewise, Perplexity and Linear push product decisions into small, excellent engineering teams and operate with next to no dedicated product managers. Redesign’s own Translucent designed and built an initial product version that was sufficient to accrue two enterprise contracts in a few months with just one AI engineer and the domain expert founder/CEO fulfilling the role of CPO/Head of Sales. 

We believe that because execution speed across the AI-innate/AI-adopted divide differs by an order of magnitude, the risk of backing a non-AI-innate founder is existential at the earliest stages of a company. 

#### Business Model: Investment Framework 
Whether delivering a product or service, AI-innate companies harness LLMs to unlock entirely new categories of economic transactions or operational scale that would previously have been impossible. One facet of the AI revolution is that this technology is enabling an unprecedented productization: AI product businesses can now capture value segments that have historically necessitated a services model oriented around specialized experts–recruiters, attorneys, physicians–operating at human scale. These “legacy” services businesses suffered from poor unit economics, sublinear returns to scale, and key person risks. Analysts traditionally wrote off many services businesses as unworthy of VC investment. 

We believe that AI will create opportunities to “venturify” services organizations whose core value creation mechanism can be scaled up and out with good technology. Redesign builds AI-innate companies that capture value at this service-to-product transition, from AI scribes to agentic FP&A to clinician-backstopped, LLM-powered, chat-based care products. But while AI products will be transformational, we also believe that there is significant return potential in AI-innate services businesses that augment and scale differentiated human capital. 

That said, we hold an exceptionally high bar for services concepts, because any service – per the above – now stands at an inherent risk of being productized. Hence, an investment in (even an AI-innate) services business must derive from a strong view of why the business is insulated from AI productization pressure. One example of this insulation from full productization might be regulation preventing certain types of healthcare delivery (like prescription issuance) from being fully automated, even if the “special human in the loop” is given meaningful scale with technology. 

To clarify this distinction further: 
AI-Innate Products: The buyer primarily pays for the direct use of an AI-powered capability or tool that is largely self-service once deployed. These businesses emphasize scalable software (e.g., LLM-powered automation), with negligible marginal cost per additional customer. 

AI-Innate Services: The buyer pays for an outcome or deliverable the company provides – often blending AI with specialized human labor, as mentioned above. AI is central to achieving economic scale and efficiency, but the core buyer relationship revolves around the outcome delivered rather than the direct usage of the software itself. 

#### AI-Innate Services 
AI-innate services businesses significantly enhance traditional healthcare models through AI integration that drives robust margin improvements, scalable unit economics, or defensible moats. While we anticipate that all future healthcare ventures will incorporate AI in some capacity, our bar for investing in a services business is intentionally high. Only those ventures where AI unlocks an economic transformation of differentiated human capital will clear this hurdle. 

Specifically, the AI-innate ventures we’ll be interested in: 
Enhance a core, valuable transaction involving human expertise that could exist independently of AI, but where AI materially improves margins, accelerates time-to-value, permits new workflow categories, or notably enhances clinical or operational outcomes. 
Leverage AI for impactful automation, deep personalization, or super-human decision-making, driving SaaS-like or superior unit economics. 
Do not always require founders with deep AI research backgrounds, but founders must have exceptional technical execution capabilities, and we will not invest in AI services businesses founded by individuals without a core competency in these areas. 
Are truly AI-innate despite being services businesses, with founders who deeply understand their domain while being builders at heart. 

Example: DUOS (RH portfolio company), which provides a bespoke Medicare resource management service for seniors. The customer purchases the outcome – expert senior support– yet AI dramatically augments each human advisor’s throughput. The margins reach a SaaS-like profile precisely because AI scales the specialized labor so efficiently. 

#### AI-Innate Products 
An AI-innate product business is one whose main offering is in and of itself an AI-powered capability with a negligible marginal cost per customer. These ventures often require proprietary model training, multi-agent orchestration, and datasets with the Core Properties to meet minimum quality standards or maintain competitive differentiation. AI is foundational in these businesses, deeply embedded in product workflows, distribution, and economic models. 

We specifically look for AI-innate product ventures that demonstrate: 
Robust Unit Economics: Clear pathways to profitable scalability with AI-driven operational efficiencies. 
Durable Competitive Advantages: Technical moats rooted in proprietary or hard-to-replicate AI models and exclusive datasets (Hippocratic), exclusive distribution channels (Abridge), or a unique strategic partner (OpenEvidence). 
Scalable Market Opportunities: Large, addressable markets that can sustainably support significant venture growth. 
Category-Creating Capabilities: Product offerings that don’t just compete for the same pie, but dramatically expand it by capturing value not traditionally available to “traditional” technology ventures. 

A critical ingredient for AI product ventures is a founding team with a strong “technical spike” in the core AI technology itself. Having a co-founder or lead team member who deeply understands the specific AI architecture or methodology is often essential to building defensible technical moats, rapid and iterative execution, brilliant product design, and maintaining a competitive edge amidst – and capitalizing on – fast-moving technological developments. Redesign complements this technical leadership by bringing robust healthcare domain knowledge, strategic partnerships, and operational support to bridge critical gaps and accelerate market readiness. AI-innate founders further value our technical and product expertise in helping them strategize and execute decisively amidst scientific uncertainty. Relatedly, our ability to assess technical founders and product concepts more deeply than traditional VC outfits is a core differentiator. 

AI product companies also often build early moats by amassing signals of institutional credibility, competitive insulation, and network effects. For example, some like Lila Sciences spin out of university labs with a marquee professor-founder, protected IP assets, and deep ties to a public/private consortium that affords an early distribution advantage. 

Investment Criteria: Redesign builds only AI-innate ventures, whether the business offers a product or service. In every case, we invest on the basis of robust unit economics, durable competitive advantages, an exceptional founding team, and category-creating technical capabilities that expand—rather than merely compete for—existing value in healthcare. 

Example: Hippocratic AI is a specialized healthcare LLM. It sells an AI platform that clients directly integrate into their workflows. Ongoing usage costs are minimal compared to a services model because the platform itself is the product and is delivered as software with near-zero incremental cost. 

#### A Good AI Business Has a Moat 
A moat insulates a venture’s economic returns from erosion by potential stakeholders (competitors, customers, suppliers), each acting in their own self-interest. This is vital for any Redesign business, and even more so for AI-driven ventures where technology shifts can rapidly upend market positions. 

Stakeholder Dynamics 
We assess each stakeholder’s leverage—whether a single large health system, a payer, a specialized data source, or a technology infrastructure provider. Even the best AI solutions are vulnerable if all negotiating power rests with one key counterparty. 

Over-Earning Thought Experiment 
We envision a scenario where the business “over-earns” by raising prices or slashing costs aggressively. Do customers quickly find an alternative, or does the company’s solution remain irreplaceable? Durable AI moats often hinge on exclusive data, workflow lock-in, or technical capabilities that can’t be easily replicated. 

OpenAI/Epic Systems Test 
We ask: “Why can’t (or won’t) a major EHR incumbent or a large-scale AI provider replicate this in six months?” A strong venture must articulate distribution and technical wedges—along with the time, expertise, and ecosystem buy-in required for others to catch up. (Replace with a relevant incumbent or market leader as appropriate) 

Technical Moat: The Core Properties of Valuable Datasets in Healthcare 
One structured framework for reasoning about the durability of a technical moat is to consider whether the datasets that a business purports to accrue are uniquely valuable given how statistical models like LLMs are trained. We believe the value of a large quantity of raw, unspecific data is outweighed by a high-quality dataset with the following Core Properties: 
Prospectively gathered: the data was collected intentionally for use in an AI system rather than queried from existing systems and retrofitted. 
Expertly annotated: the data was labeled by an entity (human or AI) at least as knowledgeable as the end users whom the system will enable. 
Quality controlled: the data has been audited by domain experts to minimize systematic errors in selection, labeling, or drift over time. 
Task-aligned: the data represents a collection of examples tightly linked to the task at hand, ideally reflecting the process of an AI agent making a recommendation to a human expert and the human expert either following or modifying the recommendation. 

We are skeptical of a mere quantity of data and instead index highly on the competitive advantages of businesses that create for themselves a right to collect and capitalize on datasets with these Core Properties. 

#### A Good AI Business Gets Better as It Gets Bigger 
In AI-driven healthcare, strong ventures not only scale effectively but also see improving returns to scale: 

Marginal Cost Advantages 
The fundamental promise of a technology product is to reduce incremental costs. Once a product is developed, it can serve new customers or patient populations with minimal overhead. A good AI-driven healthcare venture’s margins should improve over time. This also applies to services businesses, where the AI enables the human expert to scale to additional users with negligible cost. 

Data Network Effects 
High-quality healthcare data is uniquely valuable (see: Core Properties). As an AI solution gathers more data across providers and patients, the algorithms get smarter, creating a feedback loop that boosts performance and broadens the moat. 

Internal Use of AI 
Not only do good AI businesses produce a valuable core offering, but they often exhibit beneficial externalities, including enhanced internal process automation, human capital-efficient development, and an ability to adapt quickly to capture adjacent market opportunities. We invest in founders who share our thesis that exceptional businesses can be built in a fundamentally new way, embedding AI and technical acumen into the DNA of every function of the venture, from software development to marketing. Note: Reaching this self-reinforcing stage is a feature of the venture as it achieves PMF. Redesign Health’s value proposition to founders is that we can help address the “cold start” problem that most founders face. 

#### A Good AI Business Is Resilient 
Resilience refers to the ability to withstand external shocks—macro downturns, regulatory shifts, or sudden AI leaps—in a sector known for complex economics and stakeholder power imbalances. 

Cyclicality & Essential Services 
AI-driven ventures that solve core healthcare problems (e.g., clinical decision support or chronic care management) are less likely to face contraction during economic downturns. Ventures selling “nice-to-have” AI tools face more volatility when budgets tighten. 

Pro-Entropic Potential 
Some of the best AI businesses thrive in chaos by adapting quickly to new regulations or large-scale AI advancements. We seek founders and ventures that can pivot their (mental, AI, and business) models and seize new distribution channels as market conditions change. 

Model & Vendor Flexibility 
Because AI is evolving rapidly, a single-point dependency on any one model or vendor can be risky. We look for solutions that plan for multiple model providers or even proprietary models if needed. 

#### A Good AI Business’s Founders Should Want to Build with Redesign 
Healthcare distribution pathways are notoriously complex: payers, health systems, PBMs, and employers control large patient populations and tightly manage budgets. AI ventures also face added regulatory hurdles (HIPAA, CMS, FDA approvals). Nevertheless, these barriers can create attractive opportunities for startups that leverage AI in ways incumbents cannot easily replicate. 

Integration & Partnership 
The quickest path to scale often involves embedding into existing workflows—e.g., EHR integrations, payer pilots, pharmacy data exchange. By doing this well, AI ventures build deeper moats and reduce churn. 

Technical & Domain Expertise 
In healthcare AI, domain knowledge is as critical as technical prowess. For AI-innate businesses, we offer a compelling opportunity for technical founders to pair with Redesign’s healthcare ecosystem and operational capabilities. 

#### What Does This Mean for Starting AI Companies at Redesign? 
A pre-seed venture is never a “good business” from day one, especially in healthcare. The real question is: How quickly can we get there, and how can Redesign accelerate that trajectory? 

Pursue Differentiated, High-Conviction Ideas 
We focus on opportunities that can realistically achieve IPO-scale potential while maintaining a clear strategic or PE off-ramp at a nine-figure valuation. This dual path to liquidity often indicates a fundamentally sound business model—one that can either remain independent long-term or appeal to acquirers seeking top-tier AI solutions. 

Redesign Health’s Alpha 
We strive to partner with founders to build ventures that enjoy a unique edge from the start: 
First customer relationships delivered pre-funding. 
VC partner and capital commitments secured via our network. 
Foundational data sets or hard-to-secure IP licenses that serve as immediate moats. 
Tentpole advisors who lend domain or technical expertise without requiring a full-time role. 
AI products offered to portfolio companies that provide infinite, on-demand access to our differentiated knowledge and relationships 

Emphasize Comprehensive Partnership 
We stay actively engaged with each company both before and after funding—helping shape go-to-market and technology strategies, refine fundraising approaches, and offer operational insights. Our involvement is designed to anticipate needs and connect the right dots so founders can focus on building and scaling the business. We recognize that the bulk of our value is often delivered after funding. 

Complement Founder Profile & Team Composition 
For AI-native ventures especially, we require highly technical founder/CEOs who can rapidly iterate and refine advanced models. We then bring our healthcare knowledge and resources to bridge regulatory and operational gaps critical for success. 

Maintain Foresight in Scaling 
Both healthcare and AI evolve quickly. We continuously re-evaluate a venture’s moat, its path to scale, and its resilience. Shifting regulations, competitor moves, and AI breakthroughs are inevitable, and we help founders adapt proactively.


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

