import os
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

# Try to import Streamlit for secrets access (may not be available in all contexts)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-haiku-latest",
    LLMProvider.OLLAMA: "gemma3:12b",
}


def _get_secret(key: str) -> str | None:
    """Get secret from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key)
        except Exception:
            pass
    # Fall back to environment variables
    return os.environ.get(key)


def _detect_available_provider() -> LLMProvider:
    """Auto-detect which LLM provider is available based on secrets/env vars."""
    # Check for API keys
    if _get_secret("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    if _get_secret("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC
    # Default to Ollama for local development
    return LLMProvider.OLLAMA


def _get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider, with auto-detection fallback."""
    # Check for explicit provider setting
    provider_setting = _get_secret("LLM_PROVIDER")
    if provider_setting:
        try:
            return LLMProvider(provider_setting.lower())
        except ValueError:
            pass
    # Auto-detect based on available credentials
    return _detect_available_provider()


def _get_model_name(provider: LLMProvider) -> str:
    """Get model name from config or use default."""
    model_key = f"{provider.value.upper()}_MODEL"
    custom_model = _get_secret(model_key)
    return custom_model or DEFAULT_MODELS[provider]

# =============================================================================
# SALES CONFIGURATION
# =============================================================================

SALES_ARGUMENTS = """
1. TIME SAVINGS: Companies save 10-20 hours per month on invoicing and bookkeeping
2. REAL-TIME VISIBILITY: Instant view of cash flow and payment status
3. AUTOMATION: Automated invoicing reduces errors and speeds up payments
4. INDUSTRY EXPERTISE: Specialized solutions for construction and service industries
5. SOCIAL PROOF: Reference customer Rakennus Mäkelä reduced admin time by 60%
6. LOW COMMITMENT: 20-minute Teams demo, no obligations
"""

PITCH_VARIANTS = {
    "A": {
        "name": "Pain-First",
        "description": "Lead with discovery questions to uncover pain points before pitching",
        "elements": [
            "Open with discovery question about current situation",
            "Let prospect describe their challenges",
            "Connect solution to stated pain points",
            "Use empathy statements",
            "Soft close with meeting suggestion",
        ],
    },
    "B": {
        "name": "Value-First",
        "description": "Lead with bold value proposition and social proof upfront",
        "elements": [
            "Open with strong value statement and numbers",
            "Immediately mention reference customer success",
            "Create urgency with industry trends",
            "Direct questions about decision timeline",
            "Assertive close pushing for specific meeting time",
        ],
    },
}

# Standard output format instruction
OUTPUT_FORMAT = """
IMPORTANT: Structure your response as follows:
1. Start with "KEY TAKEAWAYS" - exactly 3-5 bullet points with the most important insights
2. Then provide "DETAILED ANALYSIS" with full explanations

Example format:
## KEY TAKEAWAYS
• [Most important point 1]
• [Most important point 2]
• [Most important point 3]

## DETAILED ANALYSIS
[Full detailed analysis here...]
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """Get LLM instance based on configured provider."""
    provider = _get_llm_provider()
    model = _get_model_name(provider)

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        api_key = _get_secret("OPENAI_API_KEY")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    elif provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        api_key = _get_secret("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    else:  # Ollama (local)
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)


def get_current_provider_info() -> dict:
    """Get info about the current LLM provider (useful for UI display)."""
    provider = _get_llm_provider()
    return {
        "provider": provider.value,
        "model": _get_model_name(provider),
    }


def format_transcript(transcript: list[dict]) -> str:
    lines = []
    for entry in transcript:
        speaker = "SDR" if entry["speaker"] == "sdr" else "PROSPECT"
        lines.append(f"[{entry['timestamp']}] {speaker}: {entry['text']}")
    return "\n".join(lines)


def format_call_summary(data: dict) -> str:
    metadata = data["metadata"]
    return f"""
--- {metadata['prospect']['company']} ({metadata['prospect']['industry']}) ---
Outcome: {metadata['outcome']}
Duration: {metadata['duration_seconds']}s
Prospect: {metadata['prospect']['contact_name']} ({metadata['prospect']['title']})

Transcript:
{format_transcript(data['transcript'])}
"""


def format_call_brief(data: dict) -> str:
    metadata = data["metadata"]
    return (
        f"{metadata['prospect']['company']} | {metadata['prospect']['industry']} | "
        f"{metadata['outcome']} | {metadata['duration_seconds']}s"
    )


# =============================================================================
# SDR FUNCTIONS - Call analysis, feedback, coaching
# =============================================================================


def prepare_for_call(prospect_info: str) -> str:
    """Pre-call preparation and guidance based on prospect data."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior SDR coach preparing a rep for a cold call.
Provide actionable, specific guidance they can use immediately.
{OUTPUT_FORMAT}"""),
        ("human", """Prepare me for a cold call to this prospect.

PROSPECT INFORMATION:
{prospect_info}

OUR SALES ARGUMENTS:
{sales_arguments}

Provide:

## KEY TAKEAWAYS
• Most important thing to know about this prospect
• Best opening approach for their role/industry
• Primary pain point to explore
• Top argument to lead with
• Key risk to avoid

## DETAILED ANALYSIS

PROSPECT RESEARCH:
- What this role typically cares about
- Common challenges in their industry
- Likely current situation

RECOMMENDED OPENING (exact script):
- First 15 seconds: What to say
- Permission question to ask
- Value hook for their specific role

DISCOVERY QUESTIONS TO ASK:
- 3 questions to uncover pain points
- Follow-up probes based on likely answers

BEST ARGUMENTS FOR THIS PROSPECT:
- Rank our arguments by relevance to this role/industry
- How to phrase each for maximum impact

OBJECTIONS TO EXPECT:
- 3 likely objections from this role
- Prepared responses for each

CLOSING APPROACH:
- How to ask for the meeting
- Backup if they resist"""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "prospect_info": prospect_info,
        "sales_arguments": SALES_ARGUMENTS,
    })
    return response.content


def quick_call_prep(
    company: str,
    contact_name: str,
    title: str,
    industry: str,
    employee_count: int | None = None,
    notes: str = "",
) -> str:
    """Quick structured pre-call prep from basic prospect data."""
    prospect_info = f"""Company: {company}
Contact: {contact_name}
Title: {title}
Industry: {industry}"""
    
    if employee_count:
        prospect_info += f"\nCompany Size: {employee_count} employees"
    if notes:
        prospect_info += f"\nAdditional Notes: {notes}"

    return prepare_for_call(prospect_info)


def analyze_call(data: dict) -> str:
    """Analyze a single call against sales arguments."""
    metadata = data["metadata"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales coach. Be direct and actionable.\n{OUTPUT_FORMAT}"),
        ("human", """Analyze this call against our sales arguments.

SALES ARGUMENTS:
{sales_arguments}

CALL: {outcome} | {industry}

TRANSCRIPT:
{transcript}

Analyze: Which arguments were used effectively? Which were missed? What to do differently?"""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "sales_arguments": SALES_ARGUMENTS,
        "outcome": metadata["outcome"],
        "industry": metadata["prospect"]["industry"],
        "transcript": format_transcript(data["transcript"]),
    })
    return response.content


def analyze_prospect(data: dict) -> str:
    """Analyze prospect's communication style."""
    metadata = data["metadata"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales psychologist analyzing communication styles.\n{OUTPUT_FORMAT}"),
        ("human", """Analyze prospect's style from this call.

PROSPECT: {prospect_name} ({prospect_title}) at {prospect_company}
INDUSTRY: {industry}

TRANSCRIPT:
{transcript}

Analyze: Communication style (pace, formality, decision style), Personality type, Tone and engagement level."""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "prospect_name": metadata["prospect"]["contact_name"],
        "prospect_title": metadata["prospect"]["title"],
        "prospect_company": metadata["prospect"]["company"],
        "industry": metadata["prospect"]["industry"],
        "transcript": format_transcript(data["transcript"]),
    })
    return response.content


def generate_personalized_arguments(data: dict, prospect_analysis: str) -> str:
    """Generate tailored arguments based on prospect profile."""
    metadata = data["metadata"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales strategist crafting personalized pitches.\n{OUTPUT_FORMAT}"),
        ("human", """Generate tailored arguments for this prospect.

PROSPECT PROFILE:
{prospect_analysis}

AVAILABLE ARGUMENTS:
{sales_arguments}

CONTEXT: {industry} | {prospect_title} | {outcome}

Generate 5 personalized arguments with: Message (1-2 sentences) + Why it works for them."""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "prospect_analysis": prospect_analysis,
        "sales_arguments": SALES_ARGUMENTS,
        "industry": metadata["prospect"]["industry"],
        "prospect_title": metadata["prospect"]["title"],
        "outcome": metadata["outcome"],
    })
    return response.content


def ask_about_call(data: dict, question: str, chat_history: list[dict]) -> str:
    """Chat interface for asking questions about a call."""
    metadata = data["metadata"]

    system_message = f"""You are a sales coaching assistant. Be concise and practical.
Always start with 2-3 key points, then elaborate if needed.

CALL: {metadata['sdr']['name']} called {metadata['prospect']['contact_name']} ({metadata['prospect']['title']}) at {metadata['prospect']['company']}
INDUSTRY: {metadata['prospect']['industry']}
OUTCOME: {metadata['outcome']}

SALES ARGUMENTS:
{SALES_ARGUMENTS}

TRANSCRIPT:
{format_transcript(data['transcript'])}"""

    messages = [("system", system_message)]
    for msg in chat_history:
        messages.append((msg["role"], msg["content"]))
    messages.append(("human", question))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | get_llm(0.4)
    response = chain.invoke({})
    return response.content


# =============================================================================
# STRATEGY CONSULTANT FUNCTIONS - Market research, arguments, A/B testing
# =============================================================================


def analyze_ab_test(all_calls: list[dict]) -> str:
    """Compare effectiveness of pitch variants."""
    calls_text = "\n".join(format_call_summary(call) for call in all_calls)
    pitch_a = PITCH_VARIANTS["A"]
    pitch_b = PITCH_VARIANTS["B"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales analytics expert specializing in A/B testing.\n{OUTPUT_FORMAT}"),
        ("human", """Compare pitch effectiveness across these calls.

PITCH A - "{pitch_a_name}":
{pitch_a_elements}

PITCH B - "{pitch_b_name}":
{pitch_b_elements}

CALLS:
{calls}

Analyze: Classification per call, which pitch worked better, winning elements, recommendation."""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "pitch_a_name": pitch_a["name"],
        "pitch_a_elements": "\n".join(f"- {e}" for e in pitch_a["elements"]),
        "pitch_b_name": pitch_b["name"],
        "pitch_b_elements": "\n".join(f"- {e}" for e in pitch_b["elements"]),
        "calls": calls_text,
    })
    return response.content


def generate_target_markets(product_description: str, geography: str = "Finland") -> str:
    """AI-generated target market suggestions based on product and geography."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a B2B market strategist identifying ideal target markets.
You have deep knowledge of {geography} business landscape, local industries, and regional characteristics.\n{OUTPUT_FORMAT}"""),
        ("human", """Based on this product/service, identify the best target markets in {geography}.

PRODUCT/SERVICE:
{product_description}

GEOGRAPHIC FOCUS:
{geography}

Identify 5 target market segments ranked by potential FOR THIS GEOGRAPHY. For each include:
- Industry/Vertical specific to {geography}
- Company size range typical in {geography}
- Revenue range in local currency
- Regional concentration (which cities/areas)
- Key decision maker titles (use local language titles if relevant)
- Why they need this (geography-specific pain points)
- Market potential: High/Medium/Low
- Sales cycle length
- Local competitors to be aware of
- Cultural considerations for sales approach"""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "product_description": product_description,
        "geography": geography,
    })
    return response.content


def analyze_target_market(
    product_description: str,
    target_market: str,
    geography: str = "Finland",
) -> str:
    """Deep analysis of a specific target market in a geography."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a market research expert with deep knowledge of {geography}.
You understand local business culture, regulations, and market dynamics.\n{OUTPUT_FORMAT}"""),
        ("human", """Analyze this target market for the given product in {geography}.

PRODUCT/SERVICE:
{product_description}

TARGET MARKET:
{target_market}

GEOGRAPHIC FOCUS:
{geography}

Analyze:

MARKET SIZE & OPPORTUNITY IN {geography}:
- Estimated number of companies in this segment
- Market value in local currency
- Growth trends specific to {geography}
- Regional hotspots (cities/areas with highest concentration)

IDEAL CUSTOMER PROFILE (ICP) FOR {geography}:
- Company characteristics typical in {geography}
- Local technology adoption patterns
- Regulatory/compliance factors in {geography}
- Firmographic criteria

BUYER PERSONAS (with local context):
- Decision maker titles (use Finnish/Swedish/local titles)
- Local business culture considerations
- Communication preferences in {geography}
- Typical objections in this market

COMPETITIVE LANDSCAPE IN {geography}:
- Local competitors
- International players active in {geography}
- Market positioning opportunities

ENTRY STRATEGY FOR {geography}:
- Best channels (local events, associations, media)
- Partnership opportunities
- Language considerations
- Local proof points needed"""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "product_description": product_description,
        "target_market": target_market,
        "geography": geography,
    })
    return response.content


def conduct_market_research(
    product_description: str,
    target_market: str,
    geography: str,
    research_focus: str,
) -> str:
    """AI-assisted market research for strategy development in specific geography."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior market research analyst and B2B strategy consultant.
You have expert knowledge of the {geography} market, including local business practices, 
key industry associations, regulatory environment, and cultural nuances.\n{OUTPUT_FORMAT}"""),
        ("human", """Conduct market research analysis for {geography}.

PRODUCT/SERVICE:
{product_description}

TARGET MARKET:
{target_market}

GEOGRAPHIC FOCUS:
{geography}

RESEARCH FOCUS:
{research_focus}

Provide {geography}-specific analysis:

MARKET OVERVIEW:
- Market size and growth trends in {geography}
- Key players and competitive landscape (local and international)
- Industry-specific dynamics in {geography}
- Regulatory environment

TARGET CUSTOMER PROFILE:
- Ideal customer characteristics in {geography}
- Decision makers and buying process (local business culture)
- Common pain points in {geography}
- Buying triggers and timing (seasonal, budget cycles)

POSITIONING FOR {geography}:
- Unique value propositions that resonate locally
- Differentiation strategies vs local competitors
- Trust-building factors important in {geography}

KEY MESSAGES:
- Primary value proposition (adapted for local culture)
- Local proof points and references
- Role-specific messaging in local context

OBJECTION MAPPING FOR {geography}:
- Common objections in this market
- Culturally appropriate responses
- Local proof points to prepare

GO-TO-MARKET IN {geography}:
- Best channels (local trade shows, associations, media)
- Partnership opportunities
- Local reference customer strategy
- Language and localization needs"""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "product_description": product_description,
        "target_market": target_market,
        "geography": geography,
        "research_focus": research_focus,
    })
    return response.content


def generate_sales_arguments(
    market_research: str,
    target_segment: str,
    tone: str,
) -> str:
    """Generate sales arguments based on market research."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales messaging expert creating compelling arguments.\n{OUTPUT_FORMAT}"),
        ("human", """Create sales arguments based on this research.

MARKET RESEARCH:
{market_research}

TARGET SEGMENT: {target_segment}
TONE: {tone}

Generate 6 sales arguments. For each: Argument name with hook, Details and proof, When to use."""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "market_research": market_research,
        "target_segment": target_segment,
        "tone": tone,
    })
    return response.content


# =============================================================================
# FEEDBACK LOOP - Connect call data to strategy
# =============================================================================


def analyze_segment_performance(all_calls: list[dict]) -> dict:
    """Calculate performance metrics by segment from actual call data."""
    segments = {}

    for call in all_calls:
        meta = call["metadata"]
        industry = meta["prospect"]["industry"]
        outcome = meta["outcome"]

        if industry not in segments:
            segments[industry] = {
                "total": 0,
                "booked": 0,
                "outcomes": {},
                "avg_duration": 0,
                "total_duration": 0,
                "titles": {},
                "companies": [],
            }

        seg = segments[industry]
        seg["total"] += 1
        seg["total_duration"] += meta["duration_seconds"]
        seg["avg_duration"] = seg["total_duration"] // seg["total"]

        if outcome == "meeting_booked":
            seg["booked"] += 1

        seg["outcomes"][outcome] = seg["outcomes"].get(outcome, 0) + 1

        title = meta["prospect"]["title"]
        seg["titles"][title] = seg["titles"].get(title, 0) + 1

        seg["companies"].append(meta["prospect"]["company"])

    for seg in segments.values():
        seg["conversion_rate"] = (seg["booked"] / seg["total"] * 100) if seg["total"] > 0 else 0

    return segments


def generate_strategy_feedback(all_calls: list[dict]) -> str:
    """Generate strategic feedback based on actual call performance."""
    segments = analyze_segment_performance(all_calls)
    calls_summary = "\n".join(format_call_brief(c) for c in all_calls)

    segment_summary = ""
    for name, data in segments.items():
        segment_summary += f"""
{name}:
- Calls: {data['total']}, Booked: {data['booked']}, Rate: {data['conversion_rate']:.1f}%
- Avg Duration: {data['avg_duration']}s
- Titles: {', '.join(f"{t} ({c})" for t, c in data['titles'].items())}
- Outcomes: {', '.join(f"{o} ({c})" for o, c in data['outcomes'].items())}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a sales strategy analyst providing data-driven feedback.\n{OUTPUT_FORMAT}"""),
        ("human", """Analyze call performance and provide strategic feedback.

PERFORMANCE BY SEGMENT:
{segment_summary}

ALL CALLS:
{calls_summary}

CURRENT SALES ARGUMENTS:
{sales_arguments}

Analyze: Segment performance (best/worst and why), Argument effectiveness, Target market recommendations, Strategy adjustments."""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "segment_summary": segment_summary,
        "calls_summary": calls_summary,
        "sales_arguments": SALES_ARGUMENTS,
    })
    return response.content


def analyze_argument_usage(all_calls: list[dict]) -> str:
    """Analyze which sales arguments are being used and their effectiveness."""
    calls_text = "\n".join(format_call_summary(call) for call in all_calls)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a sales effectiveness analyst.\n{OUTPUT_FORMAT}"""),
        ("human", """Analyze how sales arguments are used across these calls.

DEFINED SALES ARGUMENTS:
{sales_arguments}

CALL TRANSCRIPTS:
{calls}

Analyze: Usage matrix per argument (frequency, effectiveness), Argument gaps, Top performing combinations, Recommendations."""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "sales_arguments": SALES_ARGUMENTS,
        "calls": calls_text,
    })
    return response.content


def generate_refined_arguments(
    all_calls: list[dict],
    current_arguments: str,
) -> str:
    """Generate refined arguments based on what's actually working in calls."""
    successful_calls = [c for c in all_calls if c["metadata"]["outcome"] == "meeting_booked"]
    failed_calls = [c for c in all_calls if c["metadata"]["outcome"] != "meeting_booked"]

    success_text = "\n".join(format_call_summary(c) for c in successful_calls) if successful_calls else "None"
    fail_text = "\n".join(format_call_summary(c) for c in failed_calls[:3]) if failed_calls else "None"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a sales messaging expert refining arguments based on real data.\n{OUTPUT_FORMAT}"""),
        ("human", """Refine sales arguments based on actual call performance.

CURRENT ARGUMENTS:
{current_arguments}

SUCCESSFUL CALLS (these worked):
{success_text}

SAMPLE FAILED CALLS (these didn't work):
{fail_text}

Generate 6 REFINED arguments. For each: Argument name with hook, Exact phrasing from successful calls, Proof point, When to use, Which call it came from."""),
    ])

    chain = prompt | get_llm(0.4)
    response = chain.invoke({
        "current_arguments": current_arguments,
        "success_text": success_text,
        "fail_text": fail_text,
    })
    return response.content


def ask_strategy_question(
    question: str,
    context: str,
    chat_history: list[dict],
) -> str:
    """Strategy consultant chat interface."""
    system_message = f"""You are a senior sales and marketing strategy consultant.
Always start your response with 2-3 key points, then provide detailed explanation.

CONTEXT:
{context}

CURRENT SALES ARGUMENTS:
{SALES_ARGUMENTS}"""

    messages = [("system", system_message)]
    for msg in chat_history:
        messages.append((msg["role"], msg["content"]))
    messages.append(("human", question))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | get_llm(0.4)
    response = chain.invoke({})
    return response.content


# =============================================================================
# ACCOUNT MANAGEMENT FUNCTIONS - Reporting, metrics
# =============================================================================


def generate_performance_report(all_calls: list[dict]) -> str:
    """Generate performance report for account management."""
    total = len(all_calls)
    booked = sum(1 for c in all_calls if c["metadata"]["outcome"] == "meeting_booked")
    rate = (booked / total * 100) if total > 0 else 0
    avg_duration = sum(c["metadata"]["duration_seconds"] for c in all_calls) // total if total else 0

    industries = {}
    for call in all_calls:
        ind = call["metadata"]["prospect"]["industry"]
        if ind not in industries:
            industries[ind] = {"total": 0, "booked": 0}
        industries[ind]["total"] += 1
        if call["metadata"]["outcome"] == "meeting_booked":
            industries[ind]["booked"] += 1

    calls_summary = "\n".join(format_call_brief(c) for c in all_calls)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales operations analyst creating executive reports.\n{OUTPUT_FORMAT}"),
        ("human", """Create a performance report.

METRICS:
- Total Calls: {total}
- Meetings Booked: {booked}
- Conversion Rate: {rate:.1f}%
- Avg Duration: {avg_duration}s

BY INDUSTRY:
{industries}

CALLS:
{calls_summary}

Provide: Executive summary, Key wins, Areas for improvement, Recommendations, Forecast."""),
    ])

    chain = prompt | get_llm()
    industries_text = "\n".join(
        f"- {k}: {v['booked']}/{v['total']} ({v['booked']/v['total']*100:.0f}%)"
        for k, v in industries.items()
    )
    response = chain.invoke({
        "total": total,
        "booked": booked,
        "rate": rate,
        "avg_duration": avg_duration,
        "industries": industries_text,
        "calls_summary": calls_summary,
    })
    return response.content


def analyze_patterns(all_calls: list[dict]) -> str:
    """Analyze success/failure patterns across calls."""
    successful = []
    unsuccessful = []

    for call in all_calls:
        summary = format_call_summary(call)
        if call["metadata"]["outcome"] == "meeting_booked":
            successful.append(summary)
        else:
            unsuccessful.append(summary)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales analytics expert. Be data-driven and actionable.\n{OUTPUT_FORMAT}"),
        ("human", """Analyze patterns across these calls.

SUCCESSFUL (meeting booked):
{successful_calls}

UNSUCCESSFUL:
{unsuccessful_calls}

Analyze: Why calls were booked (3 factors), Why calls were not booked (3 factors), Top 3 recommendations."""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "successful_calls": "\n".join(successful) if successful else "None",
        "unsuccessful_calls": "\n".join(unsuccessful) if unsuccessful else "None",
    })
    return response.content


# =============================================================================
# CLIENT FUNCTIONS - Results visibility, outcome analysis
# =============================================================================


def generate_client_report(all_calls: list[dict], client_name: str) -> str:
    """Generate client-facing report."""
    total = len(all_calls)
    booked = sum(1 for c in all_calls if c["metadata"]["outcome"] == "meeting_booked")
    rate = (booked / total * 100) if total > 0 else 0

    calls_summary = "\n".join(
        f"- {c['metadata']['prospect']['company']}: {c['metadata']['outcome'].replace('_', ' ')}"
        for c in all_calls
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are preparing a professional client report. Be clear and results-focused.\n{OUTPUT_FORMAT}"),
        ("human", """Create a client report for {client_name}.

CAMPAIGN RESULTS:
- Calls Made: {total}
- Meetings Booked: {booked}
- Conversion Rate: {rate:.1f}%

CALL OUTCOMES:
{calls_summary}

Provide: Results summary, Meetings secured, Pipeline value, Insights, Next steps."""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "client_name": client_name,
        "total": total,
        "booked": booked,
        "rate": rate,
        "calls_summary": calls_summary,
    })
    return response.content


def explain_outcome(data: dict) -> str:
    """Explain why a specific call succeeded or failed."""
    metadata = data["metadata"]
    outcome = metadata["outcome"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a sales analyst explaining call outcomes clearly.\n{OUTPUT_FORMAT}"),
        ("human", """Explain why this call resulted in: {outcome}

PROSPECT: {prospect_name} ({prospect_title}) at {prospect_company}
INDUSTRY: {industry}
DURATION: {duration}s

TRANSCRIPT:
{transcript}

Analyze: Outcome explanation (why this result), Key moments (turning points), Prospect signals, Lessons to learn."""),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({
        "outcome": outcome.replace("_", " "),
        "prospect_name": metadata["prospect"]["contact_name"],
        "prospect_title": metadata["prospect"]["title"],
        "prospect_company": metadata["prospect"]["company"],
        "industry": metadata["prospect"]["industry"],
        "duration": metadata["duration_seconds"],
        "transcript": format_transcript(data["transcript"]),
    })
    return response.content
