import json
from pathlib import Path

import streamlit as st

from agent import (
    PITCH_VARIANTS,
    SALES_ARGUMENTS,
    analyze_ab_test,
    analyze_argument_usage,
    analyze_call,
    analyze_patterns,
    analyze_prospect,
    analyze_segment_performance,
    analyze_target_market,
    ask_about_call,
    ask_strategy_question,
    conduct_market_research,
    explain_outcome,
    generate_client_report,
    generate_performance_report,
    generate_personalized_arguments,
    generate_refined_arguments,
    generate_sales_arguments,
    generate_strategy_feedback,
    generate_target_markets,
    get_current_provider_info,
    prepare_for_call,
)

st.set_page_config(
    page_title="AI Marketing Demo",
    layout="wide",
)

TRANSCRIPTS_DIR = Path("test-data/transcripts")
PROSPECTS_FILE = Path("test-data/prospects.json")


# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data
def load_transcript(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


@st.cache_data
def load_prospects() -> list[dict]:
    if PROSPECTS_FILE.exists():
        with PROSPECTS_FILE.open() as f:
            data = json.load(f)
            return data.get("prospects", [])
    return []


@st.cache_data
def load_all_transcripts() -> list[dict]:
    transcripts = []
    for path in TRANSCRIPTS_DIR.glob("*.json"):
        with path.open() as f:
            transcripts.append(json.load(f))
    return transcripts


def get_json_files() -> list[Path]:
    return list(TRANSCRIPTS_DIR.glob("*.json"))


# =============================================================================
# SHARED COMPONENTS
# =============================================================================


def render_call_list(calls: list[dict], show_details: bool = True) -> None:
    for call in calls:
        meta = call["metadata"]
        outcome = meta["outcome"]
        is_success = outcome == "meeting_booked"
        color = "#7dd87d" if is_success else "#ff6b6b"
        label = "[BOOKED]" if is_success else f"[{outcome.upper().replace('_', ' ')}]"

        details = ""
        if show_details:
            details = f" — {meta['prospect']['industry']} — {meta['duration_seconds']}s"

        st.markdown(
            f"<div style='padding: 0.4rem 0; border-bottom: 1px solid #333;'>"
            f"<span style='color: {color}; font-weight: 600;'>{label}</span> "
            f"{meta['prospect']['company']}{details}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_metrics(calls: list[dict]) -> None:
    total = len(calls)
    booked = sum(1 for c in calls if c["metadata"]["outcome"] == "meeting_booked")
    rate = (booked / total * 100) if total > 0 else 0
    avg_duration = sum(c["metadata"]["duration_seconds"] for c in calls) // total if total else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", total)
    with col2:
        st.metric("Meetings Booked", booked)
    with col3:
        st.metric("Conversion Rate", f"{rate:.1f}%")
    with col4:
        st.metric("Avg Duration", f"{avg_duration}s")


def render_transcript(transcript: list[dict]) -> None:
    for entry in transcript:
        speaker = entry["speaker"]
        label = "SDR" if speaker == "sdr" else "PROSPECT"
        color = "#6b9eff" if speaker == "sdr" else "#7dd87d"
        st.markdown(
            f"<div style='margin-bottom: 0.5rem;'>"
            f"<span style='color: #888; font-size: 0.75rem;'>{entry['timestamp']}</span> "
            f"<span style='color: {color}; font-weight: 600; font-size: 0.8rem;'>{label}</span><br/>"
            f"<span style='color: #eee;'>{entry['text']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_chat(
    chat_key: str,
    chat_fn,
    data=None,
    context: str = "",
    placeholder: str = "Ask a question...",
) -> None:
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        role = "You" if msg["role"] == "human" else "AI"
        color = "#6b9eff" if msg["role"] == "human" else "#7dd87d"
        st.markdown(
            f"<div style='margin-bottom: 0.75rem;'>"
            f"<span style='color: {color}; font-weight: 600;'>{role}:</span> "
            f"{msg['content']}</div>",
            unsafe_allow_html=True,
        )

    question = st.text_input("Question", placeholder=placeholder, label_visibility="collapsed", key=f"{chat_key}_input")

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("Ask", key=f"{chat_key}_ask")
    with col2:
        if st.session_state[chat_key]:
            if st.button("Clear", key=f"{chat_key}_clear"):
                st.session_state[chat_key] = []
                st.rerun()

    if ask and question:
        with st.spinner("Thinking..."):
            if data is not None:
                answer = chat_fn(data, question, st.session_state[chat_key])
            else:
                answer = chat_fn(question, context, st.session_state[chat_key])
        st.session_state[chat_key].append({"role": "human", "content": question})
        st.session_state[chat_key].append({"role": "assistant", "content": answer})
        st.rerun()


# =============================================================================
# SDR VIEW
# =============================================================================


def render_sdr_view() -> None:
    st.header("SDR Dashboard")

    json_files = get_json_files()
    if not json_files:
        st.error("No transcript files found.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Call Prep", "My Calls", "Call Analysis", "Ask Coach"])

    all_calls = load_all_transcripts()

    with tab1:
        prospects = load_prospects()

        if not prospects:
            st.info("No prospects loaded. Add prospects to test-data/prospects.json")
        else:
            # Sidebar: Prospect list
            col_list, col_detail = st.columns([1, 2])

            with col_list:
                st.caption(f"UPCOMING CALLS ({len(prospects)})")

                # Group by date
                by_date = {}
                for p in prospects:
                    date = p.get("scheduled_date", "Unscheduled")
                    if date not in by_date:
                        by_date[date] = []
                    by_date[date].append(p)

                # Initialize selected prospect
                if "selected_prospect_idx" not in st.session_state:
                    st.session_state["selected_prospect_idx"] = 0

                # Render prospect list
                idx = 0
                for date, date_prospects in by_date.items():
                    st.markdown(f"**{date}**")
                    for p in date_prospects:
                        status_icon = "🔄" if p.get("status") == "callback" else "📞"
                        is_selected = idx == st.session_state["selected_prospect_idx"]
                        
                        btn_label = f"{p.get('scheduled_time', '')} {p['company']}"
                        if is_selected:
                            btn_label = f"→ {btn_label}"

                        if st.button(
                            btn_label,
                            key=f"prospect_btn_{idx}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state["selected_prospect_idx"] = idx
                            if "call_prep" in st.session_state:
                                del st.session_state["call_prep"]
                            st.rerun()
                        idx += 1

            with col_detail:
                # Get selected prospect
                selected_idx = st.session_state["selected_prospect_idx"]
                if selected_idx < len(prospects):
                    p = prospects[selected_idx]

                    # Header
                    st.subheader(f"{p['company']}")
                    
                    # Status badge
                    if p.get("status") == "callback":
                        st.warning(f"CALLBACK: {p.get('callback_reason', 'Requested callback')}")

                    # Prospect details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Contact:** {p['contact_name']}")
                        st.markdown(f"**Title:** {p['title']}")
                        st.markdown(f"**Phone:** {p.get('phone', 'N/A')}")
                    with col2:
                        st.markdown(f"**Industry:** {p['industry']}")
                        st.markdown(f"**Employees:** {p.get('employee_count', 'N/A')}")
                        st.markdown(f"**Time:** {p.get('scheduled_date', '')} {p.get('scheduled_time', '')}")

                    if p.get("notes"):
                        st.caption("NOTES")
                        st.info(p["notes"])

                    st.divider()

                    # Prepare for call button
                    if st.button("🎯 Prepare for This Call", key="sdr_prep_selected", use_container_width=True):
                        prospect_info = f"""Company: {p['company']}
Contact: {p['contact_name']}
Title: {p['title']}
Industry: {p['industry']}
Employees: {p.get('employee_count', 'Unknown')}
Phone: {p.get('phone', 'N/A')}
Email: {p.get('email', 'N/A')}"""
                        if p.get("notes"):
                            prospect_info += f"\nNotes: {p['notes']}"
                        if p.get("callback_reason"):
                            prospect_info += f"\nCallback Reason: {p['callback_reason']}"

                        with st.spinner("Preparing call guidance..."):
                            guidance = prepare_for_call(prospect_info)
                            st.session_state["call_prep"] = guidance
                        st.rerun()

                    # Show call prep if available
                    if "call_prep" in st.session_state:
                        st.markdown(st.session_state["call_prep"])

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Next Prospect →", key="sdr_next"):
                                st.session_state["selected_prospect_idx"] = min(
                                    selected_idx + 1, len(prospects) - 1
                                )
                                del st.session_state["call_prep"]
                                st.rerun()
                        with col2:
                            if st.button("Clear Prep", key="sdr_prep_clear"):
                                del st.session_state["call_prep"]
                                st.rerun()

        # Manual entry expander
        with st.expander("Manual Entry (paste or form)", expanded=False):
            input_method = st.radio(
                "Input Method",
                ["Paste Info", "Structured Form"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if input_method == "Paste Info":
                prospect_info = st.text_area(
                    "Prospect Information",
                    placeholder="""Paste any prospect info here, e.g.:

Company: Rakennusliike ABC Oy
Contact: Matti Virtanen
Title: Toimitusjohtaja
Industry: Construction
Employees: 45""",
                    height=150,
                    key="sdr_prep_paste",
                )

                if st.button("Prepare for Call", key="sdr_prep_btn"):
                    if prospect_info:
                        with st.spinner("Preparing call guidance..."):
                            guidance = prepare_for_call(prospect_info)
                            st.session_state["call_prep"] = guidance
                        st.rerun()
                    else:
                        st.warning("Please paste prospect information first.")

            else:
                col1, col2 = st.columns(2)
                with col1:
                    company = st.text_input("Company", key="sdr_prep_company")
                    contact = st.text_input("Contact Name", key="sdr_prep_contact")
                    title = st.text_input("Title", key="sdr_prep_title")

                with col2:
                    industry = st.selectbox(
                        "Industry",
                        ["Construction", "IT Services", "Healthcare", "Logistics", "Manufacturing", "Retail", "Professional Services", "Other"],
                        key="sdr_prep_industry",
                    )
                    employees = st.number_input("Employees", min_value=1, value=50, key="sdr_prep_emp")
                    notes = st.text_area("Notes", height=68, key="sdr_prep_notes")

                if st.button("Prepare for Call", key="sdr_prep_form_btn"):
                    if company and contact and title:
                        prospect_info = f"""Company: {company}
Contact: {contact}
Title: {title}
Industry: {industry}
Employees: {employees}"""
                        if notes:
                            prospect_info += f"\nNotes: {notes}"

                        with st.spinner("Preparing call guidance..."):
                            guidance = prepare_for_call(prospect_info)
                            st.session_state["call_prep"] = guidance
                        st.rerun()
                    else:
                        st.warning("Please fill in Company, Contact, and Title.")

    with tab2:
        render_metrics(all_calls)
        st.divider()
        render_call_list(all_calls)

    with tab3:
        selected = st.selectbox(
            "Select Call",
            json_files,
            format_func=lambda p: p.stem,
            key="sdr_call_select",
        )

        if selected:
            data = load_transcript(selected)
            meta = data["metadata"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{meta['duration_seconds']}s")
            with col2:
                st.metric("Outcome", meta["outcome"].replace("_", " ").title())
            with col3:
                st.metric("Industry", meta["prospect"]["industry"])

            subtab1, subtab2, subtab3 = st.tabs(["Transcript", "AI Feedback", "Prospect Profile"])

            with subtab1:
                render_transcript(data["transcript"])

            with subtab2:
                with st.expander("Sales Arguments", expanded=False):
                    st.markdown(SALES_ARGUMENTS)

                if st.button("Get AI Feedback", key="sdr_feedback"):
                    with st.spinner("Analyzing..."):
                        feedback = analyze_call(data)
                    st.markdown(feedback)

            with subtab3:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Analyze Prospect", key="sdr_prospect"):
                        with st.spinner("Analyzing..."):
                            profile = analyze_prospect(data)
                            st.session_state["sdr_profile"] = profile

                with col2:
                    if st.button("Generate Tailored Arguments", key="sdr_args", disabled="sdr_profile" not in st.session_state):
                        with st.spinner("Generating..."):
                            args = generate_personalized_arguments(data, st.session_state["sdr_profile"])
                            st.session_state["sdr_args"] = args

                if "sdr_profile" in st.session_state:
                    st.caption("PROSPECT PROFILE")
                    st.markdown(st.session_state["sdr_profile"])

                if "sdr_args" in st.session_state:
                    st.divider()
                    st.caption("TAILORED ARGUMENTS")
                    st.markdown(st.session_state["sdr_args"])

    with tab4:
        selected_chat = st.selectbox(
            "Select Call to Discuss",
            json_files,
            format_func=lambda p: p.stem,
            key="sdr_chat_select",
        )

        if selected_chat:
            data = load_transcript(selected_chat)
            render_chat(
                "sdr_chat",
                ask_about_call,
                data=data,
                placeholder="e.g., How could I have handled the objection better?",
            )


# =============================================================================
# STRATEGY CONSULTANT VIEW
# =============================================================================


def render_strategy_view() -> None:
    st.header("Strategy Consultant")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Target Market", "Market Research", "Feedback Loop", "Sales Arguments", "A/B Testing", "Strategy Chat"]
    )

    all_calls = load_all_transcripts()

    with tab1:
        st.subheader("Target Market Definition")

        col_prod, col_geo = st.columns([2, 1])

        with col_prod:
            product = st.text_area(
                "Product/Service Description",
                placeholder="Describe what you're selling, key features, and value proposition...",
                height=120,
                key="tm_product",
            )

        with col_geo:
            geography = st.selectbox(
                "Geographic Focus",
                [
                    "Finland",
                    "Southern Finland (Uusimaa, Helsinki region)",
                    "Western Finland (Tampere, Turku region)",
                    "Northern Finland (Oulu region)",
                    "Eastern Finland",
                    "Sweden",
                    "Stockholm region",
                    "Nordics (Finland, Sweden, Norway, Denmark)",
                    "Baltics (Estonia, Latvia, Lithuania)",
                    "DACH (Germany, Austria, Switzerland)",
                    "Benelux",
                    "UK & Ireland",
                    "Europe",
                    "Global",
                ],
                key="tm_geography",
            )
            st.session_state["selected_geography"] = geography

        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"AI-GENERATED TARGET MARKETS IN {geography.upper()}")
            if st.button("Generate Target Markets", key="tm_generate"):
                if product:
                    with st.spinner(f"Analyzing product for {geography}..."):
                        targets = generate_target_markets(product, geography)
                        st.session_state["generated_targets"] = targets
                else:
                    st.warning("Please describe your product first.")

            if "generated_targets" in st.session_state:
                st.markdown(st.session_state["generated_targets"])

        with col2:
            st.caption("SELECT OR DEFINE TARGET MARKET")

            target_options = [
                "Custom (define below)",
                "SMB Construction (10-100 employees)",
                "Mid-market IT Services (50-500 employees)",
                "Healthcare Clinics (5-50 employees)",
                "Professional Services (20-200 employees)",
                "Manufacturing SMB (25-250 employees)",
            ]

            selected_target = st.selectbox("Quick Select", target_options, key="tm_select")

            if selected_target == "Custom (define below)":
                custom_target = st.text_area(
                    "Define Target Market",
                    placeholder="e.g., Construction companies in Finland with 20-100 employees...",
                    height=100,
                    key="tm_custom",
                )
                target_market = custom_target
            else:
                target_market = selected_target

            if target_market and target_market != "Custom (define below)":
                st.session_state["selected_target_market"] = target_market

                if st.button("Analyze This Market", key="tm_analyze"):
                    if product:
                        with st.spinner(f"Analyzing target market in {geography}..."):
                            analysis = analyze_target_market(product, target_market, geography)
                            st.session_state["target_market_analysis"] = analysis
                    else:
                        st.warning("Please describe your product first.")

        if "target_market_analysis" in st.session_state:
            st.divider()
            st.caption("TARGET MARKET ANALYSIS")
            st.markdown(st.session_state["target_market_analysis"])

    with tab2:
        st.subheader("AI Market Research")

        col1, col2 = st.columns([2, 1])

        with col1:
            product_research = st.text_area(
                "Product/Service Description",
                placeholder="Describe what you're selling...",
                height=100,
                key="mr_product",
                value=st.session_state.get("tm_product", ""),
            )

        with col2:
            geography_research = st.selectbox(
                "Geographic Focus",
                [
                    "Finland",
                    "Southern Finland (Uusimaa, Helsinki region)",
                    "Western Finland (Tampere, Turku region)",
                    "Northern Finland (Oulu region)",
                    "Eastern Finland",
                    "Sweden",
                    "Stockholm region",
                    "Nordics (Finland, Sweden, Norway, Denmark)",
                    "Baltics (Estonia, Latvia, Lithuania)",
                    "DACH (Germany, Austria, Switzerland)",
                    "Benelux",
                    "UK & Ireland",
                    "Europe",
                    "Global",
                ],
                key="mr_geography",
                index=0 if "selected_geography" not in st.session_state else [
                    "Finland", "Southern Finland (Uusimaa, Helsinki region)", 
                    "Western Finland (Tampere, Turku region)", "Northern Finland (Oulu region)",
                    "Eastern Finland", "Sweden", "Stockholm region",
                    "Nordics (Finland, Sweden, Norway, Denmark)", "Baltics (Estonia, Latvia, Lithuania)",
                    "DACH (Germany, Austria, Switzerland)", "Benelux", "UK & Ireland", "Europe", "Global"
                ].index(st.session_state.get("selected_geography", "Finland")) if st.session_state.get("selected_geography", "Finland") in [
                    "Finland", "Southern Finland (Uusimaa, Helsinki region)", 
                    "Western Finland (Tampere, Turku region)", "Northern Finland (Oulu region)",
                    "Eastern Finland", "Sweden", "Stockholm region",
                    "Nordics (Finland, Sweden, Norway, Denmark)", "Baltics (Estonia, Latvia, Lithuania)",
                    "DACH (Germany, Austria, Switzerland)", "Benelux", "UK & Ireland", "Europe", "Global"
                ] else 0,
            )

        target_market_research = st.text_input(
            "Target Market",
            placeholder="e.g., Construction SMBs",
            key="mr_target",
            value=st.session_state.get("selected_target_market", ""),
        )

        focus = st.selectbox(
            "Research Focus",
            [
                "General Overview",
                "Competitive Analysis",
                "Buyer Personas",
                "Objection Handling",
                "Go-to-Market Strategy",
                "Pricing Strategy",
                "Local Partnerships",
                "Regulatory & Compliance",
            ],
        )

        if st.button("Conduct Research", key="strategy_research"):
            if product_research and target_market_research:
                with st.spinner(f"Conducting market research for {geography_research}..."):
                    research = conduct_market_research(
                        product_research, target_market_research, geography_research, focus
                    )
                    st.session_state["market_research"] = research
                st.markdown(research)
            else:
                st.warning("Please fill in product description and target market.")

        if "market_research" in st.session_state:
            st.divider()
            st.markdown(st.session_state["market_research"])

    with tab3:
        st.subheader("Feedback Loop - Learn from Calls")

        # Show segment performance from actual data
        segments = analyze_segment_performance(all_calls)

        st.caption("SEGMENT PERFORMANCE (FROM ACTUAL CALLS)")
        for name, data in segments.items():
            rate = data["conversion_rate"]
            color = "#7dd87d" if rate >= 50 else "#ffaa00" if rate >= 25 else "#ff6b6b"
            st.markdown(
                f"<div style='padding: 0.4rem 0; border-bottom: 1px solid #333;'>"
                f"<span style='color: {color}; font-weight: 600;'>{rate:.0f}%</span> "
                f"**{name}** — {data['booked']}/{data['total']} booked, "
                f"avg {data['avg_duration']}s"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.caption("STRATEGIC FEEDBACK")
            if st.button("Generate Strategy Feedback", key="fb_strategy"):
                with st.spinner("Analyzing call data for strategic insights..."):
                    feedback = generate_strategy_feedback(all_calls)
                    st.session_state["strategy_feedback"] = feedback

            if "strategy_feedback" in st.session_state:
                st.markdown(st.session_state["strategy_feedback"])

        with col2:
            st.caption("ARGUMENT EFFECTIVENESS")
            if st.button("Analyze Argument Usage", key="fb_arguments"):
                with st.spinner("Analyzing how arguments are used in calls..."):
                    arg_analysis = analyze_argument_usage(all_calls)
                    st.session_state["argument_analysis"] = arg_analysis

            if "argument_analysis" in st.session_state:
                st.markdown(st.session_state["argument_analysis"])

        st.divider()

        st.caption("REFINE ARGUMENTS FROM CALL DATA")
        if st.button("Generate Refined Arguments", key="fb_refine"):
            with st.spinner("Learning from successful calls..."):
                refined = generate_refined_arguments(all_calls, SALES_ARGUMENTS)
                st.session_state["refined_arguments"] = refined

        if "refined_arguments" in st.session_state:
            st.markdown(st.session_state["refined_arguments"])

    with tab4:
        st.subheader("Generate Sales Arguments")

        st.caption("CURRENT ARGUMENTS")
        st.markdown(SALES_ARGUMENTS)

        st.divider()

        if "market_research" in st.session_state:
            segment = st.text_input("Target Segment", placeholder="e.g., CFOs in construction companies")
            tone = st.selectbox("Tone", ["Professional", "Conversational", "Urgent", "Consultative"])

            if st.button("Generate New Arguments", key="strategy_gen_args"):
                with st.spinner("Generating arguments..."):
                    new_args = generate_sales_arguments(
                        st.session_state["market_research"],
                        segment,
                        tone,
                    )
                st.markdown(new_args)
        else:
            st.info("Conduct market research first to generate customized arguments.")

    with tab5:
        st.subheader("Pitch A/B Testing")

        col1, col2 = st.columns(2)
        with col1:
            pitch_a = PITCH_VARIANTS["A"]
            st.caption(f"PITCH A: {pitch_a['name'].upper()}")
            st.write(pitch_a["description"])
            for e in pitch_a["elements"]:
                st.write(f"· {e}")

        with col2:
            pitch_b = PITCH_VARIANTS["B"]
            st.caption(f"PITCH B: {pitch_b['name'].upper()}")
            st.write(pitch_b["description"])
            for e in pitch_b["elements"]:
                st.write(f"· {e}")

        st.divider()
        st.caption(f"ANALYZING {len(all_calls)} CALLS")
        render_call_list(all_calls, show_details=False)

        st.divider()
        if st.button("Run A/B Analysis", key="strategy_ab"):
            with st.spinner("Analyzing pitch effectiveness..."):
                analysis = analyze_ab_test(all_calls)
            st.markdown(analysis)

    with tab6:
        st.subheader("Strategy Assistant")

        context = st.text_area(
            "Context",
            placeholder="Describe your current situation, challenges, or what you're trying to achieve...",
            height=100,
        )

        render_chat(
            "strategy_chat",
            ask_strategy_question,
            context=context,
            placeholder="e.g., How should we position against Company X?",
        )


# =============================================================================
# ACCOUNT MANAGEMENT VIEW
# =============================================================================


def render_account_view() -> None:
    st.header("Account Management")

    all_calls = load_all_transcripts()

    tab1, tab2, tab3 = st.tabs(["Performance Report", "Pattern Analysis", "Call Details"])

    with tab1:
        render_metrics(all_calls)

        st.divider()

        if st.button("Generate Performance Report", key="am_report"):
            with st.spinner("Generating report..."):
                report = generate_performance_report(all_calls)
            st.markdown(report)

        st.divider()
        st.caption("BY INDUSTRY")

        industries = {}
        for call in all_calls:
            ind = call["metadata"]["prospect"]["industry"]
            if ind not in industries:
                industries[ind] = {"total": 0, "booked": 0}
            industries[ind]["total"] += 1
            if call["metadata"]["outcome"] == "meeting_booked":
                industries[ind]["booked"] += 1

        for ind, data in industries.items():
            rate = data["booked"] / data["total"] * 100
            color = "#7dd87d" if rate >= 50 else "#ff6b6b"
            st.markdown(
                f"<div style='padding: 0.3rem 0;'>"
                f"<span style='color: {color};'>{rate:.0f}%</span> "
                f"{ind} ({data['booked']}/{data['total']})"
                f"</div>",
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Success/Failure Patterns")

        if st.button("Analyze Patterns", key="am_patterns"):
            with st.spinner("Analyzing patterns..."):
                patterns = analyze_patterns(all_calls)
            st.markdown(patterns)

        st.divider()
        st.caption("ALL CALLS")
        render_call_list(all_calls)

    with tab3:
        json_files = get_json_files()
        selected = st.selectbox(
            "Select Call",
            json_files,
            format_func=lambda p: p.stem,
            key="am_call_select",
        )

        if selected:
            data = load_transcript(selected)
            meta = data["metadata"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", meta["prospect"]["company"])
            with col2:
                st.metric("Industry", meta["prospect"]["industry"])
            with col3:
                st.metric("Duration", f"{meta['duration_seconds']}s")
            with col4:
                st.metric("Outcome", meta["outcome"].replace("_", " ").title())

            with st.expander("View Transcript"):
                render_transcript(data["transcript"])


# =============================================================================
# CLIENT VIEW
# =============================================================================


def render_client_view() -> None:
    st.header("Client Portal")

    all_calls = load_all_transcripts()
    client_name = all_calls[0]["metadata"]["client"] if all_calls else "Client"

    tab1, tab2, tab3 = st.tabs(["Results Overview", "Call History", "Outcome Analysis"])

    with tab1:
        st.subheader(f"Campaign Results for {client_name}")

        render_metrics(all_calls)

        st.divider()

        if st.button("Generate Client Report", key="client_report"):
            with st.spinner("Generating report..."):
                report = generate_client_report(all_calls, client_name)
            st.markdown(report)

        st.divider()
        st.caption("MEETINGS BOOKED")

        booked = [c for c in all_calls if c["metadata"]["outcome"] == "meeting_booked"]
        for call in booked:
            meta = call["metadata"]
            meeting = meta.get("meeting_details", {})
            st.markdown(
                f"**{meta['prospect']['company']}** — {meta['prospect']['contact_name']} ({meta['prospect']['title']})"
            )
            if meeting:
                st.write(f"Meeting: {meeting.get('date', 'TBD')} at {meeting.get('time', 'TBD')}")
            st.write("---")

    with tab2:
        st.subheader("Call History")

        render_call_list(all_calls)

        st.divider()

        json_files = get_json_files()
        selected = st.selectbox(
            "View Call Details",
            json_files,
            format_func=lambda p: p.stem,
            key="client_call_select",
        )

        if selected:
            data = load_transcript(selected)
            meta = data["metadata"]

            st.caption(f"CALL TO {meta['prospect']['company'].upper()}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Contact:** {meta['prospect']['contact_name']}")
                st.write(f"**Title:** {meta['prospect']['title']}")
            with col2:
                st.write(f"**Industry:** {meta['prospect']['industry']}")
                st.write(f"**Outcome:** {meta['outcome'].replace('_', ' ').title()}")

            with st.expander("View Transcript"):
                render_transcript(data["transcript"])

    with tab3:
        st.subheader("Why Calls Succeeded or Failed")

        json_files = get_json_files()
        selected = st.selectbox(
            "Select Call to Analyze",
            json_files,
            format_func=lambda p: p.stem,
            key="client_outcome_select",
        )

        if selected:
            data = load_transcript(selected)
            meta = data["metadata"]

            outcome = meta["outcome"]
            is_success = outcome == "meeting_booked"
            color = "#7dd87d" if is_success else "#ff6b6b"

            st.markdown(
                f"<div style='font-size: 1.2rem; margin-bottom: 1rem;'>"
                f"<span style='color: {color}; font-weight: 600;'>"
                f"{outcome.replace('_', ' ').upper()}</span> — {meta['prospect']['company']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if st.button("Explain This Outcome", key="client_explain"):
                with st.spinner("Analyzing..."):
                    explanation = explain_outcome(data)
                st.markdown(explanation)


# =============================================================================
# MAIN APP
# =============================================================================


def main() -> None:
    if not get_json_files():
        st.error("No transcript files found in test-data/transcripts/")
        return

    st.sidebar.title("AI Marketing Demo")

    role = st.sidebar.radio(
        "Role",
        ["SDR", "Strategy Consultant", "Account Management", "Client"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("Select your role to access relevant features")

    # Show LLM provider info
    provider_info = get_current_provider_info()
    st.sidebar.divider()
    st.sidebar.caption(f"🤖 {provider_info['provider'].upper()}: {provider_info['model']}")

    if role == "SDR":
        render_sdr_view()
    elif role == "Strategy Consultant":
        render_strategy_view()
    elif role == "Account Management":
        render_account_view()
    else:
        render_client_view()


if __name__ == "__main__":
    main()
