"""
Streamlit Web Interface — Multi-Agent Research System
Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
import re
import html as html_lib
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import yaml
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator
from src.guardrails.safety_manager import SafetyManager

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Agent colour map for traces
# ---------------------------------------------------------------------------
AGENT_COLORS = {
    "Planner":    ("#1e3a5f", "#d0e8ff"),   # dark blue / light blue
    "Researcher": ("#1e4f2e", "#d4f5dc"),   # dark green / light green
    "Writer":     ("#4a2060", "#ecdcff"),   # dark purple / light purple
    "Critic":     ("#5c2a00", "#ffe8cc"),   # dark orange / light orange
    "default":    ("#333333", "#f0f0f0"),
}

# ---------------------------------------------------------------------------
# Config / session helpers
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    p = project_root / "config.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def init_session():
    """Set up all session-state keys on first load."""
    if "history" not in st.session_state:
        st.session_state.history: List[Dict] = []
    if "show_traces" not in st.session_state:
        st.session_state.show_traces = False
    if "show_safety_log" not in st.session_state:
        st.session_state.show_safety_log = True

    if "orchestrator" not in st.session_state:
        config = load_config()
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as exc:
            st.session_state.orchestrator = None
            st.session_state.orchestrator_error = str(exc)

    if "safety_manager" not in st.session_state:
        config = load_config()
        st.session_state.safety_manager = SafetyManager(config.get("safety", {}))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_query(query: str) -> Dict[str, Any]:
    """Run safety check → agent pipeline → output safety check."""
    sm: SafetyManager = st.session_state.safety_manager
    orch: AutoGenOrchestrator = st.session_state.orchestrator

    # --- Input safety ---
    input_check = sm.check_input_safety(query)
    if not input_check["safe"]:
        return {
            "query": query,
            "blocked": True,
            "response": sm.violation_message,
            "citations": [],
            "metadata": {},
            "safety_events": sm.get_safety_events()[-5:],
            "input_violations": input_check["violations"],
        }

    if orch is None:
        error_msg = getattr(st.session_state, "orchestrator_error", "Orchestrator not initialised")
        return {
            "query": query,
            "error": error_msg,
            "response": f"System error: {error_msg}",
            "citations": [],
            "metadata": {},
        }

    # --- Agent pipeline ---
    try:
        result = orch.process_query(input_check["query"])
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "response": f"Agent pipeline error: {exc}",
            "citations": [],
            "metadata": {},
        }

    response_text = re.sub(r'<think>.*?</think>', '', result.get("response", ""), flags=re.DOTALL).strip()

    # --- Output safety ---
    sources = result.get("metadata", {}).get("sources", [])
    output_check = sm.check_output_safety(response_text, sources)
    final_response = output_check.get("response", response_text)

    # --- Enrich metadata ---
    metadata = result.get("metadata", {})
    metadata["agent_traces"] = _build_traces(result)
    metadata["critique_score"] = _quality_score(result)
    metadata["output_action"] = output_check.get("action_taken", "none")
    metadata["output_violations"] = output_check.get("violations", [])

    return {
        "query": query,
        "response": final_response,
        "citations": _extract_citations(result),
        "metadata": metadata,
        "safety_events": sm.get_safety_events()[-10:],
        "blocked": False,
    }


def _extract_citations(result: Dict) -> List[str]:
    citations: List[str] = []
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join([str(c) for c in content])
        elif not isinstance(content, str):
            content = str(content)
        urls = re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", content)
        patterns = re.findall(r"\[Source:\s*([^\]]+)\]", content)
        for u in urls:
            if u not in citations:
                citations.append(u)
        for p in patterns:
            label = p.strip()
            if label and label not in citations:
                citations.append(label)
    return citations[:12]


def _build_traces(result: Dict) -> List[Dict]:
    """Return ordered list of {agent, content} for the conversation."""
    traces = []
    for msg in result.get("conversation_history", []):
        traces.append({
            "agent": msg.get("source", "Unknown"),
            "content": msg.get("content", ""),
        })
    return traces


def _quality_score(result: Dict) -> float:
    meta = result.get("metadata", {})
    score = 5.0
    score += min(meta.get("num_sources", 0) * 0.4, 2.0)
    if meta.get("critique"):
        score += 0.8
    score += min(meta.get("num_messages", 0) * 0.08, 2.2)
    return round(min(score, 10.0), 1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _agent_badge(agent: str) -> str:
    dark, light = AGENT_COLORS.get(agent, AGENT_COLORS["default"])
    return (
        f'<span style="background:{dark};color:{light};'
        f'padding:2px 8px;border-radius:12px;font-size:0.8em;'
        f'font-weight:bold;margin-right:6px;">{agent}</span>'
    )


def show_safety_banner(violations: List[Dict], action: str, event_type: str):
    """Render a prominent red warning box for safety violations."""
    if not violations:
        return
    lines = [f"**Safety {event_type.upper()} — Action: `{action.upper()}`**"]
    for v in violations:
        sev = v.get("severity", "?").upper()
        reason = v.get("reason", "")
        lines.append(f"- `[{sev}]` {reason}")
    st.error("\n".join(lines))


def display_response(result: Dict):
    """Render the full query result in the main column."""
    # Blocked query
    if result.get("blocked"):
        show_safety_banner(
            result.get("input_violations", []), "block", "input"
        )
        st.error(f"**Request blocked:** {result.get('response', '')}")
        return

    # General error
    if "error" in result:
        st.error(f"**System error:** {result['error']}")
        return

    meta = result.get("metadata", {})

    # Output safety warning (if content was sanitised or refused)
    output_action = meta.get("output_action", "none")
    output_violations = meta.get("output_violations", [])
    if output_violations:
        show_safety_banner(output_violations, output_action, "output")

    # --- Response ---
    st.markdown("### Research Response")
    st.markdown(result.get("response", ""))

    # --- Metrics row ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources", meta.get("num_sources", 0))
    c2.metric("Messages", meta.get("num_messages", 0))
    c3.metric("Quality Score", f"{meta.get('critique_score', 0):.1f} / 10")

    # --- Citations ---
    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})", expanded=False):
            for i, cite in enumerate(citations, 1):
                if cite.startswith("http"):
                    st.markdown(f"**[{i}]** [{cite}]({cite})")
                else:
                    st.markdown(f"**[{i}]** {cite}")

    # --- Agent traces ---
    if st.session_state.show_traces:
        traces = meta.get("agent_traces", [])
        if traces:
            display_agent_traces(traces)


def display_agent_traces(traces: List[Dict]):
    """Render per-agent message cards."""
    with st.expander("🔍 Agent Execution Traces", expanded=True):
        step = 0
        for trace in traces:
            agent = trace.get("agent", "Unknown")

            # Skip internal task-dispatch messages sent by the framework
            if agent.lower() == "user":
                continue

            step += 1
            content = trace.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            # Strip AutoGen termination token and model think blocks
            content = content.replace("TERMINATE", "").strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # Replace raw FunctionCall repr with a readable label
            if content.startswith("FunctionCall("):
                match = re.search(r"name='(\w+)'", content)
                fname = match.group(1) if match else "tool"
                qmatch = re.search(r'"query"\s*:\s*"([^"]+)"', content)
                qlabel = f': "{qmatch.group(1)}"' if qmatch else ""
                content = f"🔧 Calling {fname}{qlabel}"

            # HTML-escape so agent output with tags doesn't corrupt the layout
            content_safe = html_lib.escape(content)

            dark, light = AGENT_COLORS.get(agent, AGENT_COLORS["default"])
            badge = _agent_badge(agent)

            # Skip cards that are now empty after stripping
            if not content_safe.strip():
                continue

            st.markdown(
                f"""<div style="border-left:4px solid {dark};background:{light};
                padding:10px 14px;margin:6px 0;border-radius:4px;">
                {badge}
                <span style="font-size:0.75em;color:#666;">Step {step}</span>
                <div style="margin-top:6px;font-size:0.9em;white-space:pre-wrap;">
                {content_safe[:600]}{"…" if len(content_safe) > 600 else ""}
                </div></div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def display_sidebar():
    sm: SafetyManager = st.session_state.safety_manager
    config = load_config()

    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.show_traces = st.checkbox(
            "Show Agent Traces", value=st.session_state.show_traces
        )
        st.session_state.show_safety_log = st.checkbox(
            "Show Safety Event Log", value=st.session_state.show_safety_log
        )
        st.divider()

        # Stats
        stats = sm.get_safety_stats()
        st.title("📊 Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Queries", len(st.session_state.history))
        col2.metric("Violations", stats["violations"])
        st.metric("Safety checks run", stats["total_events"])

        st.divider()
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            sm.clear_events()
            st.rerun()

        # System info
        st.divider()
        st.markdown("### About")
        sys_cfg = config.get("system", {})
        st.markdown(f"**System:** {sys_cfg.get('name', 'Research Assistant')}")
        st.markdown(f"**Topic:** {sys_cfg.get('topic', 'Ethical AI in Education')}")
        model_cfg = config.get("models", {}).get("default", {})
        st.markdown(f"**Model:** {model_cfg.get('name', 'Unknown')}")

        # Safety event log
        if st.session_state.show_safety_log:
            st.divider()
            st.markdown("### 🛡️ Safety Event Log")
            events = sm.get_safety_events()
            if not events:
                st.info("No safety events yet.")
            else:
                for ev in reversed(events[-8:]):
                    ts = ev.get("timestamp", "")[:19]
                    t = ev.get("type", "?").upper()
                    safe = ev.get("safe", True)
                    icon = "✓" if safe else "⚠"
                    colour = "green" if safe else "red"
                    n_v = len(ev.get("violations", []))
                    st.markdown(
                        f'<div style="font-size:0.78em;color:{colour};">'
                        f"{icon} [{ts}] {t} | {n_v} violation(s)"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# History panel
# ---------------------------------------------------------------------------

def display_history():
    if not st.session_state.history:
        return
    with st.expander(f"📜 Query History ({len(st.session_state.history)})", expanded=False):
        for item in reversed(st.session_state.history):
            ts = item.get("timestamp", "")
            q = item.get("query", "")
            st.markdown(f"- **{ts}** — {q[:100]}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Ethical AI in Education — Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """<style>
        .stTextArea textarea { font-size: 1rem; }
        .block-container { padding-top: 1.5rem; }
        h1 { color: #1e3a5f; }
        </style>""",
        unsafe_allow_html=True,
    )

    init_session()

    # Header
    st.title("🤖 Multi-Agent Research Assistant")
    st.markdown(
        "**Topic:** Ethical AI in Education  |  "
        "**Agents:** Planner · Researcher · Writer · Critic  |  "
        "**Safety:** Input & Output Guardrails"
    )
    st.divider()

    display_sidebar()

    col_main, col_right = st.columns([3, 1])

    with col_main:
        query = st.text_area(
            "Enter your research query:",
            height=110,
            placeholder=(
                "e.g., What are the ethical implications of using AI for "
                "automated essay grading in secondary schools?"
            ),
        )

        if st.button("🔍 Research", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Agents are collaborating on your query…"):
                    result = process_query(query)

                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "query": query,
                        "result": result,
                    }
                )
                st.divider()
                display_response(result)
            else:
                st.warning("Please enter a research query first.")

        display_history()

    with col_right:
        st.markdown("### 💡 Example Queries")
        examples = [
            "How does AI bias affect underrepresented students?",
            "What privacy rights do students have with AI proctoring?",
            "How can universities govern AI tool use responsibly?",
            "Should AI grade essays in K-12 schools? Pros and cons.",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["_example"] = ex
                st.rerun()

        if "_example" in st.session_state:
            st.info(f"Query: {st.session_state['_example']}")

        st.divider()
        st.markdown("### ℹ️ How It Works")
        st.markdown(
            """
1. **Planner** analyses the query
2. **Researcher** gathers web sources
3. **Writer** synthesises a cited response
4. **Critic** verifies quality
5. **Guardrails** check input + output safety
            """
        )

        st.markdown("### 🛡️ Safety Categories")
        for cat in ["Harmful Content", "Prompt Injection", "Off-Topic", "PII Redaction"]:
            st.markdown(f"- {cat}")


if __name__ == "__main__":
    main()
