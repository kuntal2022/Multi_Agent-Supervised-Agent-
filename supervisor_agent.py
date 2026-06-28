"""
Multi-Agent Supervisor — Streamlit Control Room
=================================================
Visualizes the supervisor -> researcher/mathematician -> final loop in real time.
"""

import os
import time
import dotenv
import streamlit as st
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import operator
from langchain_community.tools import TavilySearchResults

dotenv.load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Control Room — Multi-Agent Supervisor",
    page_icon="🛰️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# STYLE — terminal / mission-control aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0b0e12;
    --panel: #11151b;
    --panel-border: #1f2731;
    --accent: #ff8a3d;
    --accent-dim: #5a3a22;
    --text: #d9dee5;
    --text-dim: #6b7684;
    --ok: #4dd28c;
    --wire: #2a3340;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(255,138,61,0.06), transparent 40%),
        var(--bg);
}

#MainMenu, footer, header {visibility: hidden;}

.crm-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 0;
}
.crm-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.crm-sub {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin-top: 0.3rem;
    margin-bottom: 1.6rem;
}

/* Org chart */
.org-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    padding: 1.2rem 0 0.6rem 0;
}
.node {
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid var(--wire);
    background: var(--panel);
    border-radius: 10px;
    padding: 0.65rem 1.3rem;
    font-size: 0.85rem;
    color: var(--text-dim);
    transition: all 0.25s ease;
    text-align: center;
    min-width: 150px;
}
.node-active {
    border-color: var(--accent);
    color: var(--accent);
    background: linear-gradient(180deg, rgba(255,138,61,0.10), rgba(255,138,61,0.02));
    box-shadow: 0 0 0 1px rgba(255,138,61,0.25), 0 0 18px rgba(255,138,61,0.18);
}
.node-done {
    border-color: var(--ok);
    color: var(--ok);
}
.row {
    display: flex;
    gap: 2.4rem;
    justify-content: center;
}
.vline {
    width: 1px;
    height: 22px;
    background: var(--wire);
}

/* Log lines */
.log-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.45rem;
    border-left: 2px solid var(--wire);
    background: var(--panel);
    color: var(--text-dim);
    animation: fadein 0.4s ease;
}
.log-supervisor { border-left-color: var(--accent); color: var(--text); }
.log-researcher { border-left-color: #5fb4ff; color: var(--text); }
.log-mathematician { border-left-color: #c792ea; color: var(--text); }
.log-final { border-left-color: var(--ok); color: var(--text); }

.tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
    margin-right: 0.5rem;
    text-transform: uppercase;
}
.tag-supervisor { background: var(--accent-dim); color: var(--accent); }
.tag-researcher { background: #1d3146; color: #5fb4ff; }
.tag-mathematician { background: #2c1f3d; color: #c792ea; }
.tag-final { background: #163328; color: var(--ok); }

@keyframes fadein {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Answer panel */
.answer-panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--ok);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--text);
}

div[data-testid="stTextInput"] input {
    background: var(--panel) !important;
    border: 1px solid var(--wire) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stButton button {
    background: var(--accent) !important;
    color: #1a0e05 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton button:hover {
    background: #ffa05f !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AGENT SYSTEM (same logic as the original script)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, max_tokens=4000, request_timeout=120)

base_llm = get_llm()


class SupervisorDecision(BaseModel):
    """Supervisor's routing decision - must pick one of these options"""
    next_agent: Literal["researcher", "mathematician", "final"] = Field(
        description="Which worker should work next, or 'final' if enough information is gathered."
    )
    reason: str = Field(description="Reasoning behind the decision.")


class FinalAnswer(BaseModel):
    """Final answer model"""
    answer: str = Field(description="Final answer to the user's question.")


class MathStringExtractor(BaseModel):
    """Extract exact mathematical expression from a string"""
    math_string: str = Field(description="Extract exact mathematical expression for calculation")


class SupervisorState(TypedDict):
    query: str
    history: Annotated[list, operator.add]
    final_answer: str


def calculate(expression: str) -> str:
    """Simple calculator tool that evaluates a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@st.cache_resource
def get_tavily_tool():
    return TavilySearchResults(search_kwargs={"num_results": 2})

tavily_tool = get_tavily_tool()


def web_search(query: str) -> str:
    """Simple web search with the Tavily tool"""
    output = []
    try:
        search_result = tavily_tool.invoke(query)
        for i in search_result:
            output.append(f"Title: {i.get('title', 'No title')}")
            output.append(f"Content: {i.get('content', 'No snippet')}")
            output.append(f"URL: {i.get('url', 'No link')}")
    except Exception as e:
        output.append(f"Error: {e}")
    return "\n".join(output) if output else "No results found."


def supervisor_node(state: SupervisorState) -> Command[Literal["researcher", "mathematician", "final"]]:
    done_so_far = "\n".join(state.get("history", [])) or "Nothing Yet"
    struct_llm = base_llm.with_structured_output(SupervisorDecision)
    query = state["query"]

    prompt = f"""
                You are a supervisor managing 2 workers.

                - "researcher": does a web search for the query asked by the user and gets the
                  relevant part - recent news, updates, factual updates etc

                - "mathematician": does calculation

                Original Query:
                {query}

                Work completed so far:
                {done_so_far}

                Decide what should happen next. If you have enough information from the work
                done so far for a full answer of the original query, choose 'final'.
            """
    decision: SupervisorDecision = struct_llm.invoke(prompt)

    if "log_callback" in st.session_state and st.session_state.log_callback:
        st.session_state.log_callback("supervisor", decision.next_agent, decision.reason)

    return Command(
        goto=decision.next_agent,
        update={"history": [f"[Supervisor] decided to go to {decision.next_agent} because: {decision.reason}"]}
    )


def research_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    if "log_callback" in st.session_state and st.session_state.log_callback:
        st.session_state.log_callback("researcher", None, "Searching the web...")

    result = web_search(state["query"])

    return Command(
        goto="supervisor",
        update={"history": [f"[Researcher] {result}"]}
    )


def math_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    if "log_callback" in st.session_state and st.session_state.log_callback:
        st.session_state.log_callback("mathematician", None, "Extracting expression and calculating...")

    extractor = base_llm.with_structured_output(MathStringExtractor)
    prompt = f"""You are a mathematical expression extractor. Provide the extracted result
    as a string from the given query.

    Example:
    "What is the value of 10+20?" -> "10 + 20"
    "What is the GDP of India and what is 10*20?" -> "10 * 20"

    Query:
    {state['query']}
    """
    math_string = extractor.invoke(prompt)
    result = calculate(math_string.math_string)

    return Command(
        goto="supervisor",
        update={"history": [f"[Mathematician] {math_string.math_string} = {result}"]}
    )


def final_node(state: SupervisorState) -> Command[Literal["__end__"]]:
    if "log_callback" in st.session_state and st.session_state.log_callback:
        st.session_state.log_callback("final", None, "Writing final answer...")

    done_so_far = "\n".join(state["history"])
    final_llm_struct = base_llm.with_structured_output(FinalAnswer)

    prompt = f"""
    Original Query:
    {state['query']}

    Work Completed:
    {done_so_far}

    Write the final answer for the user.

    Rules:
    - Each sentence should have a relevant and crisp answer.
    - 2 different types of answers should not be in a single line
      (Example: - India's GDP in 2025 (URL: mention the url if any): X.X Tn $
                - 10 + 20 = 30)
    - Never mention which workers have worked.
    """
    result: FinalAnswer = final_llm_struct.invoke(prompt)

    return Command(goto=END, update={"final_answer": result.answer})


@st.cache_resource
def build_graph():
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("mathematician", math_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "supervisor")
    return builder.compile()

app = build_graph()


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "events" not in st.session_state:
    st.session_state.events = []
if "active_node" not in st.session_state:
    st.session_state.active_node = None
if "running" not in st.session_state:
    st.session_state.running = False
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "log_callback" not in st.session_state:
    st.session_state.log_callback = None


def log_event(agent, goto, reason):
    st.session_state.events.append({"agent": agent, "goto": goto, "reason": reason})
    st.session_state.active_node = goto if goto else agent


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="crm-eyebrow">// SUPERVISOR · RESEARCHER · MATHEMATICIAN</div>', unsafe_allow_html=True)
st.markdown('<div class="crm-title">Control Room</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="crm-sub">One question goes in. The supervisor decides who works on it, '
    'loops until it has enough, then writes the final answer.</div>',
    unsafe_allow_html=True
)

col_input, col_button = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "query",
        placeholder="e.g. What is the GDP of India in 2025 and what is 10 + 20?",
        label_visibility="collapsed"
    )
with col_button:
    run_clicked = st.button("Dispatch →", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LAYOUT — org chart (left) + live log (right)
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.4])

org_placeholder = left.empty()
log_placeholder = right.empty()
answer_placeholder = st.empty()


def render_org_chart(active=None, done=None):
    done = done or set()

    def cls(name):
        if name == active:
            return "node node-active"
        if name in done:
            return "node node-done"
        return "node"

    html = f"""
    <div class="org-wrap">
        <div class="{cls('supervisor')}">🛰️ SUPERVISOR</div>
        <div class="vline"></div>
        <div class="row">
            <div>
                <div class="{cls('researcher')}">🔎 RESEARCHER</div>
            </div>
            <div>
                <div class="{cls('mathematician')}">📐 MATHEMATICIAN</div>
            </div>
        </div>
        <div class="vline"></div>
        <div class="{cls('final')}">✍️ FINAL ANSWER</div>
    </div>
    """
    org_placeholder.markdown(html, unsafe_allow_html=True)


def render_log():
    if not st.session_state.events:
        log_placeholder.markdown(
            '<div class="log-line" style="text-align:center; color: var(--text-dim);">'
            'Awaiting dispatch...</div>',
            unsafe_allow_html=True
        )
        return

    rows = []
    for e in st.session_state.events:
        agent = e["agent"]
        tag_class = f"tag-{agent}"
        line_class = f"log-{agent}"
        if agent == "supervisor":
            text = f"routing to <b>{e['goto']}</b> — {e['reason']}"
        else:
            text = e["reason"]
        rows.append(
            f'<div class="log-line {line_class}">'
            f'<span class="tag {tag_class}">{agent}</span>{text}</div>'
        )
    log_placeholder.markdown("".join(rows), unsafe_allow_html=True)


# initial render
render_org_chart(active=st.session_state.active_node)
render_log()
if st.session_state.final_answer:
    answer_placeholder.markdown(
        f'<div class="answer-panel">{st.session_state.final_answer}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if run_clicked and query.strip():
    st.session_state.events = []
    st.session_state.final_answer = None
    st.session_state.active_node = "supervisor"

    done_nodes = set()

    def live_log(agent, goto, reason):
        log_event(agent, goto, reason)
        if agent != "supervisor":
            done_nodes.add(agent)
        render_org_chart(active=st.session_state.active_node, done=done_nodes)
        render_log()
        time.sleep(0.35)

    st.session_state.log_callback = live_log

    with st.spinner(""):
        try:
            result = app.invoke({"query": query, "history": []})
            st.session_state.final_answer = result["final_answer"]
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    render_org_chart(active="final", done={"researcher", "mathematician", "supervisor"})
    render_log()
    if st.session_state.final_answer:
        answer_placeholder.markdown(
            f'<div class="answer-panel">{st.session_state.final_answer}</div>',
            unsafe_allow_html=True
        )

elif run_clicked:
    st.warning("Type a question first.")