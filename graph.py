# graph.py
"""
LangGraph Supervisor + Agent definitions for the Music Intelligence System.

Architecture:
  - Supervisor node reasons about state gaps and picks the next agent (or FINISH/CLARIFY).
  - Three worker agents: music_researcher, trend_analyst, prompt_strategist.
  - All routing is dynamic — no hardcoded edges between agents.
  - Trace entries are persisted to disk after every agent call.

Fixes applied (v2):
  - Bug 1: Budget cutoff now returns "FINISH" (uppercase) consistently.
  - Bug 2: Node functions return dict updates instead of mutating state in place
           (pure-function paradigm required by LangGraph reducers).
  - Bug 3: event_callback removed from GraphState to enable future checkpointing.
           It is now passed via a thread-local wrapper at execution time.
  - Bug 4: Token counting now includes full message history in the agent loop.
  - Issue 5: Supervisor state summary increased from 300 to 600 chars.
  - Issue 6: Deprecated on_event replaced with lifespan in main.py (see main.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def validate_environment() -> None:
    """Validate that required environment variables are set."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY environment variable is not set. Please create a .env file or set it in your environment.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACES_DIR = Path(__file__).parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)

AGENT_NAMES = ["music_researcher", "trend_analyst", "prompt_strategist"]
MAX_RETRIES_PER_AGENT = 3  # initial + retry + debate

# ---------------------------------------------------------------------------
# Thread-local event callback (replaces storing callable in GraphState)
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_callback() -> Optional[Callable]:
    """Retrieve the event callback for the current thread, if any."""
    return getattr(_local, "event_callback", None)


def _set_callback(cb: Optional[Callable]) -> None:
    """Set the event callback for the current thread."""
    _local.event_callback = cb


def _emit(event_type: str, data: dict) -> None:
    """Fire an event callback if one is registered for this thread."""
    cb = _get_callback()
    if cb:
        try:
            cb(event_type, data)
        except Exception:
            pass  # Never let callback errors crash the graph


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

# NOTE: cl100k_base is an OpenAI encoding. Llama-3.3 uses a different tokenizer.
# We use cl100k_base as a close-enough proxy (~±15%) for cost estimation.
# For production, use the actual Llama tokenizer via HuggingFace tokenizers.
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _count_message_tokens(messages: list) -> int:
    """Count tokens across a full message list, including tool call content."""
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += _count_tokens(str(content))
        # Also count tool call arguments if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += _count_tokens(json.dumps(tc.get("args", {})))
    return total


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    query: str
    session_id: str
    agents_called: list
    agent_call_counts: dict
    last_output_quality: str
    researcher_output: str
    analyst_output: str
    strategist_output: str
    final_answer: dict
    execution_trace: list
    total_input_tokens: int
    total_output_tokens: int
    supervisor_decision: str
    supervisor_reasoning: dict
    max_iterations: int
    iteration: int
    token_budget: int
    contradiction_detected: str
    # NOTE: event_callback intentionally removed from state.
    # It is injected via thread-local (_set_callback) so graph state
    # remains fully serialisable for future checkpointer compatibility.


def _default_state(
    query: str,
    session_id: str | None = None,
    max_iterations: int = 10,
    token_budget: int = 5000,
) -> GraphState:
    return GraphState(
        query=query,
        session_id=session_id or str(uuid.uuid4()),
        agents_called=[],
        agent_call_counts={},
        last_output_quality="none",
        researcher_output="",
        analyst_output="",
        strategist_output="",
        final_answer={},
        execution_trace=[],
        total_input_tokens=0,
        total_output_tokens=0,
        supervisor_decision="",
        supervisor_reasoning={},
        max_iterations=max_iterations,
        iteration=0,
        token_budget=token_budget,
        contradiction_detected="",
    )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Tool wrappers (with graceful fallback)
# ---------------------------------------------------------------------------


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this to find music genre
    characteristics, BPM ranges, cultural context, reference tracks,
    instrumentation data, and current music trends."""
    try:
        from duckduckgo_search import DDGS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS(timeout=10) as ddgs:
                results = list(ddgs.text(query, max_results=5))
        if not results:
            return "[DuckDuckGo returned no results]"
        lines = []
        for r in results:
            lines.append(f"- **{r.get('title', '')}**: {r.get('body', '')} ({r.get('href', '')})")
        return "\n".join(lines)
    except Exception as e:
        return f"[DuckDuckGo search failed: {e}]"


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic information about music genres,
    cultural context, artists, historical music movements, and regional
    music traditions."""
    try:
        import wikipedia
        results = wikipedia.search(query, results=3)
        if not results:
            return "[Wikipedia returned no results]"
        summaries = []
        for title in results[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summaries.append(f"### {page.title}\n{page.summary[:1500]}")
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
        return "\n\n".join(summaries) if summaries else "[Wikipedia: no usable pages found]"
    except Exception as e:
        return f"[Wikipedia search failed: {e}]"


@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for academic papers on AI music generation, music
    information retrieval, and computational musicology. Use this to
    find research-backed data on BPM, key signatures, and instrumentation."""
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))
        if not results:
            return "[ArXiv returned no results]"
        lines = []
        for paper in results:
            lines.append(
                f"- **{paper.title}** ({paper.published.year}): {paper.summary[:400]}..."
            )
        return "\n".join(lines)
    except Exception as e:
        return f"[ArXiv search failed: {e}]"


@tool
def validate_json(json_string: str) -> str:
    """Validate and parse a JSON string. Returns 'VALID' if the JSON is
    well-formed, or an error message describing what is wrong so you can
    fix it. Use this before finalising your music brief output."""
    try:
        clean = json_string.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        json.loads(clean)
        return "VALID: JSON is well-formed and parseable."
    except json.JSONDecodeError as e:
        return f"INVALID: {e}. Fix the JSON and call this tool again."


def _run_agent_loop(
    llm,
    tools: list,
    system_prompt: str,
    user_message: str,
    max_tool_calls: int = 6,
    max_retries: int = 2,
) -> tuple[str, list[dict], int, int]:
    """
    Run a tool-calling agent loop with retry logic for LLM failures.

    Returns:
        (final_text_output, list_of_tool_calls_made, input_tokens, output_tokens)

    FIX: Token counts now reflect the full message history, not just the
    initial prompt, to prevent systematic undercounting.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    tools_called_log = []
    call_count = 0

    def _invoke_with_retry(msgs):
        """Invoke the LLM with exponential backoff on transient failures."""
        for attempt in range(max_retries + 1):
            try:
                return llm_with_tools.invoke(msgs)
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in [
                    "rate_limit", "rate limit", "429", "timeout",
                    "connection", "503", "502", "overloaded",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s
                    time.sleep(wait)
                    continue
                raise

    while call_count < max_tool_calls:
        response = _invoke_with_retry(messages)
        messages.append(response)

        if not response.tool_calls:
            # Count tokens across the full conversation history
            input_tokens = _count_message_tokens(messages[:-1])  # everything before final response
            output_tokens = _count_tokens(str(response.content))
            return response.content, tools_called_log, input_tokens, output_tokens

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                tool_result = f"[Error: tool '{tool_name}' not found]"
            else:
                try:
                    tool_result = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_result = f"[Tool execution error: {e}]"

            tools_called_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result_preview": str(tool_result)[:400],
            })

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )

        call_count += 1

    # Safety: hit max_tool_calls — ask LLM to wrap up
    messages.append(
        HumanMessage(content="You have used the maximum number of tool calls. "
                             "Synthesise your findings now and produce your final output.")
    )
    final = _invoke_with_retry(messages)
    input_tokens = _count_message_tokens(messages)
    output_tokens = _count_tokens(str(final.content))
    return final.content, tools_called_log, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------

PERSIST_TRACES = os.environ.get("PERSIST_TRACES", "true").lower() != "false"


def _persist_trace(session_id: str, trace_entry: dict) -> None:
    """Atomic append to trace file using write-to-temp-then-rename pattern."""
    if not PERSIST_TRACES:
        return
    
    path = TRACES_DIR / f"{session_id}.json"
    temp_path = TRACES_DIR / f".tmp.{session_id}.{uuid.uuid4().hex}.json"
    
    try:
        # Read existing
        existing = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, Exception):
                existing = []
        
        existing.append(trace_entry)
        
        # Atomic write
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
        temp_path.rename(path)
    except Exception:
        # Cleanup temp file if exists
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass


def load_trace(session_id: str) -> list[dict]:
    """Load trace from disk. Returns [] if not found."""
    path = TRACES_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Output quality assessment
# ---------------------------------------------------------------------------

def _assess_quality(output: str) -> str:
    """Heuristic quality assessment of an agent's output."""
    if not output or output.strip() == "":
        return "tool_failed"

    failure_markers = [
        "[duckduckgo returned no results]",
        "[duckduckgo search failed",
        "[wikipedia search failed",
        "[arxiv search failed",
        "[arxiv returned no results]",
        "tool returned no results",
    ]
    if any(m in output.lower() for m in failure_markers):
        return "tool_failed"

    vague_markers = [
        "good vibes", "nice music", "pleasant sound", "general music",
        "various genres", "multiple styles", "different types",
        "around 60-180", "60-180 bpm", "wide range of",
    ]
    vague_count = sum(1 for m in vague_markers if m.lower() in output.lower())

    specific_markers = [
        "bpm", "key:", "minor", "major", "instrumentation",
        "genre:", "mood", "reference", "track",
    ]
    specific_count = sum(1 for m in specific_markers if m.lower() in output.lower())

    if len(output) < 200 and specific_count < 2:
        return "vague"
    if vague_count >= 2:
        return "vague"

    return "good"


# ---------------------------------------------------------------------------
# Agent implementations
# FIX: All agent functions now return dict updates, not mutated state.
# ---------------------------------------------------------------------------

def _run_music_researcher(state: GraphState) -> dict:
    """Music Researcher: researches genre, mood, cultural context, reference tracks."""
    query = state["query"]
    llm = _get_llm(temperature=0.4)

    system_prompt = """You are a Music Research specialist with access to web search tools.

Your goal: research the music query and return a structured report covering:
- genre: primary genre and relevant subgenres
- mood_descriptors: 3-5 specific emotional/atmospheric words (not generic like "nice")
- cultural_context: who listens to this, what occasions, what brands use it
- reference_tracks: 2-3 specific real artists or tracks that exemplify the style
- avoid_list: what this music should NOT sound like (if relevant)
- regional_notes: geographic or cultural specificity (if relevant)

HOW TO USE YOUR TOOLS:
- Call duckduckgo_search first with a specific query like "{query} music genre mood style"
- If DuckDuckGo returns no results or fails, call wikipedia_search as a fallback
- Use multiple searches if your first result is too vague
- Stop searching once you have enough specific information

RULES:
- Never fabricate reference tracks. If unsure, say so.
- If tools return no results, use your training knowledge and flag: "confidence: low"
- A short, accurate answer beats a long, vague one."""

    user_message = f"Research this music query: {query}"

    if state.get("contradiction_detected"):
        user_message += (
            f"\n\nDEFEND YOUR POSITION: A contradiction was detected:\n"
            f"{state['contradiction_detected']}\n"
            f"Search for stronger evidence to support or update your previous findings."
        )

    t0 = time.time()
    output_text, tools_called_log, input_tokens, output_tokens = _run_agent_loop(
        llm=llm,
        tools=[duckduckgo_search, wikipedia_search],
        system_prompt=system_prompt,
        user_message=user_message,
        max_tool_calls=4,
    )

    latency_ms = int((time.time() - t0) * 1000)
    quality = _assess_quality(output_text)
    tool_names = list(dict.fromkeys(t["tool"] for t in tools_called_log))
    tool_output_previews = [t["result_preview"] for t in tools_called_log]

    trace_entry = {
        "step": len(state.get("execution_trace", [])) + 1,
        "agent": "music_researcher",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "input_summary": f"Research genre/mood/cultural context for: {query}",
        "tools_called": tool_names,
        "tool_outputs": tool_output_previews,
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
    }
    _persist_trace(state["session_id"], trace_entry)

    return {
        "researcher_output": output_text,
        "last_output_quality": quality,
        "agents_called": state.get("agents_called", []) + ["music_researcher"],
        "agent_call_counts": {
            **state.get("agent_call_counts", {}),
            "music_researcher": state.get("agent_call_counts", {}).get("music_researcher", 0) + 1,
        },
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "total_input_tokens": state.get("total_input_tokens", 0) + input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + output_tokens,
    }


def _run_trend_analyst(state: GraphState) -> dict:
    """Trend Analyst: finds BPM, key, instrumentation, and AI music generation trends."""
    query = state["query"]
    prior_research = state.get("researcher_output", "")
    llm = _get_llm(temperature=0.3)

    system_prompt = """You are an AI Music Trend Analyst with access to research tools.

Your goal: find technical music parameters and current market data for the query.

REQUIRED output fields:
- bpm_range: specific min/max BPM (e.g. 70-90, NOT "slow")
- key_and_mode: likely musical key(s) and mode (major/minor/modal)
- time_signature: most common for this style
- core_instrumentation: 4-6 specific instruments or sound types
- energy_contour: how energy changes over time (builds, drops, steady)

OPTIONAL fields (only if relevant):
- ai_generation_notes: findings from ArXiv on generating this style
- market_trends: what is performing well in this niche right now

HOW TO USE YOUR TOOLS:
- Call arxiv_search for academic research on AI music generation for this style
- Call duckduckgo_search for current BPM data, production techniques, market trends
- Use specific queries like "{query} BPM tempo key instrumentation music production"
- If one source fails, try the other

RULES:
- A range of 60-140 BPM is useless. Be specific — e.g. 85-95 BPM.
- If you cannot find a specific value, estimate from genre knowledge and flag it
- Do not repeat information already in the prior research context below"""

    user_message = f"Query: {query}"
    if prior_research:
        user_message += f"\n\nPrior research context (do not repeat this):\n{prior_research[:1500]}"
    user_message += "\n\nFind the technical parameters for this music query."

    if state.get("contradiction_detected"):
        user_message += (
            f"\n\nDEFEND YOUR POSITION: A contradiction was detected:\n"
            f"{state['contradiction_detected']}\n"
            f"Search for stronger evidence to support or update your technical findings."
        )

    t0 = time.time()
    output_text, tools_called_log, input_tokens, output_tokens = _run_agent_loop(
        llm=llm,
        tools=[arxiv_search, duckduckgo_search],
        system_prompt=system_prompt,
        user_message=user_message,
        max_tool_calls=4,
    )

    latency_ms = int((time.time() - t0) * 1000)
    quality = _assess_quality(output_text)
    tool_names = list(dict.fromkeys(t["tool"] for t in tools_called_log))
    tool_output_previews = [t["result_preview"] for t in tools_called_log]

    trace_entry = {
        "step": len(state.get("execution_trace", [])) + 1,
        "agent": "trend_analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "input_summary": f"Find technical parameters (BPM/key/instrumentation) for: {query}",
        "tools_called": tool_names,
        "tool_outputs": tool_output_previews,
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
    }
    _persist_trace(state["session_id"], trace_entry)

    return {
        "analyst_output": output_text,
        "last_output_quality": quality,
        "agents_called": state.get("agents_called", []) + ["trend_analyst"],
        "agent_call_counts": {
            **state.get("agent_call_counts", {}),
            "trend_analyst": state.get("agent_call_counts", {}).get("trend_analyst", 0) + 1,
        },
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "total_input_tokens": state.get("total_input_tokens", 0) + input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + output_tokens,
    }


def _run_prompt_strategist(state: GraphState) -> dict:
    """Prompt Strategist: synthesises research into a validated JSON brief."""
    query = state["query"]
    researcher_out = state.get("researcher_output", "")
    analyst_out = state.get("analyst_output", "")
    llm = _get_llm(temperature=0.2)

    system_prompt = """You are a Music Prompt Strategist with access to a JSON validation tool.

Your job: synthesise all research into a validated JSON music generation brief.

STEP 1 — Write the JSON brief matching this exact schema:
{
  "use_case": str,
  "mood_tags": [str],
  "genre": str,
  "subgenre": str or null,
  "bpm": {"min": int, "max": int},
  "key": str,
  "time_signature": str,
  "instrumentation": [str],
  "energy_level": "low" | "medium" | "high" | "dynamic",
  "duration_seconds": int,
  "reference_tracks": [str],
  "generation_notes": str,
  "confidence_score": float,
  "gaps": [str]
}

STEP 2 — Call validate_json with your JSON string to check it is valid.
  - If VALID: your task is complete. Output ONLY the JSON — no extra text.
  - If INVALID: fix the errors and call validate_json again until it passes.

CONTRADICTION RULE:
  If the researcher and analyst data directly contradict each other on a key
  parameter (e.g. BPM 60 vs BPM 170), do NOT produce the full JSON.
  Instead output exactly: {"contradiction": "clear explanation of the conflict"}
  Do NOT call validate_json in this case.

CONSTRAINT EXTRACTION:
   If the original query contains explicit constraints (e.g. "no lyrics",
   "no vocals", "under 2 minutes", "avoid piano"), you MUST capture them:
   - "no lyrics/vocals" -> set generation_notes to include "instrumental only"
   - "avoid X" -> add to generation_notes
   - Duration constraints -> set duration_seconds accordingly

RULES:
  - confidence_score: 0.9+ = nearly all fields sourced. Below 0.5 = flag.
  - Never fabricate reference_tracks. Use [] if none were found.
  - Never omit the gaps field. Empty list [] is fine if everything is covered.
  - A brief with honest gaps beats one that hides them.
  - Output ONLY the final JSON — no markdown fences, no explanation."""

    context_parts = [f"Original query: {query}"]
    if researcher_out:
        context_parts.append(f"Music Researcher findings:\n{researcher_out[:2000]}")
    if analyst_out:
        context_parts.append(f"Trend Analyst findings:\n{analyst_out[:2000]}")
    if not researcher_out and not analyst_out:
        context_parts.append(
            "No prior agent research available. "
            "Use your training knowledge and set confidence_score low."
        )

    user_message = "\n\n".join(context_parts) + "\n\nProduce the validated JSON brief now."

    t0 = time.time()
    output_text, tools_called_log, input_tokens, output_tokens = _run_agent_loop(
        llm=llm,
        tools=[validate_json],
        system_prompt=system_prompt,
        user_message=user_message,
        max_tool_calls=4,
    )

    latency_ms = int((time.time() - t0) * 1000)
    tool_names = list(dict.fromkeys(t["tool"] for t in tools_called_log))
    tool_output_previews = [t["result_preview"] for t in tools_called_log]

    # Parse the final JSON output
    parsed_json = None
    validation_status = "success"
    try:
        clean = output_text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed_json = json.loads(clean)
    except json.JSONDecodeError as e:
        validation_status = f"final parse failed: {e}"
        try:
            start = output_text.find("{")
            end = output_text.rfind("}") + 1
            if start != -1 and end > start:
                parsed_json = json.loads(output_text[start:end])
                validation_status = "success (extracted)"
        except Exception:
            pass

    quality = "good" if parsed_json else "tool_failed"

    trace_entry = {
        "step": len(state.get("execution_trace", [])) + 1,
        "agent": "prompt_strategist",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "input_summary": f"Synthesise brief from research context for: {query}",
        "tools_called": tool_names,
        "tool_outputs": tool_output_previews,
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
        "validation_status": validation_status,
    }
    _persist_trace(state["session_id"], trace_entry)

    # Determine what to update based on strategist output
    updates = {
        "strategist_output": output_text,
        "last_output_quality": quality,
        "agents_called": state.get("agents_called", []) + ["prompt_strategist"],
        "agent_call_counts": {
            **state.get("agent_call_counts", {}),
            "prompt_strategist": state.get("agent_call_counts", {}).get("prompt_strategist", 0) + 1,
        },
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "total_input_tokens": state.get("total_input_tokens", 0) + input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + output_tokens,
    }

    if parsed_json:
        if "contradiction" in parsed_json:
            updates["contradiction_detected"] = parsed_json["contradiction"]
            updates["strategist_output"] = ""  # Clear so supervisor doesn't finish
        else:
            updates["final_answer"] = parsed_json
            updates["contradiction_detected"] = ""

    return updates


# ---------------------------------------------------------------------------
# Supervisor reasoning
# ---------------------------------------------------------------------------

def _build_supervisor_prompt(state: GraphState) -> str:
    """Build the supervisor system prompt with current state injected."""
    agents_called = state.get("agents_called", [])
    call_counts = state.get("agent_call_counts", {})
    last_quality = state.get("last_output_quality", "none")

    # FIX: Increased truncation from 300 to 600 chars so supervisor has
    # enough context to assess output quality before routing.
    state_summary_parts = []
    ro = state.get("researcher_output") or ""
    if ro:
        state_summary_parts.append(
            f"Researcher output ({len(ro)} chars): "
            f"{ro[:600]}..."
        )
    else:
        state_summary_parts.append("Researcher output: NOT YET GATHERED")

    ao = state.get("analyst_output") or ""
    if ao:
        state_summary_parts.append(
            f"Analyst output ({len(ao)} chars): "
            f"{ao[:600]}..."
        )
    else:
        state_summary_parts.append("Analyst output: NOT YET GATHERED")

    if state.get("strategist_output"):
        state_summary_parts.append("Strategist output: BRIEF PRODUCED")
    else:
        state_summary_parts.append("Strategist output: NOT YET PRODUCED")

    state_str = "\n".join(state_summary_parts)
    call_history = ", ".join(agents_called) if agents_called else "none"
    call_count_str = json.dumps(call_counts) if call_counts else "{}"
    budget_remaining = (
        state.get("token_budget", 5000)
        - state.get("total_input_tokens", 0)
        - state.get("total_output_tokens", 0)
    )
    conflict_str = (
        f"UNRESOLVED CONFLICT DETECTED: {state['contradiction_detected']}"
        if state.get("contradiction_detected")
        else "None"
    )

    return f"""You are the Supervisor of a Music Intelligence system. Your job is to produce a
complete, accurate music generation brief by coordinating specialized agents.

You have three agents:
- music_researcher: Researches genre characteristics, mood associations, cultural
  context, and reference tracks. Uses DuckDuckGo and Wikipedia.
- trend_analyst: Finds technical music parameters (BPM, key, time signature,
  instrumentation) and AI music generation trends. Uses ArXiv and DuckDuckGo.
- prompt_strategist: Synthesizes all gathered context into a validated JSON music
  brief. Uses a JSON validator to ensure output correctness.

YOUR DECISION PROCESS (run this every time before choosing):

Step 1 — Read the current state carefully.
  What has already been gathered? What is still unknown or low-quality?

Step 2 — Identify the most critical gap.
  Is it: genre/mood context? technical parameters? structured output?
  Map each gap to the agent best suited to fill it.

Step 3 — Check if you already have enough.
  If the query is narrow and specific (e.g., "BPM for lo-fi hip hop"),
  you may already have sufficient knowledge without calling any agent.
  Do not call agents for the sake of calling them. Consider the remaining token budget.

Step 4 — Check the quality of the last output.
  If an agent returned vague, generic, or tool-failed output, either:
  a) Retry that agent once with a more specific sub-query, OR
  b) Proceed to prompt_strategist with a low confidence_score and noted gaps.

Step 5 — Decide.
  Return ONLY the name of the next agent, "FINISH", or "CLARIFY".
  - FINISH: enough context exists to produce or has produced a meaningful brief.
  - CLARIFY: query has zero domain signals and proceeding would require fabrication.
    Return a structured question to the caller. Do not invoke any agent first.

NOTE: These checks are guidelines, not a fixed sequence. Prioritize what matters
most for the current query. A narrow technical query may only need Step 3.
A failing agent may make Step 4 the only thing that matters.

IMPORTANT CONSTRAINTS:
- You may call any agent in any order.
- You may skip agents if their domain is already covered.
- You may retry an agent ONCE if its output was vague or a tool failed.
  Do not call the same agent more than twice total. Current call counts: {call_count_str}
- Call prompt_strategist only when you have enough context to produce a meaningful brief.
  "Enough context" means: at least mood/genre OR at least technical parameters.
  A brief with some unknowns is better than never finishing.
- If a brief has already been produced (strategist output exists), return FINISH.
- If all agents have been called and the brief is still incomplete,
  call prompt_strategist with instruction to note gaps explicitly.
- **DEBATE PROTOCOL**: If there are Unresolved Conflicts (a contradiction detected by
  the strategist), you MUST route to either music_researcher or trend_analyst to give
  them a chance to defend their position. Do not proceed to FINISH until they have
  been called to defend.
- **COST AWARENESS**: Your remaining token budget is {budget_remaining} tokens.
  trend_analyst uses ~1500 tokens, music_researcher ~1000 tokens.
  If budget is low, prioritize prompt_strategist or FINISH.

Current query: {state['query']}
Unresolved Conflicts: {conflict_str}

Current state:
{state_str}

Agents called so far (in order): {call_history}
Agent call counts: {call_count_str}
Last agent output quality: {last_quality}

RESPOND WITH EXACTLY THIS JSON STRUCTURE (No extra text, no markdown fences):
{{
  "reasoning": "Explain your decision process based on the steps above",
  "identified_gaps": ["list", "of", "gaps"],
  "gap_mapped_to": "agent_name or N/A",
  "decision": "music_researcher, trend_analyst, prompt_strategist, FINISH, or CLARIFY",
  "expected_contribution": "What you expect the chosen agent to produce"
}}"""


def _run_supervisor(state: GraphState) -> tuple[str, dict, int, int]:
    """
    Run the supervisor to decide the next agent.

    Returns:
        (decision: str, reasoning: dict, input_tokens: int, output_tokens: int)

    FIX: Returns token counts and decision separately instead of mutating state.
    FIX: Budget cutoff consistently returns uppercase "FINISH".
    """
    llm = _get_llm(temperature=0.1)
    prompt = _build_supervisor_prompt(state)
    input_tokens = _count_tokens(prompt)

    # Check budget BEFORE making the LLM call
    budget_used = state.get("total_input_tokens", 0) + state.get("total_output_tokens", 0)
    if budget_used + input_tokens >= state.get("token_budget", 5000):
        # Budget exhausted — FIX: return uppercase FINISH, not "finish"
        if state.get("strategist_output"):
            return "FINISH", {"reasoning": "Token budget exhausted, brief already produced."}, input_tokens, 0
        else:
            return "prompt_strategist", {"reasoning": "Token budget exhausted, forcing synthesis."}, input_tokens, 0

    response = llm.invoke([HumanMessage(content=prompt)])
    decision_raw = response.content.strip()
    output_tokens = _count_tokens(decision_raw)

    # Parse decision from JSON
    clean = decision_raw.strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        clean = clean[start:end+1]

    parsed = {}
    try:
        parsed = json.loads(clean)
        decision = parsed.get("decision", "").strip().lower()
    except json.JSONDecodeError:
        decision = decision_raw.strip().lower()
        parsed = {"reasoning": "Failed to parse supervisor JSON", "decision": decision}

    decision = decision.replace("*", "").replace("`", "").strip()

    valid_decisions = {"music_researcher", "trend_analyst", "prompt_strategist", "finish", "clarify"}
    if decision not in valid_decisions:
        for token in valid_decisions:
            if token in decision:
                decision = token
                break
        else:
            # Heuristic fallback
            if state.get("strategist_output"):
                decision = "finish"
            elif state.get("researcher_output") or state.get("analyst_output"):
                decision = "prompt_strategist"
            else:
                decision = "music_researcher"

    if decision == "finish":
        return "FINISH", parsed, input_tokens, output_tokens
    if decision == "clarify":
        return "CLARIFY", parsed, input_tokens, output_tokens
    return decision, parsed, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# LangGraph node functions
# FIX: All node functions return dict updates, not mutated GraphState objects.
# ---------------------------------------------------------------------------

def supervisor_node(state: GraphState) -> dict:
    """Supervisor: reason about state and decide next step."""
    _emit("supervisor_start", {"iteration": state.get("iteration", 0) + 1})

    new_iteration = state.get("iteration", 0) + 1

    # Safety: max iterations
    if new_iteration > state.get("max_iterations", 10):
        final_decision = "FINISH" if state.get("strategist_output") else "prompt_strategist"
        return {"iteration": new_iteration, "supervisor_decision": final_decision}

    decision, reasoning, input_tokens, output_tokens = _run_supervisor(state)

    # Enforce max retries per agent
    call_counts = state.get("agent_call_counts", {})
    if decision in AGENT_NAMES and call_counts.get(decision, 0) >= MAX_RETRIES_PER_AGENT:
        if state.get("researcher_output") or state.get("analyst_output"):
            strategist_calls = call_counts.get("prompt_strategist", 0)
            decision = "FINISH" if strategist_calls >= MAX_RETRIES_PER_AGENT else "prompt_strategist"
        else:
            decision = "FINISH"

    trace_entry = {
        "step": len(state.get("execution_trace", [])) + 1,
        "agent": "supervisor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reasoning": reasoning.get("reasoning", "No reasoning provided"),
        "identified_gaps": reasoning.get("identified_gaps", []),
    }
    _persist_trace(state["session_id"], trace_entry)
    _emit("supervisor_end", {"decision": decision, "reasoning": reasoning})

    return {
        "iteration": new_iteration,
        "supervisor_decision": decision,
        "supervisor_reasoning": reasoning,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "total_input_tokens": state.get("total_input_tokens", 0) + input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + output_tokens,
    }


def researcher_node(state: GraphState) -> dict:
    """Execute the Music Researcher agent."""
    _emit("agent_start", {"agent": "music_researcher"})
    updates = _run_music_researcher(state)
    _emit("agent_end", {
        "agent": "music_researcher",
        "quality": updates.get("last_output_quality"),
    })
    return updates


def analyst_node(state: GraphState) -> dict:
    """Execute the Trend Analyst agent."""
    _emit("agent_start", {"agent": "trend_analyst"})
    updates = _run_trend_analyst(state)
    _emit("agent_end", {
        "agent": "trend_analyst",
        "quality": updates.get("last_output_quality"),
    })
    return updates


def strategist_node(state: GraphState) -> dict:
    """Execute the Prompt Strategist agent."""
    _emit("agent_start", {"agent": "prompt_strategist"})
    updates = _run_prompt_strategist(state)
    _emit("agent_end", {
        "agent": "prompt_strategist",
        "quality": updates.get("last_output_quality"),
        "contradiction": updates.get("contradiction_detected"),
    })
    return updates


# ---------------------------------------------------------------------------
# Routing function (used by conditional edges)
# ---------------------------------------------------------------------------

def route_supervisor_decision(state: GraphState) -> str:
    """Route based on the supervisor's decision."""
    decision = state.get("supervisor_decision", "FINISH")
    if decision == "FINISH":
        return "FINISH"
    if decision == "CLARIFY":
        return "CLARIFY"
    if decision in AGENT_NAMES:
        return decision
    return "FINISH"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct the LangGraph with supervisor-driven conditional routing.

    NO hardcoded edges between agents. The supervisor decides every transition.
    """
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("music_researcher", researcher_node)
    graph.add_node("trend_analyst", analyst_node)
    graph.add_node("prompt_strategist", strategist_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "music_researcher": "music_researcher",
            "trend_analyst": "trend_analyst",
            "prompt_strategist": "prompt_strategist",
            "FINISH": END,
            "CLARIFY": END,
        },
    )

    graph.add_edge("music_researcher", "supervisor")
    graph.add_edge("trend_analyst", "supervisor")
    graph.add_edge("prompt_strategist", "supervisor")

    return graph


def compile_graph():
    """Build and compile the graph."""
    return build_graph().compile()


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------

_COMPILED_GRAPH = compile_graph()


def execute_query(
    query: str,
    session_id: str | None = None,
    max_iterations: int = 10,
    token_budget: int = 5000,
    event_callback: Any = None,
) -> dict:
    """Run the full Music Intelligence pipeline for a query.

    event_callback is injected via thread-local — NOT stored in graph state —
    to keep GraphState serialisable for future checkpointer compatibility.
    """
    # Inject callback into thread-local for this execution
    _set_callback(event_callback)

    try:
        state = _default_state(query, session_id, max_iterations, token_budget)
        final_state = _COMPILED_GRAPH.invoke(state)
    finally:
        _set_callback(None)  # Always clean up

    called_unique = list(dict.fromkeys(final_state.get("agents_called", [])))
    skipped = [a for a in AGENT_NAMES if a not in called_unique]

    final_answer = final_state.get("final_answer", {})
    if not final_answer:
        decision = final_state.get("supervisor_decision", "")
        if decision == "CLARIFY":
            final_answer = {
                "status": "clarification_needed",
                "message": (
                    "The query lacks sufficient domain signals. Please provide more "
                    "context about genre, mood, use case, or target audience."
                ),
                "confidence_score": 0.0,
            }
        else:
            final_answer = {
                "status": "partial",
                "available_data": {},
                "confidence_score": 0.3,
            }
            if final_state.get("researcher_output"):
                final_answer["available_data"]["research"] = final_state["researcher_output"][:500]
            if final_state.get("analyst_output"):
                final_answer["available_data"]["analysis"] = final_state["analyst_output"][:500]

    # Groq llama-3.3-70b pricing: ~$0.59/M input, ~$0.79/M output
    input_tokens = final_state.get("total_input_tokens", 0)
    output_tokens = final_state.get("total_output_tokens", 0)
    estimated_cost = (input_tokens * 0.59 / 1_000_000) + (output_tokens * 0.79 / 1_000_000)

    return {
        "session_id": final_state["session_id"],
        "final_answer": final_answer,
        "execution_trace": final_state.get("execution_trace", []),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total": input_tokens + output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        },
        "iterations": final_state.get("iteration", 0),
        "agents_called": called_unique,
        "skipped_agents": skipped,
    }