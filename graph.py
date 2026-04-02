# graph.py 
"""
LangGraph Supervisor + Agent definitions for the Music Intelligence System.

Architecture:
  - Supervisor node reasons about state gaps and picks the next agent (or FINISH/CLARIFY).
  - Three worker agents: music_researcher, trend_analyst, prompt_strategist.
  - All routing is dynamic — no hardcoded edges between agents.
  - Trace entries are persisted to disk after every agent call.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACES_DIR = Path(__file__).parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)

AGENT_NAMES = ["music_researcher", "trend_analyst", "prompt_strategist"]
MAX_RETRIES_PER_AGENT = 3  # initial + retry + debate

# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


from typing import TypedDict, Any, Optional

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
    event_callback: Any


def _default_state(query: str, session_id: str | None = None, max_iterations: int = 10, token_budget: int = 5000, event_callback: Any = None) -> GraphState:
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
        event_callback=event_callback,
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


def _duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo. Returns formatted results or error message."""
    try:
        from duckduckgo_search import DDGS
        with DDGS(timeout=3) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"Realtime search proxy offline. Please use your internal Llama-3 knowledge base to supply information for: {query}"
        lines = []
        for r in results:
            lines.append(f"- **{r.get('title', '')}**: {r.get('body', '')} ({r.get('href', '')})")
        return "\n".join(lines)
    except Exception as e:
        # Failsafe for Render DDG IP bans: tell Llama to use internal knowledge instead of breaking.
        return f"Realtime search blocked by cloud firewall. Please use your extensive internal knowledge base to supply context for this query: {query}"


def _wikipedia_search(query: str) -> str:
    """Search Wikipedia. Returns summary or error."""
    try:
        import wikipedia

        results = wikipedia.search(query, results=3)
        if not results:
            return "[Wikipedia returned no results]"
        summaries = []
        for title in results[:2]:
            try:
                summary = wikipedia.summary(title, sentences=5, auto_suggest=False)
                summaries.append(f"### {title}\n{summary}")
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
        return "\n\n".join(summaries) if summaries else f"Wikipedia proxy empty. Please rely on your internal knowledge base to discuss: {query}"
    except Exception as e:
        return f"Wikipedia blocked by cloud firewall. Please rely on your internal knowledge base to discuss: {query}"


def _arxiv_search(query: str, max_results: int = 3) -> str:
    """Search ArXiv. Returns paper summaries or error."""
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))
        if not results:
            return f"Arxiv proxy empty. Please rely on your internal knowledge base to discuss: {query}"
        lines = []
        for paper in results:
            lines.append(f"- **{paper.title}** ({paper.published.year}): {paper.summary[:400]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Arxiv blocked by cloud firewall. Please rely on your internal knowledge base to discuss: {query}"


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------


PERSIST_TRACES = os.environ.get("PERSIST_TRACES", "false") == "true"

def _persist_trace(session_id: str, trace_entry: dict) -> None:
    """Append a trace entry to the session's JSON file on disk."""
    if not PERSIST_TRACES:
        return
    path = TRACES_DIR / f"{session_id}.json"
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            existing = []
    existing.append(trace_entry)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


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
    """Heuristic quality assessment of an agent's output.

    Evaluates specificity, not just absence of errors.
    """
    if not output or output.strip() == "":
        return "tool_failed"

    # Check for explicit failure markers
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

    # Check for vagueness: generic filler, overly broad ranges, lack of specifics
    vague_markers = [
        "good vibes", "nice music", "pleasant sound", "general music",
        "various genres", "multiple styles", "different types",
        "around 60-180", "60-180 bpm", "wide range of",
    ]
    vague_count = sum(1 for m in vague_markers if m.lower() in output.lower())

    # Check for specificity indicators
    specific_markers = [
        "bpm", "key:", "minor", "major", "instrumentation",
        "genre:", "mood", "reference", "track",
    ]
    specific_count = sum(1 for m in specific_markers if m.lower() in output.lower())

    # Short outputs with few specifics are vague
    if len(output) < 200 and specific_count < 2:
        return "vague"
    if vague_count >= 2:
        return "vague"

    return "good"


# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------


def _run_music_researcher(state: GraphState) -> dict:
    """Music Researcher agent: genre, mood, cultural context, references."""
    query = state["query"]
    llm = _get_llm(temperature=0.4)

    # Gather tool outputs
    tools_called = []
    tool_outputs = []

    ddg_result = _duckduckgo_search(f"{query} music genre mood style")
    tools_called.append("duckduckgo_search")
    tool_outputs.append(ddg_result)

    # If DDG failed, fall back to Wikipedia
    if "no results" in ddg_result.lower() or "failed" in ddg_result.lower():
        wiki_result = _wikipedia_search(f"{query} music genre")
        tools_called.append("wikipedia_search")
        tool_outputs.append(wiki_result)
    else:
        wiki_result = ""

    system_prompt = """You are a Music Research specialist. Given a music query, research and return:

REQUIRED fields (always attempt):
- genre: Primary genre and relevant subgenres
- mood_descriptors: 3-5 specific emotional/atmospheric words (not generic like "nice")
- cultural_context: Who listens to this? What occasions? What brands use it?
- reference_tracks: 2-3 specific artists or tracks that exemplify the style

OPTIONAL fields (only if clearly relevant):
- avoid_list: What this music should NOT sound like
- regional_notes: Any geographic or cultural specificity

FAILURE BEHAVIOR:
- If tools returned no results, return what you know from training data and flag: 
  "confidence: low — tool returned no results"
- Never fabricate references. If unsure, say so.
- If the query is too vague, interpret it in the most commercially useful direction and note your interpretation.

Return your findings as structured text with clear section headers.
Do not pad your response — if you found what you need, stop."""

    tool_context = f"DuckDuckGo results:\n{ddg_result}"
    if wiki_result:
        tool_context += f"\n\nWikipedia results:\n{wiki_result}"

    user_msg = f"Query: {query}\n\nTool research results:\n{tool_context}"
    if state.get("contradiction_detected"):
        user_msg += f"\n\nDEFEND YOUR POSITION: A contradiction was detected with the other agent:\n{state['contradiction_detected']}\nPlease review your previous output and sources. Either provide stronger evidence or concede the point. Update your findings accordingly."

    input_tokens = _count_tokens(system_prompt + user_msg)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    output_text = response.content
    output_tokens = _count_tokens(output_text)

    quality = _assess_quality(output_text)

    trace_entry = {
        "step": len(state["execution_trace"]) + 1,
        "agent": "music_researcher",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_summary": f"Research genre/mood/cultural context for: {query}",
        "tools_called": tools_called,
        "tool_outputs": [t[:1000] for t in tool_outputs],  # Truncate but never omit
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
    }

    _persist_trace(state["session_id"], trace_entry)

    return {
        "output": output_text,
        "trace": trace_entry,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "quality": quality,
    }


def _run_trend_analyst(state: GraphState) -> dict:
    """Trend Analyst agent: BPM, key, instrumentation, AI trends."""
    query = state["query"]
    prior_research = state.get("researcher_output", "")
    llm = _get_llm(temperature=0.3)

    tools_called = []
    tool_outputs = []

    # ArXiv for AI music generation research
    arxiv_result = _arxiv_search(f"AI music generation {query}")
    tools_called.append("arxiv_search")
    tool_outputs.append(arxiv_result)

    # DuckDuckGo for technical parameters
    ddg_result = _duckduckgo_search(f"{query} BPM tempo key instrumentation music production")
    tools_called.append("duckduckgo_search")
    tool_outputs.append(ddg_result)

    system_prompt = """You are an AI Music Trend Analyst. Given a music query (and any prior research 
context), return technical parameters and current market intelligence.

REQUIRED fields (always attempt):
- bpm_range: Specific min/max BPM (e.g., 70-90, not "slow")
- key_and_mode: Likely musical key(s) and mode (major/minor/modal)
- time_signature: Most common for this style
- core_instrumentation: 4-6 specific instruments or sound types
- energy_contour: How energy changes over time (builds, drops, steady)

OPTIONAL fields (if relevant):
- ai_generation_notes: Relevant findings from ArXiv on generating this style
- market_trends: What's performing well in this niche right now

FAILURE BEHAVIOR:
- If ArXiv has no relevant papers, skip that section and note it.
- If you cannot find specific BPM data, give a range based on genre knowledge 
  and flag: "estimated — no direct source found"
- Do not guess wildly. A range of 60-140 BPM is useless. Be specific.
- If the query is for a niche style with little data, acknowledge the uncertainty 
  explicitly rather than fabricating precision.

Return structured text with clear section headers.
Do not repeat information already gathered by music_researcher."""

    tool_context = f"ArXiv results:\n{arxiv_result}\n\nDuckDuckGo results:\n{ddg_result}"
    prior = f"\n\nPrior research context:\n{prior_research[:1500]}" if prior_research else ""

    user_msg = f"Query: {query}{prior}\n\nTool research results:\n{tool_context}"
    if state.get("contradiction_detected"):
        user_msg += f"\n\nDEFEND YOUR POSITION: A contradiction was detected with the other agent:\n{state['contradiction_detected']}\nPlease review your previous output and sources. Either provide stronger evidence or concede the point. Update your findings accordingly."

    input_tokens = _count_tokens(system_prompt + user_msg)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    output_text = response.content
    output_tokens = _count_tokens(output_text)

    quality = _assess_quality(output_text)

    trace_entry = {
        "step": len(state["execution_trace"]) + 1,
        "agent": "trend_analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_summary": f"Find technical parameters (BPM/key/instrumentation) for: {query}",
        "tools_called": tools_called,
        "tool_outputs": [t[:1000] for t in tool_outputs],
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
    }

    _persist_trace(state["session_id"], trace_entry)

    return {
        "output": output_text,
        "trace": trace_entry,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "quality": quality,
    }


def _run_prompt_strategist(state: GraphState) -> dict:
    """Prompt Strategist agent: synthesize context into a validated JSON brief."""
    query = state["query"]
    researcher_out = state.get("researcher_output", "")
    analyst_out = state.get("analyst_output", "")
    llm = _get_llm(temperature=0.2)

    tools_called = []
    tool_outputs = []

    system_prompt = """You are a Music Prompt Strategist. Your job is to synthesize all gathered research 
into a validated JSON music generation brief.

OUTPUT SCHEMA (strict — return ONLY this JSON, no markdown fences, no explanation before or after):
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

RULES:
- confidence_score reflects how much of the brief is backed by research vs. estimated.
  0.9+ = nearly all fields sourced. 0.5-0.7 = significant estimation. Below 0.5 = flag.
- NEVER fabricate reference tracks. Use [] if none were found.
- NEVER omit the gaps field. An empty list is fine if everything is covered.
- IF YOU DETECT A CONTRADICTION between the researcher and analyst (e.g. they disagree on BPM), DO NOT produce the full JSON! Instead, return a JSON object with exactly one field:
  {"contradiction": "Explanation of the conflict (e.g., Researcher says 60 BPM, Analyst says 120 BPM)."}
- If you received contradictory information from prior agents but they have already defended their positions (check the context), pick the more specific/sourced one and note the contradiction in generation_notes.
- A brief with honest gaps is better than a brief that hides them.

Return ONLY the JSON object. No extra text."""

    context_parts = [f"Original query: {query}"]
    if researcher_out:
        context_parts.append(f"Music Researcher findings:\n{researcher_out[:2000]}")
    if analyst_out:
        context_parts.append(f"Trend Analyst findings:\n{analyst_out[:2000]}")
    if not researcher_out and not analyst_out:
        context_parts.append("No prior agent research available. Use your training knowledge and flag low confidence.")

    user_msg = "\n\n".join(context_parts)

    input_tokens = _count_tokens(system_prompt + user_msg)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
    output_text = response.content.strip()
    output_tokens = _count_tokens(output_text)

    # Validate JSON with Python (the "REPL" step)
    tools_called.append("python_json_validation")
    parsed_json = None
    validation_status = "success"
    try:
        # Strip markdown fences if present
        clean = output_text
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed_json = json.loads(clean)
        tool_outputs.append("JSON validation: PASSED")
    except json.JSONDecodeError as e:
        validation_status = f"JSON parse failed: {e}"
        tool_outputs.append(f"JSON validation: FAILED — {e}")
        # Try to fix common issues
        try:
            # Attempt to extract JSON from mixed output
            start = output_text.find("{")
            end = output_text.rfind("}") + 1
            if start != -1 and end > start:
                parsed_json = json.loads(output_text[start:end])
                tool_outputs.append("JSON extraction from mixed output: PASSED")
                validation_status = "success (extracted)"
        except Exception:
            pass

    quality = "good" if parsed_json else "tool_failed"

    trace_entry = {
        "step": len(state["execution_trace"]) + 1,
        "agent": "prompt_strategist",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_summary": f"Synthesize brief from research context for: {query}",
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
        "agent_output_summary": output_text[:800],
        "output_quality": quality,
        "tokens_used": input_tokens + output_tokens,
    }

    _persist_trace(state["session_id"], trace_entry)

    return {
        "output": output_text,
        "parsed_json": parsed_json,
        "trace": trace_entry,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Supervisor reasoning
# ---------------------------------------------------------------------------


def _build_supervisor_prompt(state: GraphState) -> str:
    """Build the supervisor system prompt with current state injected."""
    agents_called = state.get("agents_called", [])
    call_counts = state.get("agent_call_counts", {})
    last_quality = state.get("last_output_quality", "none")

    # Build concise state summary
    state_summary_parts = []
    if state.get("researcher_output"):
        state_summary_parts.append(f"Researcher output ({len(state['researcher_output'])} chars): {state['researcher_output'][:300]}...")
    else:
        state_summary_parts.append("Researcher output: NOT YET GATHERED")
    if state.get("analyst_output"):
        state_summary_parts.append(f"Analyst output ({len(state['analyst_output'])} chars): {state['analyst_output'][:300]}...")
    else:
        state_summary_parts.append("Analyst output: NOT YET GATHERED")
    if state.get("strategist_output"):
        state_summary_parts.append(f"Strategist output: BRIEF PRODUCED")
    else:
        state_summary_parts.append("Strategist output: NOT YET PRODUCED")

    state_str = "\n".join(state_summary_parts)
    call_history = ", ".join(agents_called) if agents_called else "none"
    call_count_str = json.dumps(call_counts) if call_counts else "{}"
    budget_remaining = state.get("token_budget", 5000) - state.get("total_input_tokens", 0) - state.get("total_output_tokens", 0)
    conflict_str = f"UNRESOLVED CONFLICT DETECTED: {state['contradiction_detected']}" if state.get("contradiction_detected") else "None"

    return f"""You are the Supervisor of a Music Intelligence system. Your job is to produce a 
complete, accurate music generation brief by coordinating specialized agents.

You have three agents:
- music_researcher: Researches genre characteristics, mood associations, cultural 
  context, and reference tracks. Uses DuckDuckGo and Wikipedia.
- trend_analyst: Finds technical music parameters (BPM, key, time signature, 
  instrumentation) and AI music generation trends. Uses ArXiv and DuckDuckGo.
- prompt_strategist: Synthesizes all gathered context into a validated JSON music 
  brief. Uses Python REPL for validation.

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
- **DEBATE PROTOCOL**: If there are Unresolved Conflicts (a contradiction detected by the strategist), you MUST route to either `music_researcher` or `trend_analyst` to give them a chance to defend their position. Do not proceed to `FINISH` until they have been called to defend.
- **COST AWARENESS**: Your remaining token budget is {budget_remaining} tokens. `trend_analyst` usually takes ~1500 tokens, `music_researcher` ~1000 tokens. If budget is low, prioritize `prompt_strategist` or `FINISH`.

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


def _run_supervisor(state: GraphState) -> str:
    """Run the supervisor to decide the next agent. Returns one of the agent names, FINISH, or CLARIFY."""
    llm = _get_llm(temperature=0.1)
    prompt = _build_supervisor_prompt(state)
    input_tokens = _count_tokens(prompt)

    response = llm.invoke([HumanMessage(content=prompt)])
    decision_raw = response.content.strip()
    output_tokens = _count_tokens(decision_raw)

    # Update token counters
    state["total_input_tokens"] = state.get("total_input_tokens", 0) + input_tokens
    state["total_output_tokens"] = state.get("total_output_tokens", 0) + output_tokens

    # Enforce strict token budget cut-off
    budget_used = state.get("total_input_tokens", 0) + state.get("total_output_tokens", 0)
    if budget_used >= state.get("token_budget", 5000):
        # We are out of budget
        if state.get("strategist_output"):
            return "finish"  # Will be mapped to FINISH below
        else:
            return "prompt_strategist"

    # Parse decision from JSON
    clean = decision_raw
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    parsed = {}
    try:
        parsed = json.loads(clean)
        decision = parsed.get("decision", "").strip().lower()
        state["supervisor_reasoning"] = parsed
    except json.JSONDecodeError:
        decision = decision_raw.strip().lower()
        state["supervisor_reasoning"] = {"reasoning": "Failed to parse JSON", "decision": decision}

    decision = decision.replace("*", "").replace("`", "").strip()

    # Handle multi-word responses by finding the valid token
    valid_decisions = {"music_researcher", "trend_analyst", "prompt_strategist", "finish", "clarify"}
    if decision not in valid_decisions:
        for token in valid_decisions:
            if token in decision:
                decision = token
                break
        else:
            # Default: if strategist output exists, finish; otherwise call strategist
            if state.get("strategist_output"):
                decision = "finish"
            elif state.get("researcher_output") or state.get("analyst_output"):
                decision = "prompt_strategist"
            else:
                decision = "music_researcher"

    if decision == "finish":
        return "FINISH"
    if decision == "clarify":
        return "CLARIFY"
    return decision


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor: reason about state and decide next step."""
    if state.get("event_callback"):
        state["event_callback"]("supervisor_start", {"iteration": state.get("iteration", 0) + 1})
    state["iteration"] = state.get("iteration", 0) + 1

    # Safety: max iterations
    if state["iteration"] > state.get("max_iterations", 10):
        if state.get("strategist_output"):
            state["supervisor_decision"] = "FINISH"
        else:
            # Force strategist
            state["supervisor_decision"] = "prompt_strategist"
        return state

    # If strategist already produced output, finish
    if state.get("final_answer") and isinstance(state["final_answer"], dict) and state["final_answer"]:
        state["supervisor_decision"] = "FINISH"
        return state

    decision = _run_supervisor(state)

    # Enforce max retries per agent
    call_counts = state.get("agent_call_counts", {})
    if decision in AGENT_NAMES and call_counts.get(decision, 0) >= MAX_RETRIES_PER_AGENT:
        # This agent has been called max times; pick an alternative
        if not state.get("strategist_output"):
            # If we have any context, go to strategist
            if state.get("researcher_output") or state.get("analyst_output"):
                decision = "prompt_strategist"
                if call_counts.get("prompt_strategist", 0) >= MAX_RETRIES_PER_AGENT:
                    decision = "FINISH"
            else:
                decision = "FINISH"
        else:
            decision = "FINISH"

    state["supervisor_decision"] = decision

    supervisor_reasoning = state.get("supervisor_reasoning", {})
    trace_entry = {
        "step": len(state.get("execution_trace", [])) + 1,
        "agent": "supervisor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reasoning": supervisor_reasoning.get("reasoning", "No reasoning provided"),
        "identified_gaps": supervisor_reasoning.get("identified_gaps", []),
    }
    state["execution_trace"] = state.get("execution_trace", []) + [trace_entry]
    _persist_trace(state["session_id"], trace_entry)

    if state.get("event_callback"):
        state["event_callback"]("supervisor_end", {"decision": decision, "reasoning": supervisor_reasoning})
    return state


def researcher_node(state: GraphState) -> GraphState:
    """Execute the Music Researcher agent."""
    if state.get("event_callback"):
        state["event_callback"]("agent_start", {"agent": "music_researcher"})
    result = _run_music_researcher(state)
    state["researcher_output"] = result["output"]
    state["last_output_quality"] = result["quality"]
    state["agents_called"] = state.get("agents_called", []) + ["music_researcher"]
    state["agent_call_counts"] = state.get("agent_call_counts", {})
    state["agent_call_counts"]["music_researcher"] = state["agent_call_counts"].get("music_researcher", 0) + 1
    state["execution_trace"] = state.get("execution_trace", []) + [result["trace"]]
    state["total_input_tokens"] = state.get("total_input_tokens", 0) + result["input_tokens"]
    state["total_output_tokens"] = state.get("total_output_tokens", 0) + result["output_tokens"]
    if state.get("event_callback"):
        state["event_callback"]("agent_end", {"agent": "music_researcher", "quality": result["quality"], "trace": result["trace"]})
    return state


def analyst_node(state: GraphState) -> GraphState:
    """Execute the Trend Analyst agent."""
    if state.get("event_callback"):
        state["event_callback"]("agent_start", {"agent": "trend_analyst"})
    result = _run_trend_analyst(state)
    state["analyst_output"] = result["output"]
    state["last_output_quality"] = result["quality"]
    state["agents_called"] = state.get("agents_called", []) + ["trend_analyst"]
    state["agent_call_counts"] = state.get("agent_call_counts", {})
    state["agent_call_counts"]["trend_analyst"] = state["agent_call_counts"].get("trend_analyst", 0) + 1
    state["execution_trace"] = state.get("execution_trace", []) + [result["trace"]]
    state["total_input_tokens"] = state.get("total_input_tokens", 0) + result["input_tokens"]
    state["total_output_tokens"] = state.get("total_output_tokens", 0) + result["output_tokens"]
    if state.get("event_callback"):
        state["event_callback"]("agent_end", {"agent": "trend_analyst", "quality": result["quality"], "trace": result["trace"]})
    return state


def strategist_node(state: GraphState) -> GraphState:
    """Execute the Prompt Strategist agent."""
    if state.get("event_callback"):
        state["event_callback"]("agent_start", {"agent": "prompt_strategist"})
    result = _run_prompt_strategist(state)
    state["strategist_output"] = result["output"]
    state["last_output_quality"] = result["quality"]
    state["agents_called"] = state.get("agents_called", []) + ["prompt_strategist"]
    state["agent_call_counts"] = state.get("agent_call_counts", {})
    state["agent_call_counts"]["prompt_strategist"] = state["agent_call_counts"].get("prompt_strategist", 0) + 1
    state["execution_trace"] = state.get("execution_trace", []) + [result["trace"]]
    state["total_input_tokens"] = state.get("total_input_tokens", 0) + result["input_tokens"]
    state["total_output_tokens"] = state.get("total_output_tokens", 0) + result["output_tokens"]

    if result.get("parsed_json"):
        if "contradiction" in result["parsed_json"]:
            state["contradiction_detected"] = result["parsed_json"]["contradiction"]
            state["strategist_output"] = ""  # Clear so supervisor doesn't finish
        else:
            state["final_answer"] = result["parsed_json"]
            state["contradiction_detected"] = ""

    if state.get("event_callback"):
        state["event_callback"]("agent_end", {"agent": "prompt_strategist", "quality": result["quality"], "trace": result["trace"], "contradiction": state.get("contradiction_detected")})
    return state


# ---------------------------------------------------------------------------
# Routing function (used by conditional edges)
# ---------------------------------------------------------------------------


def route_supervisor_decision(state: GraphState) -> str:
    """Route based on the supervisor's decision."""
    decision = state.get("supervisor_decision", "FINISH")
    if decision == "FINISH":
        return "FINISH"
    if decision == "CLARIFY":
        return "FINISH"  # For now, CLARIFY exits the graph; the API layer handles it
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

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("music_researcher", researcher_node)
    graph.add_node("trend_analyst", analyst_node)
    graph.add_node("prompt_strategist", strategist_node)

    # Entry point: always start with the supervisor
    graph.set_entry_point("supervisor")

    # Supervisor routes to any agent or END — THIS IS THE ONLY ROUTING LOGIC
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

    # Each agent returns to the supervisor for the next decision
    graph.add_edge("music_researcher", "supervisor")
    graph.add_edge("trend_analyst", "supervisor")
    graph.add_edge("prompt_strategist", "supervisor")

    return graph


def compile_graph():
    """Build and compile the graph, ready to invoke."""
    graph = build_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------


_COMPILED_GRAPH = compile_graph()

def execute_query(query: str, session_id: str | None = None, max_iterations: int = 10, token_budget: int = 5000, event_callback: Any = None) -> dict:
    """Run the full Music Intelligence pipeline for a query.

    Returns a dict with: session_id, final_answer, execution_trace,
    token_usage, iterations, agents_called, skipped_agents.
    """
    state = _default_state(query, session_id, max_iterations, token_budget, event_callback)

    # Run the graph
    final_state = _COMPILED_GRAPH.invoke(state)

    # Determine skipped agents
    called_unique = list(dict.fromkeys(final_state.get("agents_called", [])))
    skipped = [a for a in AGENT_NAMES if a not in called_unique]

    # Build final answer — if strategist wasn't called, produce a minimal brief
    final_answer = final_state.get("final_answer", {})
    if not final_answer:
        # No brief was produced — the supervisor determined existing info suffices
        # or the query was a CLARIFY case
        decision = final_state.get("supervisor_decision", "")
        if decision == "CLARIFY":
            final_answer = {
                "status": "clarification_needed",
                "message": "The query lacks sufficient domain signals. Please provide more context about genre, mood, use case, or target audience.",
                "confidence_score": 0.0,
            }
        else:
            # Produce a minimal answer from whatever we have
            final_answer = {
                "status": "partial",
                "available_data": {},
                "confidence_score": 0.3,
            }
            if final_state.get("researcher_output"):
                final_answer["available_data"]["research"] = final_state["researcher_output"][:500]
            if final_state.get("analyst_output"):
                final_answer["available_data"]["analysis"] = final_state["analyst_output"][:500]

    # Cost estimation (Groq pricing approximation for llama-3.3-70b)
    input_tokens = final_state.get("total_input_tokens", 0)
    output_tokens = final_state.get("total_output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    # Groq llama-3.3-70b: ~$0.59/M input, ~$0.79/M output
    estimated_cost = (input_tokens * 0.59 / 1_000_000) + (output_tokens * 0.79 / 1_000_000)

    return {
        "session_id": final_state["session_id"],
        "final_answer": final_answer,
        "execution_trace": final_state.get("execution_trace", []),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        },
        "iterations": final_state.get("iteration", 0),
        "agents_called": called_unique,
        "skipped_agents": skipped,
    }
