# System Evaluation & Postmortem: Wubble Music Intelligence Agent

**Date**: March 2026
**Author**: Indrajith MP
**Project**: Wubble (Autonomous Music Brief Generator)

This document serves as the formal architectural review, empirical routing analysis, and production roadmap for the Wubble Music Intelligence Agent system following the completion of the "Uniqueness Implementation" phases.

---

## 1. Architecture Decisions

Our primary objective was to move away from rigid, linear LLM pipelines (e.g., standard LangChain sequential chains) into a non-deterministic, reasoned orchestration model using LangGraph. To survive at product-scale, two critical architectural enhancements were deployed:

### The Agent Debate Protocol

In multi-agent systems, agents frequently hallucinate conflicting parameters because they pull from divergent sources. A researcher looking at "Doom Metal" might establish a 60 BPM standard, while the trend analyst pulling parameters for "upbeat pacing" locks onto 170 BPM.

Instead of allowing the synthesis agent (`prompt_strategist`) to blend these silently into an unplayable track, we implemented the **Debate Protocol**. If the strategist detects an unresolvable conflict, it interrupts compilation, logs the contradiction, and the Supervisor loops the query back to the siloed agents. They must either explicitly defend their positions with cited source material or concede. This drastically reduces hallucinated or conflicting musical specs.

### Cost-Aware Routing

In a production SaaS environment where API calls operate on a per-token margin, calling three agents for a simple query destroys unit economics. Wubble's supervisor makes real-time calculations against an injected `token_budget`. We modeled average consumption patterns (~1500 tokens for Analyst, ~1000 for Researcher). Before invoking an agent, the supervisor evaluates its budget. If funds are low, it dynamically short-circuits to the most critical synthesis step or produces early validation stops, maximizing context per-cent rather than blindly seeking perfect context.

---

## 2. Routing Analysis & Performance Verification

The `/v1/evaluate` endpoint executes these queries concurrently. The traces below demonstrate the Supervisor's dynamic adaptation:

### Query A: "What BPM range is typical for lo-fi hip hop?"

* **Classification:** Highly specific, purely technical query.
* **Routing Strategy:** Efficient routing yielding accurate answers but bypassing irrelevant cultural sweeps.
* **Agents Called:** `["trend_analyst", "prompt_strategist"]`
* **Skipped Agents:** `["music_researcher"]`
* **Outcome:** Exact context retrieved at 1/3 the normal cost. The prompt strategist successfully formats the technical parameter into the JSON brief.

### Query B: "Music brief for a luxury EV brand ad targeting millennials — futuristic but warm"

* **Classification:** Abstract, requiring cultural context, technical mapping, and structured output.
* **Routing Strategy:** Full Pipeline Execution.
* **Agents Called:** `["music_researcher", "trend_analyst", "prompt_strategist"]`
* **Skipped Agents:** `[]`
* **Outcome:** High-confidence output seamlessly blending brand aesthetic into technical synth-wave pacing.

### Query C: "A song with slow doom metal mood but 170 BPM upbeat tempo"

* **Classification:** Intentional Contradiction.
* **Routing Strategy:** Extended Debate Protocol.
* **Agents Called:** `["music_researcher", "trend_analyst", "prompt_strategist", "music_researcher", "trend_analyst", "prompt_strategist"]`
* **Metrics:** Total tokens: ~4,200 | Latency: ~8.2s | Cost: $0.0031
* **Outcome:** First strategist execution triggered the `contradiction_detected` state. The loop forced the researcher and analyst to defend their metrics. The second strategist execution resolved the tempo discrepancy safely, proving the system self-corrects while staying within the dynamically allocated budget limit.

> Raw state dump of this debate cycle is available in [`debate_proof.json`](./debate_proof.json), showing the exact `contradiction_detected` field, the re-routing supervisor decision, and the final resolved brief.

### Query D: "Background music for an 8-hour sleep stories feature in a meditation app"

* **Classification:** Tool failure + graceful degradation under sparse search results.
* **Routing Strategy:** Partial pipeline with quality-driven fallback.
* **Agents Called:** `["music_researcher", "prompt_strategist"]`
* **Skipped Agents:** `["trend_analyst"]`
* **Metrics:** Total tokens: ~2,100 | Latency: ~5.8s | Confidence: 0.48
* **What happened in the code:** `music_researcher` ran first and DuckDuckGo returned sparse results for the highly specific "8-hour sleep stories" niche — `_assess_quality()` evaluated the output as `"vague"` due to fewer than 2 specificity markers in a short response. The Supervisor read `last_output_quality: "vague"` and `tool_outputs` showing `"[DuckDuckGo returned no results]"` in the trace, then made a cost-based decision: retrying `trend_analyst` on a query with already-depleted tool reliability was unlikely to improve output quality relative to the token cost. It routed directly to `prompt_strategist` with instruction to note gaps explicitly.
* **Output behaviour:** The final brief was produced with `confidence_score: 0.48` and a populated `gaps` field: `["specific instrumentation data unavailable", "BPM range estimated from genre knowledge — no direct source found"]`. The brief did not crash, fabricate specifics, or hide its uncertainty. `skipped_agents` in the response explicitly listed `["trend_analyst"]`.
* **Key insight:** This run proves the system degrades honestly rather than silently. A confidence score below 0.5 is not a failure — it is the system correctly reporting the quality of available evidence. A system that always returns 0.9 confidence is lying.

### Query E: "Make it sound cool"

* **Classification:** Zero-signal ambiguous query — CLARIFY path.
* **Routing Strategy:** Supervisor terminates without calling any agents.
* **Agents Called:** `[]`
* **Skipped Agents:** `["music_researcher", "trend_analyst", "prompt_strategist"]`
* **Metrics:** Total tokens: ~380 (supervisor call only) | Latency: ~1.2s | Confidence: 0.0
* **What happened in the code:** The Supervisor's `_build_supervisor_prompt()` injected the full state (all agent outputs empty, budget full) alongside the query. The LLM returned `{"decision": "CLARIFY", ...}` — recognising that proceeding would require fabricating domain signals from nothing. `route_supervisor_decision()` mapped `"CLARIFY"` to `END`, terminating the graph immediately. No agents were invoked.
* **Output behaviour:** Response contained `"status": "clarification_needed"` and `"message": "The query lacks sufficient domain signals. Please provide more context about genre, mood, use case, or target audience."` — with `confidence_score: 0.0`.
* **Why this matters for production:** Calling three agents on a zero-signal query wastes ~4,000 tokens and ~8 seconds to produce a meaningless brief. The CLARIFY branch saves both. More importantly, it is architecturally honest — the system knows when it does not know, which is a harder property to build than high-confidence outputs. The Supervisor's ability to exercise CLARIFY demonstrates it is genuinely reasoning about query sufficiency, not just dispatching agents reflexively.
* **Optimisation note for ambiguous queries:** The Supervisor prompt's Step 3 (`"Check if you already have enough"`) was specifically designed to handle this case. For production, a pre-classification layer before the graph (a lightweight intent classifier) could intercept zero-signal queries before they reach the LLM-powered Supervisor at all, reducing even the ~380-token cost of the current CLARIFY path.

---

## 3. Failure Modes & Edge Cases

No autonomous system is perfect. Here's what occurs at the edges of Wubble's capabilities:

* **Supervisor reasoning poisoning:** If the LLM returns malformed JSON for the supervisor decision, the fallback logic in `_run_supervisor()` defaults to a heuristic. We tested this by intentionally passing a corrupted prompt. The system recovered correctly 4/5 times — on the 5th it defaulted to `music_researcher` when `prompt_strategist` was more appropriate. **Fix:** add a confidence threshold — if the supervisor JSON parse fails twice in a row, force `prompt_strategist` with `confidence_score: 0.3`.
* **Empty DuckDuckGo / ArXiv Context:** If search tools return `None` or 404s, the respective agent returns a low `confidence_score`. The Supervisor reads the trace, logs a specific `reasoning` failure, and optionally attempts a minor query rephrasing for a second pass before abandoning the branch. *Example excerpt: `"tool_outputs": ["[duckduckgo returned no results]"], "output_quality": "tool_failed"`.*
* **Irreconcilable Debate:** If agents refuse to concede their positions during a debate protocol loop, the system protects against infinite looping via the `MAX_RETRIES_PER_AGENT` cut-off (set to 3). The Supervisor forcefully routes to `FINISH`, appending a `"warning"` flag into the final JSON indicating a manual review is needed.
* **Token Budget Starvation:** If a user submits an aggressive budget (e.g., `< 2000 tokens`), the supervisor simply refuses to start the expensive `trend_analyst`. It routes exclusively to the `music_researcher` and then the strategist. Production quality drops notably (the outputs sound structurally generic and lack exact musical key definitions), but the pipeline stays within economic boundaries without crashing.

---

## 4. Production Roadmap

Moving from Alpha to General Availability will require the following implementations:

1. **WebSocket Frontend Client**

   * **Implementation:** Developing a Next.js real-time web UI that connects to `ws://localhost:8000/ws/execute`. Utilizing Framer Motion and React context to decode the `event_callback` stream.
   * **Value:** Users will see beautiful micro-animations as each agent "starts" and "finishes", observing the internal reasoning JSON fly by in a hacker-esque terminal display rather than dealing with HTTP 504 timeouts on long LLM runs.

2. **Vector Store Pre-Caching (Pinecone/Redis)**

   * **Implementation:** Implementing a semantic interceptor layer in front of the LangGraph runtime. Standardized incoming queries are mapped using `text-embedding-3-small`.
   * **Value:** If a `similarity_score > 0.95` is hit against previously generated high-quality briefs (e.g. "lo-fi study beats"), the supervisor graph is bypassed entirely and the cached brief is yielded instantly, creating a 99% cost reduction and zero-latency response for common queries.

3. **MusicGen / Suno Pipeline Integration**

   * **Implementation:** Setting up an asyncio celery queue worker that watches for the `FINISH` state payload containing the JSON brief. The queue intercepts the JSON array strings, concatenates them into a raw prompt, and hits the Suno/Udio generation API via webhooks.
   * **Value:** The final JSON brief currently acts as the end-state text. Piping these parameters directly into an audio model unlocks the true capability of the system, transforming text queries dynamically into immediately playable `.wav` files.
