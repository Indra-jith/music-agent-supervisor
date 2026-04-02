# Music Agent Supervisor

A **non-deterministic, multi-agent AI system** that generates structured, technically accurate Music Generation Briefs from natural language queries. Powered by **FastAPI** and **LangGraph**, a Supervisor node reasons about state gaps in real time and dynamically routes between three specialist agents — no hardcoded pipeline, no fixed sequence.
---

## Live Demo

**API (Swagger UI):** [https://music-agent-supervisor.onrender.com/docs](https://music-agent-supervisor.onrender.com/docs)

**Frontend UI:** [https://music-agent-supervisor.onrender.com](https://music-agent-supervisor.onrender.com)

**Health check:** [https://music-agent-supervisor.onrender.com/health](https://music-agent-supervisor.onrender.com/health)

Try it immediately:

```bash
curl -X POST https://music-agent-supervisor.onrender.com/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"query": "peak-hour techno set for a Berlin club", "token_budget": 5000}'
```

---

## System Architecture

Instead of a rigid A → B → C pipeline, every request starts at the **Supervisor** — an LLM call that reads the current state, identifies gaps, and decides which agent to invoke next. Agents never call each other directly. Every transition goes back through the Supervisor.

```
          ┌─────────────────────────────┐
          │         SUPERVISOR          │
          │  Reasons about state gaps,  │
          │  token budget, and routing  │
          └──────┬────────┬─────────────┘
                 │        │
       ┌─────────▼─┐   ┌──▼──────────┐   ┌──────────────────┐
       │  music_   │   │   trend_    │   │    prompt_       │
       │ researcher│   │  analyst   │   │   strategist     │
       │           │   │            │   │                  │
       │ DuckDuckGo│   │ ArXiv      │   │ JSON validation  │
       │ Wikipedia │   │ DuckDuckGo │   │ + synthesis      │
       └─────────┬─┘   └──┬─────────┘   └──────────┬───────┘
                 │        │                          │
                 └────────┴──────────────────────────┘
                          Returns to Supervisor
```

### The four nodes

**Supervisor** — The orchestrator. Produces structured JSON reasoning (`identified_gaps`, `decision`, `expected_contribution`) before every routing choice. Tracks token budget and call counts to enforce cost-aware behaviour.

**music_researcher** — Researches genre characteristics, mood associations, cultural context, and reference tracks using DuckDuckGo and Wikipedia. Falls back to Wikipedia when DuckDuckGo rate-limits.

**trend_analyst** — Finds technical parameters: BPM range, musical key, time signature, instrumentation, and energy contour. Uses ArXiv for AI music generation research and DuckDuckGo for market data.

**prompt_strategist** — Synthesises all gathered context into a validated JSON brief. Runs Python-based JSON validation internally to catch and repair malformed outputs before returning.

---

## The Debate Protocol

The system's architectural defence against silent hallucination.

In multi-agent systems, agents pulling from different sources frequently produce contradictory outputs. Example: `music_researcher` establishes a Doom Metal track at 60 BPM; `trend_analyst` returns 170 BPM for "energetic pacing." A naive system silently blends these into an unplayable brief.

**Wubble's approach:**

1. `prompt_strategist` detects the conflict during synthesis
2. Instead of producing the brief, it returns `{"contradiction": "explanation of the conflict"}`
3. The Supervisor reads `contradiction_detected` in state and re-routes both agents
4. Agents receive: *"DEFEND YOUR POSITION: A contradiction was detected. Provide stronger evidence or concede."*
5. Agents cite sources, one concedes, `prompt_strategist` synthesises with the resolved data

See [`debate_proof.json`](./debate_proof.json) for a raw state dump of a live contradiction resolution cycle.

---

## Cost-Aware Routing

Every request includes a `token_budget`. The Supervisor tracks cumulative token consumption (using `tiktoken` with `cl100k_base` encoding) after every agent call and injects the remaining budget into its reasoning prompt before each routing decision.

- If budget is sufficient: full pipeline
- If budget is low: Supervisor skips expensive agents (`trend_analyst` ~1500 tokens, `music_researcher` ~1000 tokens) and routes directly to `prompt_strategist`
- If budget is exhausted mid-run: graph terminates immediately

The response always includes `skipped_agents` and `token_usage.estimated_cost_usd` for full transparency.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/execute` | Run a query through the supervisor pipeline |
| `WS` | `/ws/execute` | Stream agent events in real time |
| `GET` | `/v1/evaluate` | Run 5 concurrent stress tests |
| `GET` | `/v1/trace/{session_id}` | Retrieve full execution trace from disk |
| `GET` | `/health` | Service health + component status |
| `GET` | `/docs` | Interactive Swagger UI |

### Request schema (`POST /v1/execute`)

```json
{
  "query": "futuristic luxury EV brand ad targeting millennials",
  "session_id": "optional-custom-id",
  "token_budget": 5000,
  "max_iterations": 10
}
```

### Response schema

```json
{
  "session_id": "abc-123",
  "final_answer": {
    "use_case": "luxury automotive advertisement",
    "mood_tags": ["cinematic", "futuristic", "minimal", "warm"],
    "genre": "Synth-Wave",
    "bpm": { "min": 90, "max": 110 },
    "key": "F# minor",
    "time_signature": "4/4",
    "instrumentation": ["synthesizer pads", "clean electric guitar", "sub bass", "brushed drums"],
    "energy_level": "dynamic",
    "duration_seconds": 60,
    "reference_tracks": ["Tycho - Awake", "Com Truise - Galactic Melt"],
    "generation_notes": "Avoid aggressive distortion. Emphasise space and restraint.",
    "confidence_score": 0.82,
    "gaps": []
  },
  "agents_called": ["music_researcher", "trend_analyst", "prompt_strategist"],
  "skipped_agents": [],
  "token_usage": {
    "input_tokens": 3241,
    "output_tokens": 812,
    "total": 4053,
    "estimated_cost_usd": 0.002554
  },
  "iterations": 4,
  "execution_trace": [...]
}
```

### WebSocket streaming (`WS /ws/execute`)

Connect and send:
```json
{ "query": "Berlin techno", "token_budget": 5000 }
```

Receive a stream of events as they happen:
```
{"type": "supervisor_start", "iteration": 1}
{"type": "supervisor_end", "decision": "trend_analyst", "reasoning": {...}}
{"type": "agent_start", "agent": "trend_analyst"}
{"type": "agent_end", "agent": "trend_analyst", "quality": "good"}
{"type": "agent_start", "agent": "prompt_strategist"}
{"type": "agent_end", "agent": "prompt_strategist", "quality": "good"}
{"type": "final_answer", ...}
```

---

## Running Locally

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/Indra-jith/music-agent-supervisor.git
cd music-agent-supervisor
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
PERSIST_TRACES=true
```

Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API interactively.

To enable file-based trace persistence (survives restarts):
```
PERSIST_TRACES=true
```

Traces are written to `/traces/{session_id}.json` after every agent call.

---

## Evaluation

Run the built-in stress test suite (5 queries, concurrent):

```bash
curl https://music-agent-supervisor.onrender.com/v1/evaluate
```

This runs 5 queries concurrently and returns routing paths, latency, token usage, and cost for each. See [`EVAL.md`](./EVAL.md) for full analysis of routing behaviour, failure modes, and optimisation decisions.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama-3.3-70b-versatile via Groq API |
| Orchestration | LangGraph (StateGraph, conditional edges) |
| API Framework | FastAPI + Uvicorn |
| Token Counting | tiktoken (cl100k_base) |
| Tools | DuckDuckGo Search, Wikipedia, ArXiv, Python REPL |
| Deployment | Render (Procfile-based, ephemeral-safe) |

---

## Repository Structure

```
├── main.py          # FastAPI app — endpoints, WebSocket, evaluate
├── graph.py         # LangGraph supervisor + all agent definitions
├── requirements.txt # Dependencies
├── EVAL.md          # Stress test analysis, routing decisions, failure modes
├── debate_proof.json # Raw state dump from a live Debate Protocol cycle
├── Procfile         # Render deployment config
└── traces/          # Session trace files (when PERSIST_TRACES=true)
```

---

*Built by Indrajith MP as part of the Wubble.ai Agentic AI Systems internship assessment.*
