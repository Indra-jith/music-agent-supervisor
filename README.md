# Music Agent Supervisor <br/>

Music Agent Supervisor is a non-deterministic, multi-agent AI system designed to generate highly specific, technically accurate **Music Generation Briefs**. Powered by **FastAPI** and **LangGraph**, it coordinates a team of specialized AI agents to research, debate, and synthesize validated JSON music briefs from natural language queries (e.g., *"futuristic luxury EV brand ad"*).

---

## ⚡ Quick Start

You can immediately test the live API via `curl`:

```bash
curl -X POST https://your-app.onrender.com/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"query":"upbeat electronic ad for Nike","token_budget":3000}'
```
*(Note: Replace `your-app.onrender.com` with the actual Render deployment URL.)*

Alternatively, visit the interactive Swagger UI:
[Live API Documentation](https://your-app.onrender.com/docs)

---

## 🧠 System Architecture

At its core, Wubble departs from traditional, rigid LLM pipelines. Instead of sequential steps (Agent A -> Agent B -> Output), it uses a **dynamic, non-deterministic graph** supervised by a master orchestrator node.

The graph consists of four main nodes:

1. **The Supervisor Node**
   * The "brain" of the operation. It assesses the current context, the remaining token budget, and the user's query to decide the next best action dynamically.
   * **Cost-Aware Routing:** Wubble is aware of its `token_budget`. If a query is simple or the budget drops too low, the Supervisor bypasses expensive web-searching agents and forces early synthesis, maximizing efficiency.

2. **Music Researcher Agent**
   * Computes cultural and stylistic context. Uses DuckDuckGo and Wikipedia to find associated genres, mood descriptors, and real-world reference tracks.

3. **Trend Analyst Agent**
   * Computes technical parameters. Uses DuckDuckGo and ArXiv to pull data on min/max BPM, keys, time signatures, instrumentation, and energy contours.

4. **Prompt Strategist Agent**
   * Synthesizes all gathered intelligence into a validated JSON schema. It uses a Python REPL validator internally to auto-correct malformed JSON blobs before emitting them.

---

## ⚔️ The Debate Protocol (Addressing Hallucinations)

Wubble's most powerful feature is its architectural defense against AI Hallucination: **The Debate Protocol**.

In multi-agent systems, agents frequently pull from divergent sources resulting in contradictions. For example, the `Music Researcher` might establish a "Doom Metal" track should be 60 BPM, while the `Trend Analyst` pulling for "Upbeat pacing" locks onto 170 BPM.

In Wubble:
1. The `Prompt Strategist` begins generating the JSON but detects a conflict.
2. It interrupts compilation and returns a contradiction state.
3. The `Supervisor` loops the query back to the disparate agents.
4. The agents are issued a prompt injunction: **"DEFEND YOUR POSITION: A contradiction was detected... provide stronger evidence or concede."**
5. The agents self-correct and one concedes. The system then safety finishes compiling the brief.

---

## 🛠️ Tech Stack & Deployment

- **Backend Context:** FastAPI, Uvicorn 
- **LLM Orchestration:** LangGraph, Langchain
- **Language Models:** Llama-3.3-70b-versatile via Groq API
- **Deployment Strategy:** Platform Agnostic w/ Procfile tooling. Fully capable of running ephemerally without disk reliance (configurable `PERSIST_TRACES=false` environment variables for Render compatibility). 

## 🌐 Endpoints Overview

- `POST /v1/execute` : Execute a generation query.
- `GET /v1/trace/{session_id}` : Retrieve specific trace pathways on persistent disk models.
- `GET /v1/evaluate` : A built-in concurrent suite of 5 stress-tests to test the routing resilience.
- `WS /ws/execute` : Real-time Streaming WebSocket connection that publishes agent events as they happen for high-fidelity frontends.
