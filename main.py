# main.py
"""
FastAPI application for the Music Intelligence Agent System.

Endpoints:
  POST /v1/execute     — Run a music generation query through the supervisor
  GET  /v1/trace/{id}  — Retrieve persisted trace from disk
  GET  /v1/health      — System health check
"""

from __future__ import annotations

import asyncio
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph import execute_query, load_trace

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Music Intelligence Agent System",
    description=(
        "A LangGraph-powered supervisor system that dynamically routes between "
        "specialized music agents to produce structured music generation briefs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_check():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set — system cannot start")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    query: str = Field(..., description="The music generation query to process.")
    session_id: Optional[str] = Field(None, description="Optional session ID for trace retrieval.")
    max_iterations: Optional[int] = Field(10, description="Maximum supervisor iterations.", ge=1, le=30)
    token_budget: Optional[int] = Field(5000, description="Maximum token budget for the session.", ge=1000)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total: int
    estimated_cost_usd: float


class ExecuteResponse(BaseModel):
    session_id: str
    final_answer: dict
    execution_trace: list
    token_usage: TokenUsage
    iterations: int
    agents_called: list[str]
    skipped_agents: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Run a music generation query through the supervisor pipeline.

    The supervisor dynamically decides which agents to call based on
    what information is needed — not a fixed sequence.
    """
    try:
        result = execute_query(
            query=request.query,
            session_id=request.session_id,
            max_iterations=request.max_iterations or 10,
            token_budget=request.token_budget or 5000,
        )
        return ExecuteResponse(
            session_id=result["session_id"],
            final_answer=result["final_answer"],
            execution_trace=result["execution_trace"],
            token_usage=TokenUsage(**result["token_usage"]),
            iterations=result["iterations"],
            agents_called=result["agents_called"],
            skipped_agents=result["skipped_agents"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.get("/v1/trace/{session_id}")
async def get_trace(session_id: str):
    """Retrieve the full execution trace for a session from disk.

    Traces are persisted after every agent call — this endpoint reads
    from the filesystem, not memory. Survives API restarts.
    """
    trace = load_trace(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace found for session: {session_id}")
    return {"session_id": session_id, "trace": trace, "steps": len(trace)}


@app.websocket("/ws/execute")
async def websocket_execute(websocket: WebSocket):
    """Real-time streaming execution via WebSocket."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = data.get("query")
        if not query:
            await websocket.send_json({"type": "error", "message": "query is required"})
            await websocket.close()
            return

        session_id = data.get("session_id")
        max_iterations = data.get("max_iterations", 10)
        token_budget = data.get("token_budget", 5000)

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def event_callback(event_type: str, event_data: dict):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": event_type, **event_data})

        async def _consumer():
            while True:
                msg = await queue.get()
                if msg["type"] == "__DONE__":
                    break
                try:
                    await websocket.send_json(msg)
                except:
                    break

        consumer_task = asyncio.create_task(_consumer())

        def _run_graph():
            try:
                res = execute_query(query, session_id, max_iterations, token_budget, event_callback)
                event_callback("final_answer", res)
            except Exception as e:
                event_callback("error", {"message": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "__DONE__"})

        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, _run_graph)

        await consumer_task
        await websocket.close(1000)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()  # Log to standard output for debugging
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(1011)
        except:
            pass


@app.get("/v1/evaluate")
async def evaluate():
    """Built-in self-evaluator running standardized stress tests."""
    test_cases = [
        "What BPM range is typical for lo-fi hip hop?",
        "Music brief for a luxury EV brand ad targeting millennials — futuristic but warm",
        "Background music for an 8-hour sleep stories feature in a meditation app",
        "Make it sound cool",  # Ambiguous query - should clarify
        "A song with slow doom metal mood but 170 BPM upbeat tempo"  # Intentional conflict
    ]
    
    import time
    
    async def run_test(q: str, budget: int = 3000):
        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, execute_query, q, None, 10, budget)
            return {
                "query": q,
                "latency_ms": int((time.time() - t0) * 1000),
                "agents_called": res.get("agents_called", []),
                "skipped_agents": res.get("skipped_agents", []),
                "token_usage": res.get("token_usage", {}),
                "final_answer": res.get("final_answer", {}),
            }
        except Exception as e:
            return {
                "query": q,
                "latency_ms": int((time.time() - t0) * 1000),
                "error": str(e)
            }

    results = await asyncio.gather(*(run_test(q) for q in test_cases))

    total_cost = sum(r.get("token_usage", {}).get("estimated_cost_usd", 0) for r in results if "token_usage" in r)

    return {
        "test_cases": list(results),
        "total_cost_usd": round(total_cost, 6)
    }


@app.get("/health")
@app.get("/v1/health")
async def health():
    """System health check."""
    return {
        "status": "healthy",
        "service": "Music Intelligence Agent System",
        "version": "1.0.0",
        "components": {
            "supervisor": "operational",
            "music_researcher": "available",
            "trend_analyst": "available",
            "prompt_strategist": "available",
            "trace_persistence": "file-based",
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
