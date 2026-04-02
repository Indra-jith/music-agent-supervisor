# main.py
"""
Production FastAPI application for the Music Intelligence Agent System.

Fixes applied:
- Replaced deprecated @app.on_event with lifespan context manager
- Shared bounded ThreadPoolExecutor (prevents resource exhaustion)
- HTTP 422 for CLARIFY responses (proper REST semantics)
- Input sanitization to prevent prompt injection
- Request timeouts via asyncio.wait_for
- Atomic trace persistence
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from graph import execute_query, load_trace, validate_environment

# ---------------------------------------------------------------------------
# Configuration & Shared Resources
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 2000
MAX_WORKERS = 4  # Prevents thread exhaustion under load
REQUEST_TIMEOUT = 120  # Seconds
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazy initialization of shared thread pool."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="agent_worker")
    return _executor


# ---------------------------------------------------------------------------
# Lifespan Management (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources across requests."""
    # Startup
    try:
        validate_environment()
    except Exception as e:
        raise RuntimeError(f"Failed to start: {e}")
        
    _get_executor()  # Initialize executor
    try:
        yield
    finally:
        # Shutdown
        if _executor:
            _executor.shutdown(wait=True)


app = FastAPI(
    title="Music Intelligence Agent System",
    description=(
        "Production LangGraph supervisor system with bounded concurrency, "
        "request timeouts, and atomic trace persistence."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure strictly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Input Sanitization (Security)
# ---------------------------------------------------------------------------

def sanitize_query(query: str) -> str:
    """Prevent prompt injection and normalize input."""
    # Remove control characters
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
    query = query.strip()
    
    # Basic injection pattern detection
    suspicious_patterns = [
        r'ignore\s+(?:previous|all|above)\s+(?:instructions|prompts)',
        r'system\s+prompt',
        r'<\s*/\s*instruction\s*>',
        r'\{\s*\{\s*.*?\}\s*\}',  # Template injection attempts
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Query contains potentially malicious patterns")
    return query


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    query: str = Field(..., description="The music generation query to process.", max_length=MAX_QUERY_LENGTH)
    session_id: Optional[str] = Field(None, description="Optional session ID for trace retrieval.")
    max_iterations: int = Field(10, description="Maximum supervisor iterations.", ge=1, le=20)
    token_budget: int = Field(5000, description="Maximum token budget for the session.", ge=500, le=20000)

    @field_validator('query')
    @classmethod
    def validate_query_content(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")
        return sanitize_query(v)


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
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Apply global timeout to all requests."""
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timeout - query too complex or system overloaded"}
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive frontend UI."""
    return FileResponse("index.html")


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """
    Run a music generation query through the supervisor pipeline.
    """
    start_time = time.time()
    
    try:
        loop = asyncio.get_running_loop()
        
        # Use shared executor with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _get_executor(),
                lambda: execute_query(
                    query=request.query,
                    session_id=request.session_id,
                    max_iterations=request.max_iterations,
                    token_budget=request.token_budget,
                )
            ),
            timeout=REQUEST_TIMEOUT
        )
        
        # Check for CLARIFY - return 422 Unprocessable Entity per audit
        if result["final_answer"].get("status") == "clarification_needed":
            raise HTTPException(
                status_code=422,
                detail={
                    "message": result["final_answer"].get("message"),
                    "confidence_score": 0.0,
                    "session_id": result["session_id"]
                }
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
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query execution timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.get("/v1/trace/{session_id}")
async def get_trace(session_id: str):
    """Retrieve the full execution trace for a session from disk."""
    if not re.match(r'^[\w\-]+$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
        
    trace = load_trace(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace found for session: {session_id}")
    return {"session_id": session_id, "trace": trace, "steps": len(trace)}


@app.websocket("/ws/execute")
async def websocket_execute(websocket: WebSocket):
    """Real-time streaming execution via WebSocket."""
    await websocket.accept()
    loop = asyncio.get_running_loop()
    
    def event_callback(event_type: str, event_data: dict):
        """Thread-safe callback to stream events to WebSocket."""
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": event_type, **event_data}),
                loop
            )
        except Exception:
            pass
    
    try:
        # Set timeout for initial message
        data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        
        query = data.get("query", "")
        if not query:
            await websocket.send_json({"type": "error", "message": "query is required"})
            await websocket.close()
            return
            
        # Sanitize
        try:
            query = sanitize_query(query)
        except ValueError as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

        session_id = data.get("session_id")
        max_iterations = min(data.get("max_iterations", 10), 20)
        token_budget = min(data.get("token_budget", 5000), 20000)
        
        # Execute with callback
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _get_executor(),
                lambda: execute_query(
                    query=query,
                    session_id=session_id,
                    max_iterations=max_iterations,
                    token_budget=token_budget,
                    event_callback=event_callback
                )
            ),
            timeout=REQUEST_TIMEOUT
        )
        
        await websocket.send_json({"type": "final_answer", **result})
        await websocket.close(1000)
        
    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        try:
            await websocket.send_json({"type": "error", "message": "Timeout"})
            await websocket.close(1011)
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(1011)
        except Exception:
            pass


@app.get("/v1/evaluate")
async def evaluate():
    """Built-in self-evaluator with resource limits."""
    test_cases = [
        ("What BPM range is typical for lo-fi hip hop?", 3000),
        ("Music brief for a luxury EV brand ad targeting millennials — futuristic but warm", 6000),
        ("Background music for an 8-hour sleep stories feature in a meditation app", 4000),
        ("Make it sound cool", 2000),  # Ambiguous - should CLARIFY (422)
        ("Aggressive esports highlight music, no lyrics", 4000),
    ]
    
    async def run_test(q: str, budget: int):
        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            res = await asyncio.wait_for(
                loop.run_in_executor(
                    _get_executor(),
                    lambda: execute_query(q, None, 10, budget)
                ),
                timeout=30
            )
            return {
                "query": q,
                "latency_ms": int((time.time() - t0) * 1000),
                "agents_called": res.get("agents_called", []),
                "skipped_agents": res.get("skipped_agents", []),
                "token_usage": res.get("token_usage", {}),
                "final_answer": res.get("final_answer", {}),
                "status": "success"
            }
        except HTTPException as e:
            # Handle 422 CLARIFY as valid test result, not error
            if e.status_code == 422:
                return {
                    "query": q,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "status": "clarification_needed",
                    "message": e.detail.get("message") if isinstance(e.detail, dict) else str(e.detail)
                }
            return {
                "query": q,
                "latency_ms": int((time.time() - t0) * 1000),
                "error": str(e.detail),
                "status": "failed"
            }
        except Exception as e:
            return {
                "query": q,
                "latency_ms": int((time.time() - t0) * 1000),
                "error": str(e),
                "status": "failed"
            }

    # Run sequentially to avoid overwhelming the thread pool
    results = []
    for q, b in test_cases:
        results.append(await run_test(q, b))
        await asyncio.sleep(0.5)

    total_cost = sum(
        r.get("token_usage", {}).get("estimated_cost_usd", 0) 
        for r in results if "token_usage" in r
    )

    return {
        "test_cases": results,
        "total_cost_usd": round(total_cost, 6),
        "system_limits": {
            "max_workers": MAX_WORKERS,
            "request_timeout": REQUEST_TIMEOUT
        }
    }


@app.get("/health")
@app.get("/v1/health")
async def health():
    """Real health check that validates dependencies."""
    components = {
        "supervisor": "unknown",
        "groq_api": "unknown",
        "trace_storage": "unknown"
    }
    
    # Check Groq API
    try:
        import groq
        client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        client.models.list()
        components["groq_api"] = "operational"
        components["supervisor"] = "operational"
    except Exception as e:
        components["groq_api"] = f"unavailable: {str(e)}"
        components["supervisor"] = "degraded"
    
    # Check trace storage
    try:
        from graph import TRACES_DIR
        test_file = TRACES_DIR / ".healthcheck"
        test_file.write_text("test")
        test_file.unlink()
        components["trace_storage"] = "writable"
    except Exception as e:
        components["trace_storage"] = f"error: {str(e)}"
    
    overall = "healthy" if all(
        v in ["operational", "writable"] for v in components.values()
    ) else "degraded"
    
    return {
        "status": overall,
        "service": "Music Intelligence Agent System",
        "version": "2.0.0",
        "components": components,
        "config": {
            "max_workers": MAX_WORKERS,
            "request_timeout": REQUEST_TIMEOUT
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)