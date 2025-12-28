"""
api_server.py
FastAPI server for MADDUX Dashboard chat functionality.

Provides:
- Static file serving for dashboard
- /api/query endpoint for Claude-powered natural language queries
- /api/stats endpoint for dashboard statistics

The API key is provided by the user per-request (not stored server-side).
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase
from scripts.claude_query import execute_query_with_key, get_database_stats

# Paths
BASE_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
DATABASE_PATH = BASE_DIR / "database" / "maddux.db"

# FastAPI app
app = FastAPI(
    title="MADDUX Analytics API",
    description="Natural language query interface for MADDUX baseball analytics",
    version="1.0.0"
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for /api/query endpoint."""
    question: str
    api_key: str


class QueryResponse(BaseModel):
    """Response model for /api/query endpoint."""
    answer: str
    data: Optional[dict] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Response model for /api/stats endpoint."""
    total_players: int
    total_seasons: int
    years_covered: str
    model_correlation: float
    model_hit_rate: float
    model_r_squared: float
    top_projections: list


@app.get("/")
async def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard/")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MADDUX API"}


@app.get("/api/stats")
async def get_stats():
    """
    Get dashboard statistics (no API key required).
    Returns database summary and model performance metrics.
    """
    try:
        db = MadduxDatabase(str(DATABASE_PATH))
        stats = get_database_stats(db)
        db.close()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    """
    Execute a natural language query against the MADDUX database.
    
    Requires:
    - question: The natural language question to ask
    - api_key: User's Anthropic API key (starts with sk-ant-)
    
    Returns:
    - answer: Claude's response to the question
    - data: Optional structured data (tables, lists)
    - error: Error message if query failed
    """
    # Validate API key format
    if not request.api_key or not request.api_key.startswith("sk-ant-"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid API key format. Key should start with 'sk-ant-'"
        )
    
    # Validate question
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Question is too short. Please ask a complete question."
        )
    
    try:
        # Connect to database
        db = MadduxDatabase(str(DATABASE_PATH))
        
        # Execute query with user's API key
        result = execute_query_with_key(
            db=db,
            question=request.question,
            api_key=request.api_key
        )
        
        db.close()
        
        return QueryResponse(
            answer=result.get("answer", ""),
            data=result.get("data"),
            error=result.get("error")
        )
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle common errors with friendly messages
        if "authentication" in error_msg.lower() or "invalid api key" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Please check your Anthropic API key and try again."
            )
        elif "rate limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=error_msg)


# Mount static files for dashboard (must be after API routes)
app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


def main():
    """Run the API server."""
    import uvicorn
    
    print("=" * 60)
    print("MADDUX Analytics API Server")
    print("=" * 60)
    print(f"\nDashboard: http://localhost:8000/dashboard/")
    print(f"API Docs:  http://localhost:8000/docs")
    print(f"Health:    http://localhost:8000/api/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "scripts.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()

