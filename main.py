# ==============================================================================
# main.py (Updated for React Frontend Compatibility)
# ==============================================================================

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import redis
from fastapi_cache2 import FastAPICache, RedisBackend, cache
import httpx
from functools import wraps
from fastapi_cache2.decorator import cache as cache_decorator

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Add for security, rate limiting, logging
import uuid
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import json

# Your existing services
from services.search_service import GenomicsSearchService
from services.rag_service import GenomicsRAGService, RAGResponse, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Security: API Key Auth
# =====================
def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

# =====================
# Monitoring: Structured Logging (JSON)
# =====================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None)
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger()
logger.handlers = []  # Remove default handlers
logger.addHandler(handler)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# =====================
# Rate Limiting
# =====================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ==============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ==============================================================================

class QueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=500) = Field(..., description="Search query or question")
    model: str = Field(default="gpt-4", description="LLM model to use")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Optional filters
    journal: Optional[str] = Field(None, description="Filter by journal name")
    author: Optional[str] = Field(None, description="Filter by author name")
    year_start: Optional[int] = Field(None, description="Start year for filtering")
    year_end: Optional[int] = Field(None, description="End year for filtering")
    min_citations: Optional[int] = Field(None, description="Minimum citation count")
    chunk_type: Optional[str] = Field(None, description="Filter by chunk type")
    keywords: Optional[List[str]] = Field(None, description="Filter by keywords")

class SearchOnlyRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=500) = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results to return")
    
    # Same filters as QueryRequest
    journal: Optional[str] = None
    author: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    min_citations: Optional[int] = None
    chunk_type: Optional[str] = None
    keywords: Optional[List[str]] = None

class VectorMatch(BaseModel):
    id: str
    score: float
    content: str
    title: str
    source: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    query: str
    matches: List[VectorMatch]
    llm_response: str
    model_used: str
    num_sources: int
    response_time_ms: int
    filters_applied: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    matches: List[VectorMatch]
    num_results: int
    response_time_ms: int
    filters_applied: Dict[str, Any]

class StatusResponse(BaseModel):
    status: str
    index_stats: Dict[str, Any]
    available_models: List[str]
    timestamp: str
    # Add these fields for better frontend integration
    version: str = "1.0.0"
    environment: str = "production"

# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================

app = FastAPI(
    title="Genomics RAG API",
    description="Vector search and LLM-powered Q&A for genomics research",
    version="1.0.0"
)

# CORS middleware - Updated for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Your backend
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "*"  # Allow all for production (configure as needed)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Redis cache setup (init in startup)
redis_client = None

# Global services
search_service: Optional[GenomicsSearchService] = None
rag_service: Optional[GenomicsRAGService] = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def build_filters(request) -> Dict[str, Any]:
    """Build Pinecone filters from request parameters"""
    filters = {}
    
    # Journal filter
    if request.journal:
        filters["$or"] = [
            {"journal": {"$eq": request.journal}},
            {"crossref_journal": {"$eq": request.journal}}
        ]
    
    # Author filter
    if request.author:
        filters["authors"] = {"$in": [request.author]}
    
    # Year range filter
    if request.year_start or request.year_end:
        year_start = request.year_start or 1900
        year_end = request.year_end or datetime.now().year
        
        year_filter = {
            "$or": [
                {"publication_year": {"$gte": year_start, "$lte": year_end}},
                {"crossref_year": {"$gte": year_start, "$lte": year_end}}
            ]
        }
        
        if filters.get("$or"):
            filters["$and"] = [
                {"$or": filters.pop("$or")},
                year_filter
            ]
        else:
            filters.update(year_filter)
    
    # Citation filter
    if request.min_citations:
        filters["citation_count"] = {"$gte": request.min_citations}
    
    # Chunk type filter
    if request.chunk_type:
        filters["chunk_type"] = {"$eq": request.chunk_type}
    
    # Keywords filter
    if request.keywords:
        filters["keywords"] = {"$in": request.keywords}
    
    return {k: v for k, v in filters.items() if v is not None}

# ==============================================================================
# STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global search_service, rag_service, redis_client
    
    try:
        logger.info("🚀 Initializing Genomics RAG API...")
        
        # Check required environment variables
        required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Initialize search service
        openai_api_key = os.getenv('OPENAI_API_KEY')
        search_service = GenomicsSearchService(openai_api_key=openai_api_key)
        logger.info("✅ Search service initialized")
        
        # Initialize RAG service
        rag_service = GenomicsRAGService(
            openai_api_key=openai_api_key
        )
        logger.info("✅ RAG service initialized")
        
        # Test connection
        stats = search_service.get_search_statistics()
        logger.info(f"📊 Vector store stats: {stats.get('total_vectors', 0)} vectors")
        # Redis setup
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
        logger.info("✅ Redis cache initialized")
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# Health check with dependency checks
@app.get("/health")
async def health_check():
    health = {"status": "healthy", "timestamp": datetime.now().isoformat(), "service": "Genomics RAG API", "version": "1.0.0"}
    # Check OpenAI
    try:
        # Use httpx for connection pooling and timeout
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.post(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            )
            health["openai"] = resp.status_code == 200
    except Exception as e:
        health["openai"] = False
        health["openai_error"] = str(e)
    # Check Pinecone (TODO: add real check if possible)
    try:
        # Placeholder: real check should ping Pinecone
        health["pinecone"] = True
    except Exception as e:
        health["pinecone"] = False
        health["pinecone_error"] = str(e)
    # Check Redis
    try:
        if redis_client and redis_client.ping():
            health["redis"] = True
        else:
            health["redis"] = False
    except Exception as e:
        health["redis"] = False
        health["redis_error"] = str(e)
    return health

# Response time logging middleware
@app.middleware("http")
async def log_response_time(request: Request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = int((time.time() - start) * 1000)
    logger.info(json.dumps({"path": request.url.path, "method": request.method, "duration_ms": duration, "request_id": getattr(request.state, 'request_id', None)}))
    response.headers["X-Response-Time-ms"] = str(duration)
    return response

# Retry logic and connection pooling for OpenAI/Pinecone (TODO: implement in service layer)
# TODO: Add circuit breaker pattern for external dependencies

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API status and statistics"""
    try:
        stats = {}
        
        # Check if search service is initialized
        if search_service:
            try:
                stats = search_service.get_search_statistics()
            except Exception as e:
                logger.error(f"Search service error: {e}")
                stats = {"error": f"Search service error: {str(e)}"}
        
        # Updated available models list for React frontend
        available_models = [
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "mistral-large"
        ]
        
        return StatusResponse(
            status="healthy" if search_service and rag_service else "degraded",
            index_stats=stats,
            available_models=available_models,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            environment=os.getenv('ENVIRONMENT', 'production')
        )
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        
        # Return a degraded status instead of crashing
        return StatusResponse(
            status="error",
            index_stats={"error": str(e)},
            available_models=[],
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            environment=os.getenv('ENVIRONMENT', 'production')
        )

@app.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
@cache_decorator(expire=60, namespace="search")  # Cache for 60s
async def search_only(request: SearchOnlyRequest, api_key: str = Depends(get_api_key)):
    """Vector search only (no LLM)"""
    start_time = datetime.now()
    
    try:
        if not search_service:
            raise HTTPException(status_code=500, detail="Search service not initialized")
        
        # Build filters
        filters = build_filters(request)
        
        logger.info(f"🔍 Vector search: '{request.query[:50]}...'")
        
        # Perform search
        chunks = search_service.search_similar_chunks(
            query_text=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Format matches
        matches = []
        for chunk in chunks:
            match = VectorMatch(
                id=chunk['id'],
                score=chunk['score'],
                content=chunk['content'],
                title=chunk['title'],
                source=chunk['source'],
                metadata=chunk['metadata']
            )
            matches.append(match)
        
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response = SearchResponse(
            query=request.query,
            matches=matches,
            num_results=len(matches),
            response_time_ms=response_time,
            filters_applied=filters
        )
        
        logger.info(f"✅ Search completed in {response_time}ms with {len(matches)} results")
        return response
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/query", response_model=RAGResponse)
@limiter.limit("20/minute")
@cache_decorator(expire=60, namespace="query")  # Cache for 60s
async def query_with_llm(request: QueryRequest, api_key: str = Depends(get_api_key)):
    """Main endpoint: Vector search + LLM response"""
    start_time = datetime.now()
    
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")
        
        # Build filters
        filters = build_filters(request)
        
        logger.info(f"🤖 LLM Query: '{request.query[:50]}...'")
        
        # Handle model switching more gracefully
        try:
            if hasattr(rag_service.llm, 'model') and request.model != rag_service.llm.model:
                from langchain_openai import ChatOpenAI
                rag_service.llm = ChatOpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model=request.model,
                    temperature=request.temperature
                )
                logger.info(f"🔄 Switched to model: {request.model}")
        except Exception as model_error:
            logger.warning(f"Model switch failed, using default: {model_error}")
        
        # Get RAG response
        rag_response = rag_service.ask_question(
            question=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Cost Tracking: Log OpenAI token usage if available
        if hasattr(rag_response, 'metadata') and 'usage' in rag_response.metadata:
            logger.info(f"OpenAI tokens used: {rag_response.metadata['usage']}")
        
        # Format matches - improved error handling
        matches = []
        sources = rag_response.sources
        
        for i, source in enumerate(sources):
            try:
                match = VectorMatch(
                    id=source.get('id', f"source_{i}"),  # Better ID handling
                    score=float(source.get('relevance_score', 0.0)),
                    content=source.get('content_preview', ''),
                    title=source.get('title', 'Unknown Title'),
                    source=source.get('source_file', 'Unknown Source'),
                    metadata={
                        'journal': source.get('journal'),
                        'year': source.get('year'),
                        'authors': source.get('authors', []),
                        'doi': source.get('doi'),
                        'citation_count': source.get('citation_count', 0),
                        'chunk_type': source.get('chunk_type'),
                        'chunk_index': source.get('chunk_index')
                    }
                )
                matches.append(match)
            except Exception as match_error:
                logger.warning(f"Error formatting match {i}: {match_error}")
                continue
        
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response = RAGResponse(
            query=request.query,
            matches=matches,
            llm_response=rag_response.answer,
            model_used=request.model,
            num_sources=len(matches),
            response_time_ms=response_time,
            filters_applied=filters
        )
        
        logger.info(f"✅ Query completed in {response_time}ms with {len(matches)} sources")
        return response
        
    except Exception as e:
        logger.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==============================================================================
# SPECIALIZED ENDPOINTS
# ==============================================================================

@app.post("/query/methods")
async def query_methods(request: QueryRequest):
    """Ask questions focused on methodology sections"""
    request.chunk_type = "methods"
    return await query_with_llm(request)

@app.post("/query/results")
async def query_results(request: QueryRequest):
    """Ask questions focused on results sections"""
    request.chunk_type = "results"
    return await query_with_llm(request)

@app.post("/query/abstracts")
async def query_abstracts(request: QueryRequest):
    """Ask questions focused on abstracts only"""
    request.chunk_type = "abstract"
    return await query_with_llm(request)

@app.post("/query/high-impact")
async def query_high_impact(request: QueryRequest):
    """Query high-impact papers (high citation count)"""
    request.min_citations = request.min_citations or 20
    return await query_with_llm(request)

@app.post("/query/recent")
async def query_recent(request: QueryRequest):
    """Query recent papers (last 3 years)"""
    current_year = datetime.now().year
    request.year_start = current_year - 3
    request.year_end = current_year
    return await query_with_llm(request)

# ==============================================================================
# NEW ENDPOINTS FOR REACT FRONTEND
# ==============================================================================

@app.get("/models")
async def get_available_models():
    """Get list of available models for React frontend"""
    return {
        "models": [
            {"value": "gpt-4", "label": "GPT-4", "description": "Most capable model"},
            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "description": "Faster GPT-4 variant"},
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "description": "Fast and cost-effective"},
            {"value": "claude-3-opus", "label": "Claude 3 Opus", "description": "Anthropic's most capable model"},
            {"value": "claude-3-sonnet", "label": "Claude 3 Sonnet", "description": "Balanced performance"},
            {"value": "mistral-large", "label": "Mistral Large", "description": "European AI model"}
        ],
        "default": "gpt-4"
    }

@app.get("/filters/options")
async def get_filter_options():
    """Get available filter options for React frontend"""
    try:
        # You could enhance this to get actual values from your vector store
        return {
            "chunk_types": [
                {"value": "abstract", "label": "Abstract"},
                {"value": "methods", "label": "Methods"},
                {"value": "results", "label": "Results"},
                {"value": "discussion", "label": "Discussion"},
                {"value": "content", "label": "General Content"}
            ],
            "journals": [
                "Nature", "Science", "Cell", "Nature Medicine", "Nature Genetics",
                "Journal of Molecular Biology", "Molecular Metabolism", "Diabetes"
            ],
            "year_range": {
                "min": 2000,
                "max": datetime.now().year
            }
        }
    except Exception as e:
        logger.error(f"Filter options error: {e}")
        return {"error": str(e)}

# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Global exception: {exc}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": str(exc) if os.getenv('ENVIRONMENT') != 'production' else "An error occurred"
        }
    )

# ==============================================================================
# RUN THE SERVER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    is_production = os.getenv('ENVIRONMENT') == 'production'
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        reload=not is_production,
        log_level="info",
        access_log=True
    )
