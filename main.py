# ==============================================================================
# main.py (Updated with Auth0 Authentication)
# ==============================================================================

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import redis
# from fastapi_cache import FastAPICache, RedisBackend, cache
import httpx
from functools import wraps
# from fastapi_cache.decorator import cache as cache_decorator

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Add for security, rate limiting, logging, caching
import uuid
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import json
from functools import lru_cache

# Your existing services
from services.search_service import GenomicsSearchService
from services.rag_service import GenomicsRAGService, RAGResponse, RAGConfig

# Auth0 authentication - TEMPORARILY DISABLED
# from auth.auth0_middleware import (
#     get_current_user, 
#     get_user_permissions, 
#     require_permission,
#     rate_limit_per_user
# )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Auth0 Security Configuration - TEMPORARILY DISABLED
# =====================
# def get_current_user_optional(request: Request):
#     """Get current user if authenticated, otherwise return None"""
#     try:
#         auth_header = request.headers.get('Authorization')
#         if auth_header and auth_header.startswith('Bearer '):
#             token = auth_header.split(' ')[1]
#             from auth.auth0_middleware import get_current_user, get_token_auth_header
#             return get_current_user(token)
#     except Exception:
#         pass
#     return None

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

# =====================
# Rate Limiting (will be configured after app creation)
# =====================
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# ==============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ==============================================================================

class QueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=500) = Field(..., description="Search query or question")
    model: str = Field(default="gpt-4o", description="LLM model to use")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Frontend filters object
    filters: Optional[Dict[str, Any]] = Field(None, description="Frontend filter object")
    
    # Optional individual filters (for backward compatibility)
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

class UserInfo(BaseModel):
    sub: str
    email: str
    name: str
    picture: Optional[str] = None
    permissions: List[str] = []

# ==============================================================================
# FASTAPI APP INITIALIZATION
# ==============================================================================

app = FastAPI(
    title="Diabetes Research Assistant API",
    description="Secure API for diabetes research with Auth0 authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =====================
# CORS Configuration
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://your-frontend-domain.com"  # Add your production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Gzip Compression
# =====================
app.add_middleware(GZipMiddleware, minimum_size=1000)

# =====================
# Rate Limiting Middleware
# =====================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # TEMPORARILY DISABLED RATE LIMITING
    response = await call_next(request)
    return response
    
    # Get user ID for rate limiting
    # user = get_current_user_optional(request)
    # user_id = user.get('sub', 'anonymous') if user else 'anonymous'
    
    # Apply rate limiting per user
    # try:
    #     rate_limit_per_user(user_id, max_requests=100, window_minutes=60)
    # except HTTPException as e:
    #     return JSONResponse(
    #         status_code=e.status_code,
    #         content={"detail": e.detail}
    #     )
    
    # response = await call_next(request)
    # return response

# =====================
# Request ID Middleware
# =====================
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to log record
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None
    )
    record.request_id = request_id
    logger.handle(record)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def build_filters(request) -> Dict[str, Any]:
    """Build filters from request parameters"""
    filters = {}
    
    # Handle frontend filters object
    if hasattr(request, 'filters') and request.filters:
        frontend_filters = request.filters
        
        # Map frontend filter names to backend filter names
        if frontend_filters.get('contentType') and frontend_filters['contentType'] != 'content':
            filters['chunk_type'] = frontend_filters['contentType']
        
        if frontend_filters.get('timePeriod') and frontend_filters['timePeriod'] != 'all':
            # Convert time period to year range
            current_year = datetime.now().year
            if frontend_filters['timePeriod'] == 'recent':
                filters['year_start'] = current_year - 2
                filters['year_end'] = current_year
            elif frontend_filters['timePeriod'] == '5year':
                filters['year_start'] = current_year - 5
                filters['year_end'] = current_year
            elif frontend_filters['timePeriod'] == 'decade':
                filters['year_start'] = current_year - 10
                filters['year_end'] = current_year
        
        if frontend_filters.get('citationLevel') and frontend_filters['citationLevel'] != 'all':
            # Convert citation level to min citations
            if frontend_filters['citationLevel'] == 'high':
                filters['min_citations'] = 50
            elif frontend_filters['citationLevel'] == 'medium':
                filters['min_citations'] = 10
            elif frontend_filters['citationLevel'] == 'emerging':
                filters['min_citations'] = 1
    
    # Handle individual filters (backward compatibility)
    if hasattr(request, 'journal') and request.journal:
        filters['journal'] = request.journal
    if hasattr(request, 'author') and request.author:
        filters['author'] = request.author
    if hasattr(request, 'year_start') and request.year_start:
        filters['year_start'] = request.year_start
    if hasattr(request, 'year_end') and request.year_end:
        filters['year_end'] = request.year_end
    if hasattr(request, 'min_citations') and request.min_citations:
        filters['min_citations'] = request.min_citations
    if hasattr(request, 'chunk_type') and request.chunk_type:
        filters['chunk_type'] = request.chunk_type
    if hasattr(request, 'keywords') and request.keywords:
        filters['keywords'] = request.keywords
    
    return filters

# ==============================================================================
# GLOBAL SERVICES
# ==============================================================================

# Initialize services
search_service = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global search_service, rag_service
    
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize search service with API key
        search_service = GenomicsSearchService(openai_api_key=openai_api_key)
        logger.info("✅ Search service initialized")
        
        # Initialize RAG service
        rag_service = GenomicsRAGService()
        logger.info("✅ RAG service initialized")
        
        # Test services
        if search_service and rag_service:
            logger.info("✅ All services ready")
        else:
            logger.error("❌ Service initialization failed")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

# ==============================================================================
# HEALTH & STATUS ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "search": search_service is not None,
            "rag": rag_service is not None
        }
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Detailed status endpoint"""
    try:
        # Get index stats if available
        index_stats = {}
        if search_service and hasattr(search_service, 'vector_store'):
            try:
                index_stats = search_service.vector_store.get_index_stats()
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
                index_stats = {"error": "Could not retrieve stats"}
        
        return StatusResponse(
            status="operational",
            index_stats=index_stats,
            available_models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

# ==============================================================================
# RESPONSE TIME LOGGING MIDDLEWARE
# ==============================================================================

@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Request processed in {process_time:.3f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ==============================================================================
# MAIN API ENDPOINTS (WITH AUTHENTICATION)
# ==============================================================================

@app.post("/search", response_model=SearchResponse)
async def search_only(
    request: SearchOnlyRequest, 
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Search-only endpoint (requires authentication)"""
    start_time = datetime.now()
    
    try:
        if not search_service:
            raise HTTPException(status_code=500, detail="Search service not initialized")
        
        # Build filters
        filters = build_filters(request)
        
        logger.info(f"🔍 Search: '{request.query[:50]}...' by user {current_user.get('email', 'unknown')}")
        
        # Perform search
        search_results = search_service.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Format matches with deduplication
        matches = []
        seen_papers = set()  # Track unique papers
        
        for i, result in enumerate(search_results):
            # Create a unique paper identifier
            paper_id = result.get('source', '') or result.get('title', 'Unknown')
            
            # Skip if we've already seen this paper
            if paper_id in seen_papers:
                continue
            
            seen_papers.add(paper_id)
            
            match = VectorMatch(
                id=result.get('id', f"result_{i}"),
                score=float(result.get('score', 0.0)),
                content=result.get('content', ''),
                title=result.get('title', 'Unknown Title'),
                source=result.get('source', 'Unknown Source'),
                metadata={
                    **result.get('metadata', {}),
                    'paper_id': paper_id  # Add paper ID for frontend reference
                }
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
        logger.exception("Error in /search endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/query", response_model=RAGResponse)
async def query_with_llm(
    request: QueryRequest, 
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Main endpoint: Vector search + LLM response (requires authentication)"""
    start_time = datetime.now()
    
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")
        
        # Build filters
        filters = build_filters(request)
        
        logger.info(f"🤖 LLM Query: '{request.query[:50]}...' by user unknown")
        
        # Handle model switching more gracefully with performance optimizations
        try:
            if hasattr(rag_service.llm, 'model') and request.model != rag_service.llm.model:
                from langchain_openai import ChatOpenAI
                rag_service.llm = ChatOpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=800,  # Reduced for faster responses
                    request_timeout=25  # Reduced timeout for faster failure
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
        
        # Format matches with deduplication - improved error handling
        matches = []
        sources = rag_response.sources
        seen_papers = set()  # Track unique papers
        
        for i, source in enumerate(sources):
            try:
                # Create a unique paper identifier
                paper_id = source.get('source_file', '') or source.get('title', 'Unknown')
                
                # Skip if we've already seen this paper (unless it's the first occurrence)
                if paper_id in seen_papers:
                    continue
                
                seen_papers.add(paper_id)
                
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
                        'chunk_index': source.get('chunk_index'),
                        'paper_id': paper_id  # Add paper ID for frontend reference
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
            llm_response=rag_response.answer,  # Access the answer field from RAG service
            model_used=request.model,
            num_sources=len(matches),
            response_time_ms=response_time,
            filters_applied=filters
        )
        
        logger.info(f"✅ Query completed in {response_time}ms with {len(matches)} sources using {request.model}")
        return response
        
    except Exception as e:
        logger.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==============================================================================
# USER MANAGEMENT ENDPOINTS
# ==============================================================================

@app.get("/user/profile", response_model=UserInfo)
async def get_user_profile(# current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Get current user profile"""
    return UserInfo(
        sub=current_user.get('sub'),
        email=current_user.get('email'),
        name=current_user.get('name'),
        picture=current_user.get('picture'),
        permissions=current_user.get('permissions', [])
    )

@app.get("/user/permissions")
async def get_user_permissions_endpoint(# current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Get current user permissions"""
    return {
        "permissions": current_user.get('permissions', []),
        "email": current_user.get('email')
    }

# ==============================================================================
# SPECIALIZED ENDPOINTS (WITH PERMISSION CHECKS)
# ==============================================================================

@app.post("/query/methods")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_methods(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions focused on methodology sections (requires research permission)"""
    request.chunk_type = "methods"
    return await query_with_llm(request, current_user)

@app.post("/query/results")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_results(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions focused on results sections (requires research permission)"""
    request.chunk_type = "results"
    return await query_with_llm(request, current_user)

@app.post("/query/reasoning")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_with_reasoning(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions with detailed reasoning (requires research permission)"""
    # This could use a different prompt or model configuration
    return await query_with_llm(request, current_user)

@app.post("/query/abstracts")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_abstracts(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions focused on abstracts (requires research permission)"""
    request.chunk_type = "abstract"
    return await query_with_llm(request, current_user)

@app.post("/query/high-impact")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_high_impact(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions focused on high-impact papers (requires research permission)"""
    request.min_citations = 50
    return await query_with_llm(request, current_user)

@app.post("/query/recent")
# @require_permission("read:research")  # TEMPORARILY DISABLED
async def query_recent(
    request: QueryRequest,
    # current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Ask questions focused on recent papers (requires research permission)"""
    from datetime import datetime
    current_year = datetime.now().year
    request.year_start = current_year - 2
    request.year_end = current_year
    return await query_with_llm(request, current_user)

# ==============================================================================
# ADMIN ENDPOINTS (WITH ADMIN PERMISSION CHECKS)
# ==============================================================================

@app.get("/admin/stats")
# @require_permission("admin:access")  # TEMPORARILY DISABLED
async def get_admin_stats(# current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Get admin statistics (requires admin permission)"""
    return {
        "total_users": "N/A",  # Would need user database
        "total_queries": "N/A",  # Would need query logging
        "system_status": "operational",
        "admin_user": current_user.get('email')
    }

@app.get("/admin/users")
# @require_permission("admin:access")  # TEMPORARILY DISABLED
async def get_admin_users(# current_user: Dict[str, Any] = Depends(get_current_user)  # TEMPORARILY DISABLED
):
    """Get user list (requires admin permission)"""
    return {
        "message": "User management not implemented",
        "admin_user": current_user.get('email')
    }

# ==============================================================================
# PUBLIC ENDPOINTS (NO AUTHENTICATION REQUIRED)
# ==============================================================================

@app.get("/models")
async def get_available_models():
    """Get available AI models (public endpoint)"""
    return {
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o", "description": "Latest and most capable model (Recommended)"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Fastest and most cost-effective"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Previous generation"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Budget option"}
        ]
    }

@app.get("/filters/options")
async def get_filter_options():
    """Get available filter options (public endpoint)"""
    return {
        "content_types": [
            {"value": "abstract", "label": "Abstracts"},
            {"value": "methods", "label": "Methods"},
            {"value": "results", "label": "Results"},
            {"value": "discussion", "label": "Discussion"},
            {"value": "content", "label": "All Content"}
        ],
        "time_periods": [
            {"value": "recent", "label": "Last 2 Years"},
            {"value": "5year", "label": "Last 5 Years"},
            {"value": "decade", "label": "Last 10 Years"},
            {"value": "all", "label": "All Years"}
        ],
        "citation_levels": [
            {"value": "high", "label": "High Impact (50+ citations)"},
            {"value": "medium", "label": "Medium Impact (10-49)"},
            {"value": "emerging", "label": "Emerging (1-9)"},
            {"value": "all", "label": "All Papers"}
        ]
    }

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
