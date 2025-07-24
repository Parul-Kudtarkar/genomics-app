# Production FastAPI Deployment Guide

## 📋 Overview

This guide documents the complete production deployment of a FastAPI-based Genomics RAG system on AWS EC2 with Ubuntu. The setup includes Gunicorn as the WSGI server, Nginx as a reverse proxy, and proper security configurations.

## 🏗️ Architecture

```
Internet → AWS Security Group → Nginx (Port 80) → Gunicorn (Port 8000) → FastAPI App
```

## 📁 Project Structure

```
/home/ubuntu/genomics-app/
.
├── api
│   ├── __init__.py
│   └── endpoints
│       └── __init__.py
├── api.log
├── config
│   ├── __init__.py
│   └── vector_db.py
├── enhanced_pdf_processing.log
├── genomics-api.pid
├── genomics-api.service
├── genomics_api.pid
├── gunicorn.conf.py
├── logs
│   ├── access.log
│   └── error.log
├── main.py
├── maintenance.sh
├── pdf_ingestion_pipeline.py
├── processed_files.json
├── requirements.txt
├── restart_api.sh
├── scripts
│   ├── __init__.py
│   ├── analytics.py
│   ├── credential_checker.py
│   ├── migrate_existing_data.py
│   ├── setup_environment.py
│   └── setup_vector_db.py
├── services
│   ├── __init__.py
│   ├── rag_service.py
│   ├── search_service.py
│   └── vector_store.py
├── start_api.sh
├── stop_api.sh
├── test_api.py
├── test_langchain_rag.py
└── tests
    ├── __init__.py
    └── test_vector_store.py

7 directories, 34 files
```

## 🚀 Step-by-Step Deployment

### Step 1: Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y nginx python3-pip python3-venv

# Install Python dependencies in virtual environment
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
pip install gunicorn uvicorn
```

### Step 2: AWS Security Group Configuration

**Required Inbound Rules:**
- **SSH (22)**: Your IP / 0.0.0.0/0
- **HTTP (80)**: 0.0.0.0/0
- **HTTPS (443)**: 0.0.0.0/0 (for SSL later)

**Remove:**
- Any rules for port 8000 (API is now internal only)

### Step 3: Production Scripts Creation

#### 3.1 Gunicorn Configuration

**File: `/home/ubuntu/genomics-app/gunicorn.conf.py`**

```python
import multiprocessing

# Server socket
bind = "127.0.0.1:8000"  # Internal only
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 10

# Restart workers after requests (prevents memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "/home/ubuntu/genomics-app/logs/access.log"
errorlog = "/home/ubuntu/genomics-app/logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'genomics_api'
daemon = False
pidfile = "/home/ubuntu/genomics-app/genomics-api.pid"
user = "ubuntu"
group = "ubuntu"
```

#### 3.2 API Management Scripts

**File: `/home/ubuntu/genomics-app/start_api.sh`**

```bash
#!/bin/bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

gunicorn main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --daemon \
  --pid genomics-api.pid \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info

echo "✅ Genomics API started on localhost:8000"
echo "PID: $(cat genomics-api.pid)"
echo "Logs: /home/ubuntu/genomics-app/logs/"
echo "Test: curl http://localhost:8000/health"
```

**File: `/home/ubuntu/genomics-app/stop_api.sh`**

```bash
#!/bin/bash
cd /home/ubuntu/genomics-app

if [ -f genomics-api.pid ]; then
    PID=$(cat genomics-api.pid)
    kill $PID
    rm genomics-api.pid
    echo "✅ API stopped (PID: $PID)"
else
    echo "❌ No PID file found. Checking for running processes..."
    pkill -f "gunicorn.*main:app"
    echo "✅ Killed any remaining gunicorn processes"
fi
```

**File: `/home/ubuntu/genomics-app/restart_api.sh`**

```bash
#!/bin/bash
cd /home/ubuntu/genomics-app
echo "🔄 Restarting API..."
./stop_api.sh
sleep 2
./start_api.sh
```

**File: `/home/ubuntu/genomics-app/status.sh`**

```bash
#!/bin/bash
echo "=== Genomics API Status ==="
echo "Time: $(date)"
echo ""

if [ -f genomics-api.pid ]; then
    PID=$(cat genomics-api.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ API is running (PID: $PID)"
        echo "Memory usage: $(ps -p $PID -o %mem --no-headers)%"
        echo "CPU usage: $(ps -p $PID -o %cpu --no-headers)%"
    else
        echo "❌ API PID file exists but process is dead"
    fi
else
    echo "❌ API is not running (no PID file)"
fi

echo ""
echo "=== Recent logs ==="
tail -5 /home/ubuntu/genomics-app/logs/error.log

echo ""
echo "=== Quick health check ==="
curl -s http://localhost:8000/health || echo "❌ Health check failed"
```

#### 3.3 Make Scripts Executable

```bash
chmod +x /home/ubuntu/genomics-app/start_api.sh
chmod +x /home/ubuntu/genomics-app/stop_api.sh
chmod +x /home/ubuntu/genomics-app/restart_api.sh
chmod +x /home/ubuntu/genomics-app/status.sh
```

### Step 4: Nginx Configuration

#### 4.1 Remove Default Configuration

```bash
sudo rm -f /etc/nginx/sites-enabled/default
```

#### 4.2 Create Production Configuration

**File: `/etc/nginx/sites-available/genomics-app`**

```nginx
server {
    listen 80;
    server_name _;  # Accept any domain name
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # API endpoints with /api prefix (for React frontend)
    location /api/ {
        rewrite ^/api/(.*) /$1 break;  # Remove /api prefix
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_http_version 1.1;
        proxy_intercept_errors on;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Direct API access (without /api prefix)
    location ~ ^/(health|status|docs|redoc|openapi.json|search|query)$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Specialized endpoints
    location ~ ^/query/(methods|results|abstracts|high-impact|recent|compare)$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Root location (placeholder for React frontend)
    location / {
        return 200 'Genomics API is running! Try /health, /status, /docs, or /api/health';
        add_header Content-Type text/plain;
    }
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
}
```

#### 4.3 Enable Configuration

```bash
sudo ln -s /etc/nginx/sites-available/genomics-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl start nginx
```

### Step 5: Resolve Port Conflicts

```bash
# Kill any process using port 80
sudo fuser -k 80/tcp 2>/dev/null || true

# Stop conflicting services
sudo systemctl stop apache2 2>/dev/null || true
```

### Step 6: Environment Configuration

**File: `/home/ubuntu/genomics-app/.env`**

```env
# API Keys (UPDATE WITH YOUR ACTUAL KEYS)
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false
```

### Step 7: Start Services

```bash
# Start API
cd /home/ubuntu/genomics-app
./start_api.sh

# Restart nginx to ensure clean start
sudo systemctl restart nginx
```

## 🧪 Testing & Validation

### Basic Health Checks

```bash
# Test API directly
curl http://localhost/health
curl http://localhost/status
curl http://localhost/docs

# Test with /api prefix (for React)
curl http://localhost/api/health
curl http://localhost/api/status

# Test externally
curl http://YOUR_PUBLIC_IP/health
curl http://YOUR_PUBLIC_IP/docs
```

### API Functionality Tests

```bash
# Vector search test
curl -X POST "http://localhost/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene editing", "top_k": 3}'

curl -X POST "http://localhost/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene editing", "top_k": 3}'

# LLM query test
curl -X POST "http://localhost/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is CRISPR?", "model": "gpt-3.5-turbo", "top_k": 2}'
```

### Service Status Checks

```bash
# Check API status
cd /home/ubuntu/genomics-app && ./status.sh

# Check nginx status
sudo systemctl status nginx

# Check process list
ps aux | grep gunicorn
ps aux | grep nginx
```

## 📊 Monitoring & Maintenance

### Log Locations

- **API Access Logs**: `/home/ubuntu/genomics-app/logs/access.log`
- **API Error Logs**: `/home/ubuntu/genomics-app/logs/error.log`
- **Nginx Access Logs**: `/var/log/nginx/access.log`
- **Nginx Error Logs**: `/var/log/nginx/error.log`

### Monitoring Commands

```bash
# View API logs in real-time
tail -f /home/ubuntu/genomics-app/logs/error.log

# View nginx logs
sudo tail -f /var/log/nginx/error.log

# Check API status
cd /home/ubuntu/genomics-app && ./status.sh

# Check system resources
htop
df -h
free -h
```

### Maintenance Tasks

```bash
# Restart API
cd /home/ubuntu/genomics-app && ./restart_api.sh

# Restart nginx
sudo systemctl restart nginx

# View recent logs
tail -50 /home/ubuntu/genomics-app/logs/error.log
```

## 🔒 Security Considerations

### Current Security Measures

1. **API Internal Binding**: FastAPI only accessible via nginx proxy
2. **Security Headers**: XSS protection, content type validation
3. **Process Isolation**: API runs as ubuntu user, not root
4. **Log Rotation**: Prevents disk space issues

### Recommended Enhancements

1. **SSL/TLS**: Add Let's Encrypt certificate
2. **Rate Limiting**: Implement nginx rate limiting
3. **API Authentication**: Add API key validation
4. **Firewall**: Configure UFW for additional protection

## 🚀 Production Deployment Checklist

- [ ] AWS Security Group configured (ports 22, 80, 443)
- [ ] Virtual environment activated with dependencies
- [ ] Environment variables configured in `.env`
- [ ] Gunicorn configuration created
- [ ] Production scripts created and executable
- [ ] Nginx configuration created and enabled
- [ ] Port conflicts resolved
- [ ] API started successfully
- [ ] Nginx started successfully
- [ ] Health checks passing
- [ ] API functionality tests passing
- [ ] Monitoring scripts working
- [ ] Log rotation configured (optional)
- [ ] SSL certificate installed (optional)

## 🔧 Troubleshooting

### Common Issues

**API won't start:**
```bash
# Check logs
cat /home/ubuntu/genomics-app/logs/error.log

# Check if port is in use
sudo netstat -tulpn | grep :8000

# Test API manually
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
python main.py
```

**Nginx won't start:**
```bash
# Check nginx config
sudo nginx -t

# Check what's using port 80
sudo netstat -tulpn | grep :80

# Check nginx logs
sudo tail /var/log/nginx/error.log
```

**Can't access externally:**
```bash
# Check AWS Security Group
# Verify port 80 is open to 0.0.0.0/0

# Test locally first
curl http://localhost/health

# Check nginx is running
sudo systemctl status nginx
```

## 📱 React Frontend Integration (Future)

When ready to add React frontend:

1. **Build React app**: `npm run build`
2. **Copy build files** to: `/home/ubuntu/genomics-app/frontend/build/`
3. **Update nginx config** to serve React from `/` 
4. **React API calls** will use `/api/*` endpoints

**Example React API integration:**
```javascript
// React will call these endpoints
const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: "CRISPR", top_k: 5 })
});
```

## 📋 Required Files 


1. FastAPI application
```
#main.py 
# ==============================================================================
# main.py (main API file)
# ==============================================================================

# main.py
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Your existing services
from services.search_service import GenomicsSearchService
from services.rag_service import GenomicsRAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ==============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query or question")
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
    query: str = Field(..., description="Search query")
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

# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================

app = FastAPI(
    title="Genomics RAG API",
    description="Vector search and LLM-powered Q&A for genomics research",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    global search_service, rag_service
    
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
            search_service=search_service,
            openai_api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.1
        )
        logger.info("✅ RAG service initialized")
        
        # Test connection
        stats = search_service.get_search_statistics()
        logger.info(f"📊 Vector store stats: {stats.get('total_vectors', 0)} vectors")
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
@app.get("/status")
async def get_status():
    """Working status endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API is running successfully"
    }
@app.post("/search", response_model=SearchResponse)
async def search_only(request: SearchOnlyRequest):
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
async def query_with_llm(request: QueryRequest):
    """Main endpoint: Vector search + LLM response"""
    start_time = datetime.now()
    
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")
        
        # Build filters
        filters = build_filters(request)
        
        logger.info(f"🔍 Processing query: '{request.query[:50]}...' with model: {request.model}")
        
        # Update RAG service model if different
        if request.model != rag_service.llm.model_name:
            from langchain_openai import ChatOpenAI
            rag_service.llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=request.model,
                temperature=request.temperature
            )
        
        # Get RAG response
        rag_response = rag_service.ask_question(
            question=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )
        
        # Format matches
        matches = []
        for source in rag_response.get('sources', []):
            match = VectorMatch(
                id=f"{source.get('source_file', '')}_chunk",
                score=source.get('relevance_score', 0.0),
                content=source.get('content_preview', ''),
                title=source.get('title', 'Unknown'),
                source=source.get('source_file', 'Unknown'),
                metadata={
                    'journal': source.get('journal'),
                    'year': source.get('year'),
                    'authors': source.get('authors', []),
                    'doi': source.get('doi'),
                    'citation_count': source.get('citation_count', 0)
                }
            )
            matches.append(match)
        
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response = RAGResponse(
            query=request.query,
            matches=matches,
            llm_response=rag_response.get('answer', 'No response generated'),
            model_used=request.model,
            num_sources=len(matches),
            response_time_ms=response_time,
            filters_applied=filters
        )
        
        logger.info(f"✅ Query completed in {response_time}ms with {len(matches)} sources")
        return response
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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
        port=int(os.getenv('PORT',8000)),
        reload=not is_production,
        log_level="info",
        access_log=True
    )
import logging
import traceback

# Enhanced error handling for status endpoint
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
        
        available_models = [
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo"
        ]
        
        return StatusResponse(
            status="healthy" if search_service and rag_service else "degraded",
            index_stats=stats,
            available_models=available_models,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a degraded status instead of crashing
        return StatusResponse(
            status="error",
            index_stats={"error": str(e)},
            available_models=[],
            timestamp=datetime.now().isoformat()
        )
```
2 Python dependencies
```
cat requirements.txt 
# requirements.txt

# FastAPI and server
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Your existing dependencies (keep these exact versions)
langchain==0.2.16
langchain-openai==0.1.23
langchain-core==0.2.38
openai==1.51.2
pinecone-client==3.2.2
httpx>=0.25.0,<0.28.0
python-dotenv==1.0.1
numpy==1.26.4

# PDF processing (your existing)
PyPDF2==3.0.1
pdfplumber==0.10.3

# Optional: Production server
gunicorn==21.2.0
```
3. Environment variables
```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=genomics-publications
EMBEDDING_DIMENSION=1536
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
OPENAI_API_KEY=
# Production Settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# Performance
WORKERS=4
MAX_REQUESTS=1000
```
5. **`/etc/nginx/sites-available/genomics-app`** - Final nginx configuration
```
server {
    listen 80;
    server_name koi.pankbase.org;  # Accept any domain name
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # API endpoints (backend) - proxy /api/* to your FastAPI
    location /api/ {
        rewrite ^/api/(.*) /$1 break;  # Remove /api prefix
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_http_version 1.1;
        proxy_intercept_errors on;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Direct API access (no /api prefix) - for now until React is ready
    location ~ ^/(health|docs|redoc|openapi.json|search|query)$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # React frontend (will be served here later)
    location / {
        # For now, return a simple message until React is ready
        return 200 'Genomics API is running! Try /health,  /docs, or /api/health';
        add_header Content-Type text/plain;
    }
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
```
```
#/etc/systemd/system/genomics-api.service
[Unit]
Description=Genomics RAG API
After=network.target

[Service]
Type=exec
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/genomics-app
Environment=PATH=/home/ubuntu/venv/bin
ExecStart=/home/ubuntu/venv/bin/gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8\
000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

```
---

**Deployment Date**: July 16th 2025
**Server**: AWS EC2 Ubuntu  
**Status**: ✅ Production Ready
