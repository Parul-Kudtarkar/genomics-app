# Production FastAPI Deployment Guide

## 📋 Overview

This guide documents the complete production deployment of an enhanced FastAPI-based Genomics RAG system on AWS EC2 with Ubuntu. The setup includes Gunicorn as the WSGI server, Nginx as a reverse proxy, Redis caching, rate limiting, API key authentication, and a React frontend.

## 🚀 Why FastAPI for Genomics RAG?

### Performance & Speed Requirements
FastAPI is the ideal choice for genomics RAG systems due to its exceptional performance characteristics:

```python
# FastAPI handles high-performance requirements:
- Vector search queries (milliseconds response time)
- LLM API calls (async processing) 
- Multiple concurrent researchers
- Real-time RAG responses
- High-throughput data processing
```

**Key Performance Benefits:**
- **Async/Await Support**: Perfect for I/O-bound operations like OpenAI and Pinecone API calls
- **High Throughput**: Can handle thousands of requests per second
- **Low Latency**: Critical for real-time search and query responses
- **Built on Starlette**: High-performance ASGI framework foundation

### Automatic API Documentation
```python
# FastAPI automatically generates interactive API docs
@app.post("/query", response_model=RAGResponse)
async def query_with_llm(request: QueryRequest):
    """Main endpoint: Vector search + LLM response"""
    # Researchers can test APIs directly at /docs or /redoc
    # No need for external tools like Postman
```

**Benefits for Genomics Research:**
- **Self-Documenting**: Researchers understand endpoints immediately
- **OpenAPI/Swagger**: Industry standard documentation
- **Interactive Testing**: Built-in API testing interface
- **Type Safety**: Automatic validation and error messages

### Type Safety & Validation
```python
# Pydantic models ensure robust data validation
class QueryRequest(BaseModel):
    query: constr(strip_whitespace=True, min_length=1, max_length=500) = Field(...)
    model: str = Field(default="gpt-4", description="LLM model to use")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Genomics-specific filters
    journal: Optional[str] = Field(None, description="Filter by journal name")
    author: Optional[str] = Field(None, description="Filter by author name")
    year_start: Optional[int] = Field(None, description="Start year for filtering")
    min_citations: Optional[int] = Field(None, description="Minimum citation count")
```

**Why This Matters for Genomics:**
- **Data Validation**: Prevents invalid queries that could break RAG pipelines
- **Type Hints**: Better IDE support and code maintainability
- **Automatic Error Messages**: Clear feedback for researchers
- **Research Data Integrity**: Ensures consistent query formats

### Modern Python Features
```python
# Leverages modern Python for genomics research
from typing import List, Dict, Any, Optional
from datetime import datetime

# Async support for concurrent research operations
@app.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")  # Rate limiting for research usage
@cache_decorator(expire=60, namespace="search")  # Redis caching
async def search_only(request: SearchOnlyRequest, api_key: str = Depends(get_api_key)):
    """Vector search only (no LLM)"""
    # Handles multiple concurrent research queries efficiently
```

### Built-in Security Features
```python
# FastAPI provides security out of the box
from fastapi import Depends, HTTPException, status

def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

# Automatic authentication on every research request
@app.post("/query", response_model=RAGResponse)
async def query_with_llm(request: QueryRequest, api_key: str = Depends(get_api_key)):
    # Secure access to genomics research data
```

### Perfect AI/ML Integration
```python
# Seamless integration with genomics research stack
from services.search_service import GenomicsSearchService
from services.rag_service import GenomicsRAGService
from langchain_openai import ChatOpenAI

# FastAPI works perfectly with:
# - LangChain for RAG pipelines
# - OpenAI API for LLM calls
# - Pinecone for vector search
# - Redis for research result caching
# - Scientific computing libraries
```

## 🔬 Genomics Research-Specific Benefits

### Complex Query Processing
```python
# Handles sophisticated genomics research queries
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

@app.post("/query/reasoning")
async def query_with_reasoning(request: QueryRequest):
    """Enhanced reasoning with step-by-step analysis"""
    # Complex reasoning chains for genomics research validation
```

### Real-time Research Collaboration
```python
# Multiple researchers can use the system simultaneously
# FastAPI's async nature handles concurrent research requests
# No blocking when one researcher is doing complex analysis
# Perfect for collaborative genomics research teams
```

### Scalable Research Architecture
```python
# System scales with growing research demands
# FastAPI + Gunicorn + Nginx + Redis
# Can handle increasing numbers of researchers
# Supports growing genomics datasets
```

## 🆚 Framework Comparison for Genomics RAG

| Feature | FastAPI | Flask | Django | Express.js |
|---------|---------|-------|--------|------------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Async Support** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Auto Documentation** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| **Type Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **AI/ML Integration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Learning Curve** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Research Friendly** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 🎯 Perfect Fit for Genomics Research Workflow

### Research Process Integration
```python
# Complete genomics research workflow:
# 1. Literature Search: POST /search
# 2. Research Questions: POST /query  
# 3. Method Analysis: POST /query/methods
# 4. Results Analysis: POST /query/results
# 5. Reasoning Validation: POST /query/reasoning
# 6. Filter by criteria: Use query parameters
# 7. Real-time responses: Async processing
```

### Production-Ready for Research
```python
# Complete production deployment includes:
# - Gunicorn for production serving
# - Nginx for reverse proxy and load balancing
# - Redis for research result caching
# - Rate limiting for fair research usage
# - Comprehensive logging for research audit trails
# - Security for sensitive research data
```

### Future-Proof Research Platform
```python
# Easy to extend for evolving research needs:
# - New LLM models for genomics
# - Additional vector databases
# - Enhanced filtering for research criteria
# - Real-time collaboration features
# - Integration with research databases
# - Advanced analytics capabilities
```

## 🚀 Key Benefits for Genomics RAG System

1. **⚡ Speed**: Handles vector search and LLM queries with millisecond response times
2. **📈 Scalability**: Grows with research team and dataset size
3. **🔧 Developer Experience**: Easy to maintain and extend for research needs
4. **👥 Research Friendly**: Self-documenting APIs for researchers of all technical levels
5. **🛡️ Production Ready**: Built-in security and monitoring for research environments
6. **🤖 AI/ML Native**: Works seamlessly with modern AI/ML libraries
7. **📊 Real-time**: Supports concurrent research queries without blocking
8. **🔍 Search Optimized**: Perfect for complex genomics literature search
9. **📝 Documentation**: Automatic API docs reduce researcher onboarding time
10. **🔬 Scientific**: Type safety ensures research data integrity

## 🏗️ Architecture

```
Internet → AWS Security Group → Nginx (Port 80) → Gunicorn (Port 8000) → FastAPI App → Redis Cache
                                                                    ↓
                                                              React Frontend
```

## 📁 Project Structure

```
/home/ubuntu/genomics-app/
.
├── api/
│   ├── __init__.py
│   └── endpoints/
│       └── __init__.py
├── config/
│   ├── __init__.py
│   └── vector_db.py
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
├── services/
│   ├── __init__.py
│   ├── rag_service.py
│   ├── search_service.py
│   └── vector_store.py
├── scripts/
│   ├── __init__.py
│   ├── analytics.py
│   ├── credential_checker.py
│   ├── migrate_existing_data.py
│   ├── setup_environment.py
│   └── setup_vector_db.py
├── tests/
│   ├── __init__.py
│   └── test_vector_store.py
├── logs/
│   ├── access.log
│   └── error.log
├── main.py
├── requirements.txt
├── gunicorn.conf.py
├── genomics-api.service
├── start_api.sh
├── stop_api.sh
├── restart_api.sh
├── status.sh
└── maintenance.sh
```

## 🚀 Step-by-Step Deployment

### Step 1: Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y nginx python3-pip python3-venv redis-server

# Install Python dependencies in virtual environment
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
pip install -r requirements.txt

# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
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
import os

# Server socket
bind = "127.0.0.1:8000"  # Internal only
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 10

# Restart workers after requests (prevents memory leaks)
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = 50

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info')
accesslog = "/home/ubuntu/genomics-app/logs/access.log"
errorlog = "/home/ubuntu/genomics-app/logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'genomics_api'
daemon = False
pidfile = "/home/ubuntu/genomics-app/genomics-api.pid"
user = "ubuntu"
group = "ubuntu"

# Preload app for better performance
preload_app = True
```

#### 3.2 API Management Scripts

**File: `/home/ubuntu/genomics-app/start_api.sh`**

```bash
#!/bin/bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if Redis is running
if ! systemctl is-active --quiet redis-server; then
    echo "⚠️  Starting Redis..."
    sudo systemctl start redis-server
fi

# Start API with enhanced configuration
gunicorn main:app \
  --config gunicorn.conf.py \
  --daemon \
  --pid genomics-api.pid

echo "✅ Genomics API started on localhost:8000"
echo "PID: $(cat genomics-api.pid)"
echo "Logs: /home/ubuntu/genomics-app/logs/"
echo "Redis: $(systemctl is-active redis-server)"
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

# Check API status
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

# Check Redis status
echo ""
echo "=== Redis Status ==="
if systemctl is-active --quiet redis-server; then
    echo "✅ Redis is running"
    echo "Redis memory: $(redis-cli info memory | grep used_memory_human | cut -d: -f2)"
else
    echo "❌ Redis is not running"
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

**Step 1: Add Rate Limiting to Main Nginx Config**

First, add the rate limiting zones to the main Nginx configuration:

```bash
# Edit the main nginx.conf file
sudo nano /etc/nginx/nginx.conf
```

Add these lines inside the `http` block (before the `include` directives):

```nginx
# Rate limiting zones (add this inside the http block)
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=search:10m rate=30r/m;
```

**Step 2: Create Site Configuration**

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
        limit_req zone=api burst=20 nodelay;
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
    
    # Search endpoints with higher rate limits
    location ~ ^/api/(search|query) {
        limit_req zone=search burst=5 nodelay;
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Direct API access (without /api prefix)
    location ~ ^/(health|status|docs|redoc|openapi.json|search|query)$ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Specialized endpoints
    location ~ ^/query/(methods|results|abstracts|high-impact|recent|reasoning)$ {
        limit_req zone=search burst=5 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # React frontend (served from build directory)
    location / {
        root /home/ubuntu/genomics-app/frontend/build;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
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

### Step 5: Environment Configuration

**File: `/home/ubuntu/genomics-app/.env`**

```env
# API Keys (UPDATE WITH YOUR ACTUAL KEYS)
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications
API_KEY=your_secure_api_key_here

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_DIMENSION=1536

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# Performance
WORKERS=4
MAX_REQUESTS=1000

# LLM Configuration
DEFAULT_LLM_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.1
DEFAULT_TOP_K=8

# Caching Configuration
ENABLE_CACHING=true
CACHE_SIZE=1000
REDIS_URL=redis://localhost:6379

# RAG Configuration
RAG_TIMEOUT=30
```

### Step 6: React Frontend Deployment

```bash
# Install Node.js dependencies
cd /home/ubuntu/genomics-app/frontend
npm install

# Build for production
npm run build

# Set proper permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/genomics-app/frontend/build
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
# Vector search test (with API key)
curl -X POST "http://localhost/search" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -d '{"query": "CRISPR gene editing", "top_k": 3}'

# LLM query test
curl -X POST "http://localhost/query" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -d '{"query": "What is CRISPR?", "model": "gpt-3.5-turbo", "top_k": 2}'

# Test specialized endpoints
curl -X POST "http://localhost/query/methods" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -d '{"query": "What methods are used for gene editing?"}'
```

### Service Status Checks

```bash
# Check API status
cd /home/ubuntu/genomics-app && ./status.sh

# Check nginx status
sudo systemctl status nginx

# Check Redis status
sudo systemctl status redis-server

# Check process list
ps aux | grep gunicorn
ps aux | grep nginx
ps aux | grep redis
```

## 📊 Monitoring & Maintenance

### Log Locations

- **API Access Logs**: `/home/ubuntu/genomics-app/logs/access.log`
- **API Error Logs**: `/home/ubuntu/genomics-app/logs/error.log`
- **Nginx Access Logs**: `/var/log/nginx/access.log`
- **Nginx Error Logs**: `/var/log/nginx/error.log`
- **Redis Logs**: `/var/log/redis/redis-server.log`

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

# Check Redis memory usage
redis-cli info memory
```

### Maintenance Tasks

```bash
# Restart API
cd /home/ubuntu/genomics-app && ./restart_api.sh

# Restart nginx
sudo systemctl restart nginx

# Restart Redis
sudo systemctl restart redis-server

# Clear Redis cache
redis-cli flushall

# View recent logs
tail -50 /home/ubuntu/genomics-app/logs/error.log
```

## 🔒 Security Considerations

### Current Security Measures

1. **API Key Authentication**: All endpoints require valid API key
2. **Rate Limiting**: Nginx and FastAPI rate limiting
3. **API Internal Binding**: FastAPI only accessible via nginx proxy
4. **Security Headers**: XSS protection, content type validation
5. **Process Isolation**: API runs as ubuntu user, not root
6. **Input Validation**: Pydantic models with constraints
7. **Caching**: Redis-based caching with TTL
8. **Request Logging**: Structured JSON logging with request IDs

### Recommended Enhancements

1. **SSL/TLS**: Add Let's Encrypt certificate
2. **API Authentication**: Implement JWT tokens
3. **Firewall**: Configure UFW for additional protection
4. **Monitoring**: Add Prometheus metrics
5. **Backup**: Implement automated backups

## 🚀 Production Deployment Checklist

- [ ] AWS Security Group configured (ports 22, 80, 443)
- [ ] Virtual environment activated with dependencies
- [ ] Environment variables configured in `.env`
- [ ] Redis server installed and running
- [ ] Gunicorn configuration created
- [ ] Production scripts created and executable
- [ ] Nginx configuration created and enabled
- [ ] React frontend built and deployed
- [ ] Port conflicts resolved
- [ ] API started successfully
- [ ] Nginx started successfully
- [ ] Redis started successfully
- [ ] Health checks passing
- [ ] API functionality tests passing
- [ ] Frontend accessible
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

# Common Nginx configuration errors:
# 1. "limit_req_zone directive is not allowed here"
#    Solution: Move limit_req_zone to /etc/nginx/nginx.conf http block
# 2. "upstream directive is not allowed here"
#    Solution: Move upstream definitions to /etc/nginx/nginx.conf http block
# 3. "server_name directive is not allowed here"
#    Solution: Ensure server_name is inside server block
```

**Redis issues:**
```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping

# Check Redis logs
sudo tail /var/log/redis/redis-server.log
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

## 📱 React Frontend Integration

The React frontend is now fully integrated:

1. **Build Location**: `/home/ubuntu/genomics-app/frontend/build/`
2. **Nginx Serves**: React app from root `/` 
3. **API Calls**: Use `/api/*` endpoints
4. **Proxy Configuration**: Development proxy to `http://localhost:8000`

**Example React API integration:**
```javascript
// React will call these endpoints with API key
const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 
        'Content-Type': 'application/json',
        'x-api-key': process.env.REACT_APP_API_KEY
    },
    body: JSON.stringify({ query: "CRISPR", top_k: 5 })
});
```

## 📋 Enhanced Features

### New API Endpoints

- **`/query/reasoning`**: Enhanced reasoning with step-by-step analysis
- **`/models`**: Get available LLM models
- **`/filters/options`**: Get available filter options
- **Enhanced `/status`**: Detailed system status with metrics

### Performance Features

- **Redis Caching**: 60-second cache for search and query results
- **Rate Limiting**: 30/min for search, 20/min for queries
- **Request Logging**: Structured JSON logs with request IDs
- **Gzip Compression**: Automatic compression for all responses

### Security Features

- **API Key Authentication**: Required for all endpoints
- **Input Validation**: Pydantic models with constraints
- **Rate Limiting**: Nginx and FastAPI rate limiting
- **Security Headers**: Comprehensive security headers

---

**Deployment Date**: January 2025
**Server**: AWS EC2 Ubuntu  
**Status**: ✅ Production Ready with Enhanced Features
