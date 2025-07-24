# 📋 Production-Ready Diabetes Research Assistant Implementation Guide

This guide describes the complete, production-ready deployment for your diabetes research assistant, now featuring an advanced Apple Intelligence React frontend with rich search, filtering, and results UI.

## 🎯 **System Overview**

**What We're Building:**
- React frontend with Apple Intelligence design
- Advanced search card with model selection
- Animated filter panel and filter pills for metadata-rich filtering
- FastAPI backend with enhanced metadata
- Pinecone vector database with rich paper data
- Nginx production deployment
- Enhanced search with deduplication and text cleaning

## 🗂️ **Final Project Structure**
```
/home/ubuntu/genomics-app/
├── main.py                          # Enhanced FastAPI backend
├── services/
│   ├── search_service.py           # Vector search with metadata
│   ├── rag_service.py              # LangChain RAG integration
│   └── vector_store.py             # Pinecone client
├── config/
│   └── vector_db.py                # Configuration management
├── frontend/
│   ├── package.json                # React dependencies
│   ├── README.md                   # Frontend usage
│   ├── public/
│   │   └── index.html              # React root
│   └── src/
│       ├── App.js                  # Main app shell
│       ├── index.js                # React entry point
│       ├── utils/
│       │   └── metadataHelpers.js  # Metadata helpers
│       └── components/
│           ├── Search/
│           │   ├── AdvancedSearchCard.js
│           │   ├── FilterPanel.js
│           │   └── FilterPills.js
│           └── Results/
│               └── EnhancedResultCard.js
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
├── start_api.sh                    # API management scripts
├── stop_api.sh
├── restart_api.sh
└── logs/                           # Application logs
```

## 🚀 **Frontend Features**

### 1. **Advanced Search Card**
- Apple-style card for research question input
- Model selector (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)
- Animated Filters button to open filter panel

### 2. **Filter Panel**
- Animated, Apple-style panel (framer-motion)
- Filter by content type, time period, citation level, data quality
- All filter state is managed and passed to backend

### 3. **Filter Pills**
- Active filters are shown as pill badges above the search card
- Each pill can be removed to reset that filter
- Only non-default filters are shown

### 4. **Results**
- LLM response is shown in a styled card
- Source publications are shown as EnhancedResultCard components
- Each card displays title, journal, year, chunk type, citations, authors, and a content preview
- All metadata is handled with null safety and Crossref-preferred logic

## 🧑‍💻 **How to Run the Frontend**

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```
2. **Start the development server:**
   ```bash
   npm start
   ```
   The app will run at http://localhost:3000 and proxy API requests to http://localhost:8000.

## 🧩 **Key Files and Components**
- `src/App.js` — Main app shell, orchestrates search, filters, and results
- `src/components/Search/AdvancedSearchCard.js` — Main search UI
- `src/components/Search/FilterPanel.js` — Animated filter panel
- `src/components/Search/FilterPills.js` — Active filter pills
- `src/components/Results/EnhancedResultCard.js` — Rich result card for each paper
- `src/utils/metadataHelpers.js` — All null-safe, Crossref-preferred metadata helpers

## 🔧 **Step 1: Backend Dependencies**

### **1.1 Python Requirements**
```bash
# File: requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# LangChain RAG Dependencies
langchain==0.2.16
langchain-openai==0.1.23
langchain-core==0.2.38
openai==1.51.2
pinecone-client==3.2.2
httpx>=0.25.0,<0.28.0

# Enhanced PDF Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
python-dotenv==1.0.1
numpy==1.26.4
requests==2.31.0
```

### **1.2 Install Dependencies**
```bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
pip install -r requirements.txt
```

## 🔧 **Step 2: Environment Configuration**

### **2.1 Production Environment Variables**
```bash
# File: .env
# API Keys
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_DIMENSION=1536

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false
PORT=8000

# Enhanced Features
EXTRACT_PDF_METADATA=true
ENRICH_WITH_CROSSREF=true
CROSSREF_EMAIL=your-email@domain.com

# Performance
WORKERS=4
MAX_REQUESTS=1000
```

## 🔧 **Step 4: API Management Scripts**

### **4.1 Start API Script**
```bash
# File: start_api.sh
cat > start_api.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate

# Create logs directory
mkdir -p logs

# Start the API with Gunicorn
gunicorn main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --daemon \
  --pid genomics-api.pid \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info

echo "✅ Diabetes Research API started on localhost:8000"
echo "PID: $(cat genomics-api.pid)"
echo "Logs: /home/ubuntu/genomics-app/logs/"
echo "Test: curl http://localhost:8000/health"
EOF

chmod +x start_api.sh
```

### **4.2 Stop API Script**
```bash
# File: stop_api.sh
cat > stop_api.sh << 'EOF'
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
EOF

chmod +x stop_api.sh
```

### **4.3 Restart API Script**
```bash
# File: restart_api.sh
cat > restart_api.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/genomics-app
echo "🔄 Restarting Diabetes Research API..."
./stop_api.sh
sleep 2
./start_api.sh
EOF

chmod +x restart_api.sh
```

### **4.4 Status Check Script**
```bash
# File: status.sh
cat > status.sh << 'EOF'
#!/bin/bash
echo "=== Diabetes Research API Status ==="
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
echo "=== Quick health check ==="
curl -s http://localhost:8000/health || echo "❌ Health check failed"
EOF

chmod +x status.sh
```

## 🌐 **Step 5: Nginx Production Configuration**

### **5.1 Nginx Site Configuration**
```bash
# File: /etc/nginx/sites-available/diabetes-research
sudo tee /etc/nginx/sites-available/diabetes-research << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline' fonts.googleapis.com fonts.gstatic.com" always;
    
    # Root directory for React build
    root /home/ubuntu/genomics-app/frontend/build;
    index index.html index.htm;
    
    # API endpoints - proxy /api/* to FastAPI backend
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_http_version 1.1;
        proxy_intercept_errors on;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin "*";
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization";
            add_header Access-Control-Max-Age 1728000;
            add_header Content-Type 'text/plain; charset=utf-8';
            add_header Content-Length 0;
            return 204;
        }
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Direct API access (for testing)
    location ~ ^/(health|status|docs|redoc|openapi.json|search|query|models)$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static assets with caching
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri $uri/ =404;
    }
    
    # Favicon and assets
    location ~* \.(ico|png|jpg|jpeg|gif|svg|webp|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    
    # CSS and JS files
    location ~* \.(css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    
    # React Router - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
        
        # Disable caching for HTML files
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
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
EOF
```

### **5.2 Enable Nginx Configuration**
```bash
# Remove default site and enable our configuration
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/diabetes-research /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Enable and start nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

## 🚀 **Step 6: Build and Deploy**

### **6.1 Build React Application**
```bash
cd /home/ubuntu/genomics-app/frontend
npm run build

# Set proper permissions
sudo chown -R ubuntu:ubuntu build/
sudo chmod -R 755 build/

# Verify build
ls -la build/
```

### **6.2 Start Backend Services**
```bash
cd /home/ubuntu/genomics-app

# Start the API
./start_api.sh

# Restart nginx
sudo systemctl restart nginx
```

## 🧪 **Step 7: Production Testing**

### **7.1 Comprehensive Test Script**
```bash
# File: test_production.sh
cat > test_production.sh << 'EOF'
#!/bin/bash
echo "🩺 Testing Diabetes Research Assistant Production Deployment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="$3"
    
    echo -n "Testing $name... "
    status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} (Status: $status)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC} (Status: $status, Expected: $expected_status)"
        ((TESTS_FAILED++))
    fi
}

echo -e "\n${YELLOW}🌐 Testing Frontend${NC}"
test_endpoint "React App" "http://localhost/" "200"

echo -e "\n${YELLOW}🔌 Testing API${NC}"
test_endpoint "Health Check" "http://localhost/api/health" "200"
test_endpoint "Status Check" "http://localhost/api/status" "200"
test_endpoint "API Docs" "http://localhost/api/docs" "200"

echo -e "\n${YELLOW}🤖 Testing AI Functionality${NC}"
echo -n "Testing Query Endpoint... "
response=$(curl -s -X POST "http://localhost/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is diabetes?", "model": "gpt-3.5-turbo", "top_k": 2}')

if echo "$response" | grep -q "query\|llm_response"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}📊 Services Status${NC}"
if pgrep -f "gunicorn.*main:app" > /dev/null; then
    echo -e "${GREEN}✅ FastAPI Running${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FastAPI Not Running${NC}"
    ((TESTS_FAILED++))
fi

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx Running${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Nginx Not Running${NC}"
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}📋 Test Summary${NC}"
echo "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo "Tests Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! Production deployment ready!${NC}"
    
    # Get public IP
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_SERVER_IP")
    echo -e "\n${YELLOW}🌐 Access your application:${NC}"
    echo "   Frontend: http://$PUBLIC_IP/"
    echo "   API Docs: http://$PUBLIC_IP/api/docs"
    echo "   Health:   http://$PUBLIC_IP/api/health"
else
    echo -e "\n${RED}❌ Some tests failed. Check logs and configuration.${NC}"
fi
EOF

chmod +x test_production.sh
./test_production.sh
```

## 🔧 **Step 8: Maintenance & Monitoring**

### **8.1 Log Monitoring**
```bash
# View API logs
tail -f logs/error.log

# View nginx logs
sudo tail -f /var/log/nginx/error.log

# Check system resources
htop
df -h
```

### **8.2 Maintenance Commands**
```bash
# Restart services
./restart_api.sh
sudo systemctl restart nginx

# Update application
git pull  # if using git
./restart_api.sh

# Check status
./status.sh
```

Your production-ready diabetes research assistant is now fully configured and ready for deployment! 🎉
