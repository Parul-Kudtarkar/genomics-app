# 🚀 Production Deployment Guide - Diabetes Research Assistant

This guide will walk you through deploying your diabetes research assistant with Auth0 authentication to production.

## 🎯 **What We're Deploying**

- **React Frontend** with Auth0 authentication
- **FastAPI Backend** with JWT validation (using existing start script)
- **Auth0 Authentication** with role-based access
- **Nginx Reverse Proxy** with SSL
- **Production Security** and monitoring

## 📋 **Prerequisites**

1. **VPS/Cloud Server** (Ubuntu 20.04+ recommended)
2. **Domain Name** (e.g., `lit-koi.pankbase.org`)
3. **Auth0 Account** (configured per AUTH0_SETUP.md)
4. **SSL Certificate** (Let's Encrypt recommended)
5. **Git Repository** with your code

## 🚀 **Step 1: Server Setup**

### **1.1 Initial Server Configuration**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release

# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.9+
sudo apt install -y python3 python3-pip python3-venv

# Install Nginx
sudo apt install -y nginx

# Install Certbot for SSL
sudo apt install -y certbot python3-certbot-nginx

# Install Gunicorn for Python process management
sudo apt install -y gunicorn
```

### **1.2 Create Application User**

```bash
# Create application user
sudo useradd -m -s /bin/bash ubuntu
sudo usermod -aG sudo ubuntu

# Switch to application user
sudo su - ubuntu

# Create application directory
mkdir -p /home/ubuntu/genomics-app
cd /home/ubuntu/genomics-app
```

## 🚀 **Step 2: Application Deployment**

### **2.1 Clone and Setup Application**

```bash
# Clone your repository
git clone https://github.com/yourusername/genomics-app.git .
# OR upload your files manually

# Set proper permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/genomics-app
```

### **2.2 Backend Setup**

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create production environment file
cat > .env << 'EOF'
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

# Auth0 Configuration
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_AUDIENCE=https://lit-koi.pankbase.org/api
AUTH0_ISSUER=https://your-domain.auth0.com/
EOF

# Create logs directory
mkdir -p logs
```

### **2.3 Frontend Setup**

```bash
# Install frontend dependencies
cd frontend
npm install

# Create frontend environment file
cat > .env << 'EOF'
# Auth0 Configuration
REACT_APP_AUTH0_DOMAIN=your-domain.auth0.com
REACT_APP_AUTH0_CLIENT_ID=your-client-id
REACT_APP_AUTH0_AUDIENCE=https://lit-koi.pankbase.org/api
REACT_APP_AUTH0_REDIRECT_URI=https://lit-koi.pankbase.org

# API Configuration
REACT_APP_API_BASE_URL=https://lit-koi.pankbase.org/api
EOF

# Build frontend for production
npm run build

# Set proper permissions
sudo chown -R ubuntu:ubuntu build/
sudo chmod -R 755 build/
```

## 🚀 **Step 3: Process Management (Using Existing Scripts)**

### **3.1 Use Existing Start Script**

```bash
# Make scripts executable
cd /home/ubuntu/genomics-app
chmod +x start_api.sh stop_api.sh restart_api.sh

# Start the API using existing script
./start_api.sh

# Verify it's running
curl http://localhost:8000/health
```

### **3.2 Alternative: Systemd Service (Recommended for Production)**

```bash
# Create systemd service file
sudo tee /etc/systemd/system/genomics-api.service << 'EOF'
[Unit]
Description=Genomics Research API
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/genomics-app
Environment=PATH=/home/ubuntu/venv/bin
ExecStart=/home/ubuntu/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable genomics-api
sudo systemctl start genomics-api

# Check status
sudo systemctl status genomics-api
```

## 🚀 **Step 4: Nginx Configuration**

### **4.1 Create Nginx Site Configuration**

```bash
# Create Nginx site configuration
sudo tee /etc/nginx/sites-available/genomics-research << 'EOF'
server {
    listen 80;
    server_name lit-koi.pankbase.org www.lit-koi.pankbase.org;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' https: data: blob: 'unsafe-inline' 'unsafe-eval' https://your-domain.auth0.com https://cdn.auth0.com;" always;
    
    # Root directory for React build
    root /home/ubuntu/genomics-app/frontend/build;
    index index.html index.htm;
    
    # API endpoints - proxy to FastAPI backend
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
        
        # CORS headers for Auth0
        add_header Access-Control-Allow-Origin "https://lit-koi.pankbase.org" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        add_header Access-Control-Allow-Credentials "true" always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin "https://lit-koi.pankbase.org";
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization";
            add_header Access-Control-Allow-Credentials "true";
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

# Enable the site
sudo ln -s /etc/nginx/sites-available/genomics-research /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Start Nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

## 🚀 **Step 5: SSL Certificate**

### **5.1 Install SSL Certificate**

```bash
# Install SSL certificate with Let's Encrypt
sudo certbot --nginx -d lit-koi.pankbase.org -d www.lit-koi.pankbase.org

# Test automatic renewal
sudo certbot renew --dry-run

# Set up automatic renewal
sudo crontab -e
# Add this line:
# 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🚀 **Step 6: Auth0 Production Configuration**

### **6.1 Update Auth0 Settings**

1. **Login to Auth0 Dashboard**
   - Go to [manage.auth0.com](https://manage.auth0.com)

2. **Update Application Settings**
   ```
   Allowed Callback URLs: https://lit-koi.pankbase.org
   Allowed Logout URLs: https://lit-koi.pankbase.org
   Allowed Web Origins: https://lit-koi.pankbase.org
   ```

3. **Update API Settings**
   - Go to "APIs" → Your API
   - Verify audience: `https://lit-koi.pankbase.org/api`

### **6.2 Update Environment Variables**

```bash
# Update frontend environment
cd /home/ubuntu/genomics-app/frontend
sed -i 's|http://localhost:3000|https://lit-koi.pankbase.org|g' .env
sed -i 's|http://localhost:8000|https://lit-koi.pankbase.org|g' .env

# Rebuild frontend
npm run build

# Update backend environment
cd /home/ubuntu/genomics-app
# Update .env with production Auth0 settings
```

## 🚀 **Step 7: Security Hardening**

### **7.1 Firewall Configuration**

```bash
# Configure UFW firewall
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Check firewall status
sudo ufw status
```

### **7.2 Security Headers**

```bash
# Add security headers to Nginx (already included in config above)
# The configuration includes:
# - X-Frame-Options
# - X-Content-Type-Options
# - X-XSS-Protection
# - Content-Security-Policy
```

## 🚀 **Step 8: Monitoring and Maintenance**

### **8.1 Service Management**

```bash
# Check service status
sudo systemctl status genomics-api
sudo systemctl status nginx

# View logs
sudo journalctl -u genomics-api -f
sudo tail -f /var/log/nginx/error.log

# Restart services
sudo systemctl restart genomics-api
sudo systemctl restart nginx
```

### **8.2 Application Logs**

```bash
# View application logs
tail -f /home/ubuntu/genomics-app/logs/error.log
tail -f /home/ubuntu/genomics-app/logs/access.log

# Check API health
curl https://lit-koi.pankbase.org/api/health
```

### **8.3 Performance Monitoring**

```bash
# Monitor system resources
htop
df -h
free -h

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s https://lit-koi.pankbase.org/api/health
```

## 🚀 **Step 9: Testing Production Deployment**

### **9.1 Comprehensive Test Script**

```bash
# Create test script
cat > /home/ubuntu/genomics-app/test_production.sh << 'EOF'
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
test_endpoint "React App" "https://lit-koi.pankbase.org/" "200"

echo -e "\n${YELLOW}🔌 Testing API${NC}"
test_endpoint "Health Check" "https://lit-koi.pankbase.org/api/health" "200"
test_endpoint "Status Check" "https://lit-koi.pankbase.org/api/status" "200"
test_endpoint "API Docs" "https://lit-koi.pankbase.org/api/docs" "200"

echo -e "\n${YELLOW}🤖 Testing AI Functionality${NC}"
echo -n "Testing Query Endpoint... "
response=$(curl -s -X POST "https://lit-koi.pankbase.org/api/query" \
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
if systemctl is-active --quiet genomics-api; then
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
    
    echo -e "\n${YELLOW}🌐 Access your application:${NC}"
    echo "   Frontend: https://lit-koi.pankbase.org/"
    echo "   API Docs: https://lit-koi.pankbase.org/api/docs"
    echo "   Health:   https://lit-koi.pankbase.org/api/health"
else
    echo -e "\n${RED}❌ Some tests failed. Check logs and configuration.${NC}"
fi
EOF

chmod +x /home/ubuntu/genomics-app/test_production.sh
./test_production.sh
```

## ✅ **Verification Checklist**

- [ ] Server setup complete (Ubuntu, Node.js, Python, Nginx)
- [ ] Application deployed to `/home/ubuntu/genomics-app`
- [ ] Python virtual environment created and dependencies installed
- [ ] Frontend built and deployed
- [ ] Backend running with existing start script or systemd service
- [ ] Nginx configured and serving frontend
- [ ] API proxying working (`/api/*` → FastAPI)
- [ ] SSL certificate installed and working
- [ ] Auth0 configured with production URLs
- [ ] Environment variables set for production
- [ ] Firewall configured
- [ ] Monitoring and logging set up
- [ ] All tests passing

## 🎉 **Your Application is Now Live!**

**Frontend**: https://lit-koi.pankbase.org/
**API Docs**: https://lit-koi.pankbase.org/api/docs
**Health Check**: https://lit-koi.pankbase.org/api/health

## 📞 **Support and Maintenance**

### **Common Commands**
```bash
# Restart services
sudo systemctl restart genomics-api
sudo systemctl restart nginx

# View logs
sudo journalctl -u genomics-api -f
sudo tail -f /var/log/nginx/error.log

# Update application
cd /home/ubuntu/genomics-app
git pull
./restart_api.sh
npm run build  # in frontend directory

# Check status
./test_production.sh
```

Your diabetes research assistant is now production-ready with enterprise-grade security! 🚀 