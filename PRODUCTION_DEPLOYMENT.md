# 🚀 Production Deployment Guide - Diabetes Research Assistant

This guide will walk you through deploying your diabetes research assistant with Auth0 authentication to production.

## 🎯 **What We're Deploying**

- **React Frontend** with Auth0 authentication
- **FastAPI Backend** with JWT validation
- **Auth0 Authentication** with role-based access
- **Nginx Reverse Proxy** with SSL
- **Production Security** and monitoring

## 📋 **Prerequisites**

1. **VPS/Cloud Server** (Ubuntu 20.04+ recommended)
2. **Domain Name** (e.g., `yourdomain.com`)
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

# Install PM2 for process management
sudo npm install -g pm2
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
AUTH0_AUDIENCE=https://your-api-identifier
AUTH0_ISSUER=https://your-domain.auth0.com/
EOF

# Create logs directory
mkdir -p logs
```

### **2.3 Frontend Setup**

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Create production environment file
cat > .env << 'EOF'
# Auth0 Configuration
REACT_APP_AUTH0_DOMAIN=your-domain.auth0.com
REACT_APP_AUTH0_CLIENT_ID=your-client-id
REACT_APP_AUTH0_AUDIENCE=https://your-api-identifier
REACT_APP_AUTH0_REDIRECT_URI=https://yourdomain.com

# API Configuration
REACT_APP_API_BASE_URL=https://yourdomain.com
EOF

# Build for production
npm run build

# Set proper permissions
sudo chown -R ubuntu:ubuntu build/
sudo chmod -R 755 build/
```

## 🚀 **Step 3: Process Management**

### **3.1 Create PM2 Configuration**

```bash
# Create PM2 ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'genomics-api',
    script: 'main.py',
    cwd: '/home/ubuntu/genomics-app',
    interpreter: '/home/ubuntu/genomics-app/venv/bin/python',
    instances: 2,
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 8000
    },
    error_file: '/home/ubuntu/genomics-app/logs/err.log',
    out_file: '/home/ubuntu/genomics-app/logs/out.log',
    log_file: '/home/ubuntu/genomics-app/genomics-app/logs/combined.log',
    time: true,
    max_memory_restart: '1G',
    restart_delay: 4000,
    max_restarts: 10
  }]
};
EOF

# Start the application
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### **3.2 Create Systemd Service (Alternative)**

```bash
# Create systemd service file
sudo tee /etc/systemd/system/genomics-api.service << 'EOF'
[Unit]
Description=Genomics Research API
After=network.target

[Service]
Type=exec
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/genomics-app
Environment=PATH=/home/ubuntu/genomics-app/venv/bin
ExecStart=/home/ubuntu/genomics-app/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable genomics-api
sudo systemctl start genomics-api
```

## 🚀 **Step 4: Nginx Configuration**

### **4.1 Create Nginx Site Configuration**

```bash
# Create Nginx site configuration
sudo tee /etc/nginx/sites-available/genomics-research << 'EOF'
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
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
        add_header Access-Control-Allow-Origin "https://yourdomain.com" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        add_header Access-Control-Allow-Credentials "true" always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin "https://yourdomain.com";
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
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

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
   Allowed Callback URLs: https://yourdomain.com
   Allowed Logout URLs: https://yourdomain.com
   Allowed Web Origins: https://yourdomain.com
   ```

3. **Update API Settings**
   - Go to "APIs" → Your API
   - Verify audience: `https://your-api-identifier`

### **6.2 Update Environment Variables**

```bash
# Update frontend environment
cd /home/diabetes-app/diabetes-research/frontend
sed -i 's|http://localhost:3000|https://yourdomain.com|g' .env
sed -i 's|http://localhost:8000|https://yourdomain.com|g' .env

# Rebuild frontend
npm run build

# Update backend environment
cd /home/diabetes-app/diabetes-research
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
# Add security headers to Nginx
sudo tee /etc/nginx/conf.d/security-headers.conf << 'EOF'
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' https: data: blob: 'unsafe-inline' 'unsafe-eval' https://your-domain.auth0.com https://cdn.auth0.com;" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
EOF

# Reload Nginx
sudo systemctl reload nginx
```

### **7.3 Rate Limiting**

```bash
# Install rate limiting module
sudo apt install -y nginx-extras

# Add rate limiting to Nginx
sudo tee /etc/nginx/conf.d/rate-limiting.conf << 'EOF'
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

# Apply to API endpoints
location /api/ {
    limit_req zone=api burst=20 nodelay;
    # ... existing proxy configuration
}

# Apply to login endpoints
location /auth/ {
    limit_req zone=login burst=5 nodelay;
    # ... existing configuration
}
EOF
```

## 🚀 **Step 8: Monitoring and Logging**

### **8.1 Application Monitoring**

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor application with PM2
pm2 monit

# View logs
pm2 logs diabetes-api

# Check application status
pm2 status
```

### **8.2 Log Rotation**

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/diabetes-api << 'EOF'
/home/diabetes-app/diabetes-research/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 diabetes-app diabetes-app
    postrotate
        pm2 reloadLogs
    endscript
}
EOF
```

### **8.3 Health Monitoring**

```bash
# Create health check script
cat > /home/ubuntu/health-check.sh << 'EOF'
#!/bin/bash

# Check if API is responding
if curl -f -s https://yourdomain.com/api/health > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is down"
    # Restart the application
    pm2 restart genomics-api
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️  Disk usage is high: ${DISK_USAGE}%"
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "⚠️  Memory usage is high: ${MEMORY_USAGE}%"
fi
EOF

chmod +x /home/ubuntu/health-check.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/ubuntu/health-check.sh") | crontab -
```

## 🚀 **Step 9: Backup Strategy**

### **9.1 Database Backup**

```bash
# Create backup script
cat > /home/ubuntu/backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application files
tar -czf $BACKUP_DIR/app_$DATE.tar.gz /home/ubuntu/genomics-app

# Backup environment files
cp /home/ubuntu/genomics-app/.env $BACKUP_DIR/env_$DATE.backup

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.backup" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /home/ubuntu/backup.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /home/ubuntu/backup.sh") | crontab -
```

## 🚀 **Step 10: Testing Production Deployment**

### **10.1 Test Script**

```bash
# Create comprehensive test script
cat > /home/ubuntu/test-production.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Production Deployment"
echo "================================"

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
test_endpoint "React App" "https://yourdomain.com/" "200"

echo -e "\n${YELLOW}🔌 Testing API${NC}"
test_endpoint "Health Check" "https://yourdomain.com/api/health" "200"
test_endpoint "Status Check" "https://yourdomain.com/api/status" "200"

echo -e "\n${YELLOW}🔒 Testing Security${NC}"
test_endpoint "Unauthenticated Query" "https://yourdomain.com/api/query" "401"

echo -e "\n${YELLOW}📊 Services Status${NC}"
if pm2 list | grep -q "genomics-api.*online"; then
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
    echo -e "\n${GREEN}🎉 Production deployment successful!${NC}"
    echo -e "\n${YELLOW}🌐 Access your application:${NC}"
    echo "   Frontend: https://yourdomain.com/"
    echo "   API Docs: https://yourdomain.com/api/docs"
    echo "   Health:   https://yourdomain.com/api/health"
else
    echo -e "\n${RED}❌ Some tests failed. Check configuration.${NC}"
fi
EOF

chmod +x /home/ubuntu/test-production.sh
./test-production.sh
```

## 🔧 **Step 11: Maintenance Commands**

### **11.1 Application Management**

```bash
# Restart application
pm2 restart genomics-api

# View logs
pm2 logs genomics-api --lines 100

# Monitor resources
pm2 monit

# Update application
cd /home/ubuntu/genomics-app
git pull
npm install  # if frontend changes
pip install -r requirements.txt  # if backend changes
pm2 restart genomics-api
```

### **11.2 System Maintenance**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep -E "(genomics|nginx|pm2)"

# View system logs
sudo journalctl -u nginx -f
sudo journalctl -u genomics-api -f
```

## ✅ **Deployment Checklist**

- [ ] Server setup completed
- [ ] Application deployed
- [ ] Environment variables configured
- [ ] Auth0 production settings updated
- [ ] Nginx configured with SSL
- [ ] PM2 process management setup
- [ ] Security headers implemented
- [ ] Rate limiting configured
- [ ] Monitoring and logging setup
- [ ] Backup strategy implemented
- [ ] Production tests passed
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Health checks working

## 🚨 **Troubleshooting**

### **Common Issues**

1. **SSL Certificate Issues**
   ```bash
   sudo certbot --nginx -d yourdomain.com
   sudo nginx -t && sudo systemctl reload nginx
   ```

2. **Application Not Starting**
   ```bash
   pm2 logs genomics-api
   pm2 restart genomics-api
   ```

3. **Nginx Configuration Errors**
   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. **Auth0 Authentication Issues**
   - Check Auth0 callback URLs
   - Verify environment variables
   - Check browser console for errors

### **Performance Optimization**

```bash
# Enable Nginx gzip compression
sudo nano /etc/nginx/nginx.conf

# Add to http block:
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
```

Your diabetes research assistant is now production-ready with enterprise-grade security, monitoring, and scalability! 🎉

## 📞 **Support**

For issues:
1. Check application logs: `pm2 logs diabetes-api`
2. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
3. Check system logs: `sudo journalctl -xe`
4. Monitor resources: `htop` and `pm2 monit`

The system is now ready for production use with Auth0 authentication! 🔒 