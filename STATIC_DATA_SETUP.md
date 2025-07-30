# Static Vector Store Data Setup

This setup allows you to generate static JSON files with all vector store contents for instant frontend loading.

## 🚀 Quick Setup

### 1. Generate Static Data
```bash
# Run the generation script
python scripts/generate_vector_store_contents.py
```

This will create:
- `static/vector_store_contents.json` - Full data with all papers
- `static/vector_store_preview.json` - Lightweight preview

### 2. Configure Nginx
Add this to your Nginx configuration (`/etc/nginx/sites-available/genomics-app`):

```nginx
location /static/ {
    alias /home/ubuntu/genomics-app/static/;
    expires 1h;
    add_header Cache-Control "public, immutable";
    add_header Access-Control-Allow-Origin "*";
    
    location ~* \.json$ {
        add_header Content-Type "application/json";
    }
}
```

Then reload Nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 3. Rebuild Frontend
```bash
cd frontend
npm run build
```

## 📊 Benefits

### ⚡ Performance
- **Instant loading** - No API calls needed
- **Cached data** - Browser caches JSON files
- **Reduced server load** - No backend queries for contents

### 🎯 User Experience
- **Immediate display** - Contents show instantly
- **Offline capability** - Works without backend
- **Faster navigation** - No loading delays

### 🔧 Maintenance
- **Scheduled updates** - Run script periodically
- **Version control** - Track data changes
- **Easy debugging** - Static files are inspectable

## 🔄 Update Process

### Manual Update
```bash
# Regenerate data
python scripts/generate_vector_store_contents.py

# Rebuild frontend (if needed)
cd frontend && npm run build
```

### Automated Update (Cron)
```bash
# Add to crontab for daily updates
0 2 * * * cd /home/ubuntu/genomics-app && python scripts/generate_vector_store_contents.py
```

## 📁 File Structure

```
genomics-app/
├── static/
│   ├── vector_store_contents.json    # Full data
│   └── vector_store_preview.json    # Preview data
├── scripts/
│   └── generate_vector_store_contents.py
└── frontend/
    └── src/
        └── components/
            └── VectorStore/
                └── StaticVectorStoreContents.js
```

## 🎨 Features

### Generated Data Includes:
- **Paper metadata** - Titles, sources, chunk counts
- **Statistics** - Total papers, chunks, breakdowns
- **Search index** - Quick lookup capabilities
- **Timestamps** - Generation and update times

### Frontend Features:
- **Instant loading** - No API delays
- **Refresh capability** - Manual refresh button
- **Error handling** - Graceful fallbacks
- **Status display** - Generation timestamps

## 🔍 Monitoring

### Check Data Status
```bash
# View generated data
ls -la static/

# Check file sizes
du -h static/*.json

# View statistics
jq '.statistics' static/vector_store_contents.json
```

### Monitor Updates
```bash
# Check last generation time
jq '.metadata.generated_at' static/vector_store_contents.json

# Monitor file changes
watch -n 60 'ls -la static/'
```

## 🛠️ Troubleshooting

### Common Issues:

1. **File not found**
   ```bash
   # Ensure static directory exists
   mkdir -p static/
   python scripts/generate_vector_store_contents.py
   ```

2. **Nginx 404 errors**
   ```bash
   # Check Nginx configuration
   sudo nginx -t
   # Check file permissions
   ls -la static/
   ```

3. **Frontend loading errors**
   ```bash
   # Check browser console
   # Verify file is accessible
   curl https://your-domain.com/static/vector_store_contents.json
   ```

## 📈 Performance Metrics

### Expected Performance:
- **Load time**: < 100ms (vs 2-5 seconds with API)
- **File size**: ~50-200KB (depending on papers)
- **Cache hit rate**: 95%+ (browser cached)
- **Server load**: 0% (no backend queries)

### Monitoring Commands:
```bash
# Check file sizes
du -h static/*.json

# Monitor access logs
tail -f /var/log/nginx/access.log | grep static

# Check browser cache headers
curl -I https://your-domain.com/static/vector_store_contents.json
``` 