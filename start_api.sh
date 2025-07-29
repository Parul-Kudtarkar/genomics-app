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
  --pid genomics-api.pid

echo "✅ Genomics API started on localhost:8000"
echo "PID: $(cat genomics-api.pid)"
echo "Logs: /home/ubuntu/genomics-app/logs/"
echo "Redis: $(systemctl is-active redis-server)"
echo "Test: curl http://localhost:8000/health"
