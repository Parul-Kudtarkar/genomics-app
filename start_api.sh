#!/bin/bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
gunicorn main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --daemon \
  --pid /home/ubuntu/genomics-app/genomics-api.pid \
  --access-logfile /home/ubuntu/genomics-app/logs/access.log \
  --error-logfile /home/ubuntu/genomics-app/logs/error.log \
  --log-level info

echo "✅ Genomics API started"
echo "PID: $(cat /home/ubuntu/genomics-app/genomics-api.pid)"
echo "Logs: /home/ubuntu/genomics-app/logs/"
echo "Test: curl http://localhost:8000/health"
