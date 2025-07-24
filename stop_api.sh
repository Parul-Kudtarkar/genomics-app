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
