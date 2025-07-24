#!/bin/bash

# Check service status
echo "=== Service Status ==="
sudo systemctl status genomics-api --no-pager

echo -e "\n=== Recent Logs ==="
sudo journalctl -u genomics-api --since "1 hour ago" --no-pager | tail -20

echo -e "\n=== API Health Check ==="
curl -s http://localhost/api/health | python3 -m json.tool

echo -e "\n=== Disk Usage ==="
df -h /home/ubuntu/genomics-app/logs/

echo -e "\n=== Memory Usage ==="
free -h

echo -e "\n=== Process Info ==="
ps aux | grep gunicorn
