#!/bin/bash
cd /home/ubuntu/genomics-app
echo "🔄 Restarting API..."
./stop_api.sh
sleep 2
./start_api.sh
