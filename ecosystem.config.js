module.exports = {
  apps: [{
    name: 'genomics-api',
    script: 'main.py',
    interpreter: '/home/ubuntu/venv/bin/python',
    cwd: '/home/ubuntu/genomics-app',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: 8000
    },
    error_file: '/home/ubuntu/genomics-app/logs/err.log',
    out_file: '/home/ubuntu/genomics-app/logs/out.log',
    log_file: '/home/ubuntu/genomics-app/logs/combined.log',
    time: true
  }]
}; 