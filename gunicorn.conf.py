import multiprocessing

# Server socket
bind = "127.0.0.1:8000"  # Only local access (nginx will proxy)
backlog = 2048

# Worker processes
workers = 4  # Increased for better concurrency
worker_class = "uvicorn.workers.UvicornWorker"  # FastAPI requires this
worker_connections = 1000
timeout = 45  # Further reduced timeout for faster responses
keepalive = 3  # Reduced keepalive for better performance

# Restart workers after this many requests (prevents memory leaks)
max_requests = 2000  # Increased for better performance
max_requests_jitter = 100  # Increased jitter

# Logging
loglevel = "info"
accesslog = "/home/ubuntu/genomics-app/logs/access.log"
errorlog = "/home/ubuntu/genomics-app/logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'genomics_api'

# Daemon
daemon = True
pidfile = "/home/ubuntu/genomics-app/genomics-api.pid"
user = "ubuntu"
group = "ubuntu"
tmp_upload_dir = None

# SSL (if you want HTTPS directly)
# keyfile = "/path/to/private.key"
# certfile = "/path/to/certificate.crt"

