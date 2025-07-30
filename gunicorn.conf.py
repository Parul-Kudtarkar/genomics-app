import multiprocessing

# Server socket
bind = "127.0.0.1:8000"  # Only local access (nginx will proxy)
backlog = 2048

# Worker processes
workers = 2  # Reduced for stability
worker_class = "uvicorn.workers.UvicornWorker"  # FastAPI requires this
worker_connections = 500
timeout = 60  # Increased timeout for stability
keepalive = 5  # Increased keepalive for stability

# Restart workers after this many requests (prevents memory leaks)
max_requests = 1000  # Reduced for stability
max_requests_jitter = 50  # Reduced jitter

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

