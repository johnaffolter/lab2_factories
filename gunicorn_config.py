# Gunicorn configuration to handle large headers
import multiprocessing

# Worker configuration
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Request limits - INCREASED FOR LARGE HEADERS
limit_request_line = 16384  # 16KB (default is 4094)
limit_request_fields = 200  # Maximum number of headers (default is 100)
limit_request_field_size = 16384  # 16KB per header (default is 8190)

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")
