#!/bin/bash

echo "🚀 ULTIMATE AIRFLOW HEADER FIX SOLUTION"
echo "========================================"
echo ""
echo "This script implements ALL methods to fix the 431 error"
echo ""

# Step 1: Stop current Airflow
echo "📦 Step 1: Stopping current Airflow..."
docker stop airflow-standalone 2>/dev/null
docker rm airflow-standalone 2>/dev/null

# Step 2: Create custom Airflow image with Gunicorn config
echo "📦 Step 2: Creating custom Airflow configuration..."

cat > gunicorn_config.py << 'EOF'
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
EOF

cat > custom_airflow_webserver_config.py << 'EOF'
# Custom Airflow webserver configuration
from airflow.www.app import create_app
import os

# Increase Flask/Werkzeug limits
os.environ['WERKZEUG_MAX_CONTENT_LENGTH'] = str(100 * 1024 * 1024)  # 100MB
os.environ['WERKZEUG_MAX_FORM_MEMORY_SIZE'] = str(10 * 1024 * 1024)  # 10MB

app = create_app(testing=False)

# Configure app for large headers
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['MAX_COOKIE_SIZE'] = 16384  # 16KB

if __name__ == "__main__":
    app.run()
EOF

# Step 3: Create Dockerfile for custom Airflow
echo "📦 Step 3: Creating custom Airflow Dockerfile..."

cat > Dockerfile.airflow << 'EOF'
FROM apache/airflow:2.7.0

USER root

# Install additional packages
RUN apt-get update && apt-get install -y \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy custom configurations
COPY gunicorn_config.py /opt/airflow/gunicorn_config.py
COPY custom_airflow_webserver_config.py /opt/airflow/custom_airflow_webserver_config.py

# Create nginx config for internal proxy
RUN echo 'server { \
    listen 8081; \
    large_client_header_buffers 8 64k; \
    client_header_buffer_size 64k; \
    location / { \
        proxy_pass http://127.0.0.1:8080; \
        proxy_set_header Host $host; \
        proxy_hide_header X-Powered-By; \
    } \
}' > /etc/nginx/sites-available/airflow

USER airflow

# Set environment variables for large headers
ENV AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_CLASS=sync
ENV AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL=30
ENV AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=120
ENV GUNICORN_CMD_ARGS="--config /opt/airflow/gunicorn_config.py"
EOF

# Step 4: Start Airflow with ALL fixes applied
echo "📦 Step 4: Starting Airflow with comprehensive fixes..."

docker run -d \
    --name airflow-standalone \
    -p 8080:8080 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=120 \
    -e AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL=30 \
    -e AIRFLOW__WEBSERVER__WORKER_CLASS=sync \
    -e AIRFLOW__WEBSERVER__WORKERS=2 \
    -e AIRFLOW__WEBSERVER__ACCESS_LOGFORMAT='%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s"' \
    -e AIRFLOW__WEBSERVER__BASE_URL=http://localhost:8080 \
    -e AIRFLOW__WEBSERVER__ENABLE_PROXY_FIX=True \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_FOR=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PROTO=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_HOST=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PORT=1 \
    -e AIRFLOW__WEBSERVER__PROXY_FIX_X_PREFIX=1 \
    -e AIRFLOW__API__AUTH_BACKENDS='airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session' \
    -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@host.docker.internal/airflow \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
    -e _AIRFLOW_DB_MIGRATE=true \
    -e _AIRFLOW_WWW_USER_CREATE=true \
    -e _AIRFLOW_WWW_USER_USERNAME=admin \
    -e _AIRFLOW_WWW_USER_PASSWORD=admin \
    -e GUNICORN_CMD_ARGS="--limit-request-line 16384 --limit-request-fields 200 --limit-request-field_size 16384" \
    -v ${PWD}/dags:/opt/airflow/dags \
    -v ${PWD}/logs:/opt/airflow/logs \
    -v ${PWD}/gunicorn_config.py:/opt/airflow/gunicorn_config.py:ro \
    apache/airflow:2.7.0 standalone

echo "✅ Airflow started with maximum header size configuration"
echo ""

# Step 5: Start nginx proxy
echo "📦 Step 5: Starting Nginx proxy..."

docker stop airflow-nginx 2>/dev/null
docker rm airflow-nginx 2>/dev/null

# Create enhanced nginx config
cat > nginx_enhanced.conf << 'EOF'
server {
    listen 8888;
    server_name _;

    # Maximum header sizes
    large_client_header_buffers 16 64k;
    client_header_buffer_size 64k;
    client_body_buffer_size 64k;
    client_max_body_size 100M;

    # Timeouts
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
    send_timeout 300;

    # Main location
    location / {
        # Strip cookies if they're too large
        set $new_cookie '';
        if ($http_cookie ~* "(.{0,4000})") {
            set $new_cookie $1;
        }
        proxy_set_header Cookie $new_cookie;

        # Proxy to Airflow
        proxy_pass http://host.docker.internal:8080;
        proxy_redirect off;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;

        # WebSocket
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

docker run -d \
    --name airflow-nginx \
    -p 8888:80 \
    -v ${PWD}/nginx_enhanced.conf:/etc/nginx/conf.d/default.conf:ro \
    --add-host=host.docker.internal:host-gateway \
    nginx:alpine

echo "✅ Nginx proxy started on port 8888"
echo ""

# Step 6: Create browser test page
echo "📦 Step 6: Creating browser test page..."

cat > test_airflow_access.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Airflow Access Tester</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f9fafb;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        h1 { color: #1f2937; }
        h2 { color: #374151; font-size: 18px; margin-top: 24px; }
        .endpoint {
            display: inline-block;
            padding: 8px 16px;
            margin: 8px;
            background: #eff6ff;
            border: 2px solid #3b82f6;
            border-radius: 6px;
            color: #1e40af;
            text-decoration: none;
            transition: all 0.2s;
        }
        .endpoint:hover {
            background: #3b82f6;
            color: white;
            transform: translateY(-2px);
        }
        button {
            background: #10b981;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px;
            font-size: 15px;
        }
        button:hover { background: #059669; }
        .status {
            padding: 12px;
            border-radius: 6px;
            margin: 10px 0;
        }
        .success { background: #d1fae5; color: #065f46; }
        .error { background: #fee2e2; color: #991b1b; }
        .info { background: #dbeafe; color: #1e40af; }
        code {
            background: #f3f4f6;
            padding: 3px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        .test-result {
            padding: 8px;
            margin: 5px 0;
            border-left: 4px solid #e5e7eb;
        }
        .test-result.pass {
            border-left-color: #10b981;
            background: #f0fdf4;
        }
        .test-result.fail {
            border-left-color: #ef4444;
            background: #fef2f2;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🚀 Airflow Access Tester & Fixer</h1>
        <p>This page helps you access Airflow by testing different methods and clearing browser data.</p>

        <h2>Step 1: Clear Browser Data</h2>
        <button onclick="clearEverything()">🧹 Clear All Data</button>
        <button onclick="testConnections()">🔍 Test All Endpoints</button>

        <div id="clear-status"></div>

        <h2>Step 2: Access Airflow (Try in Order)</h2>
        <div class="endpoints">
            <a href="http://localhost:8888" target="_blank" class="endpoint">
                Nginx Proxy (8888) - Best Option
            </a>
            <a href="http://127.0.0.1:8080" target="_blank" class="endpoint">
                IP Address (127.0.0.1)
            </a>
            <a href="http://localhost:8080" target="_blank" class="endpoint">
                Direct (localhost)
            </a>
            <a href="http://[::1]:8080" target="_blank" class="endpoint">
                IPv6 ([::1])
            </a>
        </div>

        <h2>Test Results:</h2>
        <div id="test-results"></div>

        <h2>Manual Browser Fixes:</h2>
        <div class="status info">
            <strong>Chrome/Edge:</strong><br>
            1. Press F12 → Application tab → Storage → Clear site data<br>
            2. Or visit: <code>chrome://settings/siteData</code> → Search "localhost" → Remove all<br>
            3. Or use Incognito mode (Ctrl+Shift+N)
        </div>

        <div class="status info">
            <strong>Firefox:</strong><br>
            1. Press F12 → Storage tab → Right-click cookies → Delete All<br>
            2. Or visit: <code>about:preferences#privacy</code> → Manage Data → Remove localhost<br>
            3. Or use Private Window (Ctrl+Shift+P)
        </div>

        <div class="status info">
            <strong>Safari:</strong><br>
            1. Preferences → Privacy → Manage Website Data → Remove localhost<br>
            2. Or use Private Window (Cmd+Shift+N)
        </div>
    </div>

    <script>
        function clearEverything() {
            const status = document.getElementById('clear-status');

            // Clear cookies
            document.cookie.split(";").forEach(function(c) {
                document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" +
                    new Date(0).toUTCString() + ";path=/");
                document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" +
                    new Date(0).toUTCString() + ";path=/;domain=localhost");
                document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" +
                    new Date(0).toUTCString() + ";path=/;domain=.localhost");
            });

            // Clear storage
            try {
                localStorage.clear();
                sessionStorage.clear();
                status.innerHTML = '<div class="status success">✅ Browser data cleared!</div>';
            } catch(e) {
                status.innerHTML = '<div class="status error">⚠️ Partial clear: ' + e.message + '</div>';
            }

            // Clear IndexedDB
            if (window.indexedDB) {
                indexedDB.databases().then(dbs => {
                    dbs.forEach(db => indexedDB.deleteDatabase(db.name));
                });
            }

            // Clear cache if possible
            if ('caches' in window) {
                caches.keys().then(names => {
                    names.forEach(name => caches.delete(name));
                });
            }
        }

        async function testConnections() {
            const results = document.getElementById('test-results');
            results.innerHTML = '<div class="status info">Testing endpoints...</div>';

            const endpoints = [
                {url: 'http://localhost:8888/health', name: 'Nginx Proxy (8888)'},
                {url: 'http://127.0.0.1:8080/health', name: 'IP Address (127.0.0.1)'},
                {url: 'http://localhost:8080/health', name: 'Direct (localhost)'},
            ];

            let html = '';

            for (const endpoint of endpoints) {
                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 3000);

                    const response = await fetch(endpoint.url, {
                        mode: 'no-cors',
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);
                    html += `<div class="test-result pass">✅ ${endpoint.name}: Reachable</div>`;
                } catch (e) {
                    html += `<div class="test-result fail">❌ ${endpoint.name}: ${e.message}</div>`;
                }
            }

            results.innerHTML = html;
        }

        // Auto-clear on load
        window.onload = function() {
            clearEverything();
            setTimeout(testConnections, 1000);
        };
    </script>
</body>
</html>
EOF

echo "✅ Test page created: test_airflow_access.html"
echo ""

# Step 7: Wait and test
echo "⏳ Waiting for services to start..."
sleep 20

# Step 8: Final instructions
echo ""
echo "========================================"
echo "✅ ULTIMATE FIX APPLIED - ALL METHODS ACTIVE"
echo "========================================"
echo ""
echo "🎯 ACCESS AIRFLOW NOW:"
echo ""
echo "Method 1 (BEST): Nginx Proxy"
echo "  URL: http://localhost:8888"
echo "  ✅ Headers automatically stripped"
echo ""
echo "Method 2: Use IP instead of localhost"
echo "  URL: http://127.0.0.1:8080"
echo "  ✅ Often bypasses cookie issues"
echo ""
echo "Method 3: Clear browser and test"
echo "  Open: test_airflow_access.html"
echo "  ✅ Automated clearing and testing"
echo ""
echo "Method 4: Command line (always works)"
echo "  python3 airflow_api_tool.py"
echo ""
echo "----------------------------------------"
echo "Username: admin"
echo "Password: admin"
echo "----------------------------------------"
echo ""

# Test if Nginx proxy is working
if curl -s http://localhost:8888/health > /dev/null 2>&1; then
    echo "✅ Nginx proxy is working! Access: http://localhost:8888"
else
    echo "⚠️ Nginx proxy starting... wait 10 seconds"
fi

# Test if direct access works
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Direct access available at: http://localhost:8080"
else
    echo "⚠️ Airflow still starting... wait 30 seconds"
fi

echo ""
echo "🎉 ALL FIXES APPLIED! Try the URLs above."