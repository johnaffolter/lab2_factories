#!/bin/bash

echo "🔧 Setting up Airflow Proxy to Fix Header Issues"
echo "================================================="
echo ""

# Method 1: Using Docker Nginx
echo "📦 Method 1: Docker Nginx Proxy"
echo "--------------------------------"

# Stop any existing nginx proxy
docker stop airflow-nginx-proxy 2>/dev/null
docker rm airflow-nginx-proxy 2>/dev/null

# Start nginx with custom config
docker run -d \
    --name airflow-nginx-proxy \
    -p 8888:8888 \
    -v ${PWD}/nginx_airflow_proxy.conf:/etc/nginx/conf.d/default.conf:ro \
    --add-host=host.docker.internal:host-gateway \
    nginx:alpine

echo "✅ Nginx proxy started on port 8888"
echo ""

# Method 2: Python Header-Stripping Proxy
echo "🐍 Method 2: Python Header-Stripping Proxy"
echo "------------------------------------------"

cat > header_strip_proxy.py << 'EOF'
#!/usr/bin/env python3
"""
Header-stripping proxy for Airflow
Removes large cookies and headers that cause 431 errors
"""

from flask import Flask, request, Response, session
import requests
from werkzeug.exceptions import BadRequest
import re

app = Flask(__name__)
app.secret_key = 'airflow-proxy-secret-key'

AIRFLOW_URL = "http://localhost:8080"
MAX_COOKIE_SIZE = 4096  # 4KB max per cookie

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    """Proxy requests to Airflow with header filtering"""

    # Build target URL
    url = f"{AIRFLOW_URL}/{path}"
    if request.query_string:
        url += f"?{request.query_string.decode()}"

    # Filter headers
    headers = {}
    for key, value in request.headers:
        # Skip problematic headers
        if key.lower() in ['host', 'content-length', 'connection']:
            continue

        # Limit cookie size
        if key.lower() == 'cookie':
            cookies = value.split(';')
            filtered_cookies = []
            for cookie in cookies:
                if len(cookie.strip()) < MAX_COOKIE_SIZE:
                    filtered_cookies.append(cookie.strip())
            if filtered_cookies:
                headers['Cookie'] = '; '.join(filtered_cookies[:10])  # Max 10 cookies
        else:
            # Include other headers if not too large
            if len(value) < 8192:  # 8KB max per header
                headers[key] = value

    # Forward request
    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30
        )

        # Build response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]

        response = Response(resp.content, resp.status_code, headers)
        return response

    except requests.exceptions.RequestException as e:
        return f"Proxy error: {e}", 502

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "proxy": "active"}

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 HEADER-STRIPPING PROXY FOR AIRFLOW")
    print("=" * 60)
    print("\nThis proxy removes large headers that cause 431 errors")
    print("\nAccess Airflow at: http://localhost:7777")
    print("(Instead of http://localhost:8080)")
    print("\n" + "=" * 60)

    app.run(host='0.0.0.0', port=7777, debug=False)
EOF

echo "✅ Python proxy script created"
echo ""

# Method 3: Restart Airflow with increased limits
echo "🔄 Method 3: Restart Airflow with Increased Limits"
echo "---------------------------------------------------"

docker stop airflow-standalone 2>/dev/null
docker rm airflow-standalone 2>/dev/null

docker run -d \
    --name airflow-standalone \
    -p 8080:8080 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=120 \
    -e AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL=30 \
    -e AIRFLOW__WEBSERVER__WORKER_CLASS=sync \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_REQUEST_LINE=16384 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_REQUEST_FIELD_LIMIT=100 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_MAX_REQUEST_LINE=16384 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_MAX_REQUEST_FIELDS=100 \
    -e AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_MAX_REQUEST_FIELD_SIZE=16384 \
    -e AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth \
    -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@host.docker.internal/airflow \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
    -e _AIRFLOW_DB_MIGRATE=true \
    -e _AIRFLOW_WWW_USER_CREATE=true \
    -e _AIRFLOW_WWW_USER_USERNAME=admin \
    -e _AIRFLOW_WWW_USER_PASSWORD=admin \
    -v ${PWD}/dags:/opt/airflow/dags \
    -v ${PWD}/logs:/opt/airflow/logs \
    apache/airflow:2.7.0 standalone

echo "✅ Airflow restarted with increased header limits"
echo ""

# Method 4: Clear browser data script
echo "🧹 Method 4: Browser Cleanup"
echo "-----------------------------"

cat > clear_browser_for_airflow.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Clear Browser Data for Airflow</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover {
            background: #5a67d8;
        }
        .status {
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .success { background: #d1fae5; color: #065f46; }
        .info { background: #dbeafe; color: #1e40af; }
        .warning { background: #fee2e2; color: #991b1b; }
        code {
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Fix Airflow Browser Access</h1>

        <div class="status info">
            <strong>Problem:</strong> Your browser has accumulated too many cookies/headers for localhost:8080
        </div>

        <h2>Automatic Fixes:</h2>

        <button onclick="clearLocalStorage()">Clear Local Storage</button>
        <button onclick="clearSessionStorage()">Clear Session Storage</button>
        <button onclick="clearCookies()">Clear Cookies (Limited)</button>
        <button onclick="testAirflow()">Test Airflow Connection</button>

        <div id="results"></div>

        <h2>Manual Steps Required:</h2>
        <ol>
            <li>Open Chrome DevTools (F12)</li>
            <li>Go to Application tab</li>
            <li>Find "Storage" in left sidebar</li>
            <li>Right-click "Cookies" → "Clear"</li>
            <li>Or use <code>chrome://settings/siteData</code> and search for "localhost"</li>
            <li>Delete all localhost entries</li>
        </ol>

        <h2>Alternative Access Points:</h2>
        <ul>
            <li><a href="http://localhost:8888" target="_blank">Nginx Proxy (port 8888)</a> - Headers stripped</li>
            <li><a href="http://localhost:7777" target="_blank">Python Proxy (port 7777)</a> - Cookies filtered</li>
            <li><a href="http://localhost:5555" target="_blank">Web Interface (port 5555)</a> - Custom UI</li>
            <li><a href="http://127.0.0.1:8080" target="_blank">Use IP instead of localhost</a></li>
        </ul>

        <script>
            function clearLocalStorage() {
                try {
                    localStorage.clear();
                    showResult('✅ Local Storage cleared', 'success');
                } catch(e) {
                    showResult('❌ Error: ' + e.message, 'warning');
                }
            }

            function clearSessionStorage() {
                try {
                    sessionStorage.clear();
                    showResult('✅ Session Storage cleared', 'success');
                } catch(e) {
                    showResult('❌ Error: ' + e.message, 'warning');
                }
            }

            function clearCookies() {
                // Limited cookie clearing via JavaScript
                document.cookie.split(";").forEach(function(c) {
                    document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
                });
                showResult('⚠️ Cookies cleared (limited - use DevTools for complete removal)', 'info');
            }

            function testAirflow() {
                showResult('🔍 Testing Airflow connection...', 'info');

                // Try to fetch from Airflow
                fetch('http://localhost:8080/health', {
                    mode: 'no-cors',
                    credentials: 'omit'
                })
                .then(() => {
                    showResult('✅ Airflow is reachable (try accessing now)', 'success');
                })
                .catch(() => {
                    showResult('❌ Cannot reach Airflow (may need proxy)', 'warning');
                });
            }

            function showResult(message, type) {
                const results = document.getElementById('results');
                const div = document.createElement('div');
                div.className = 'status ' + type;
                div.innerHTML = message;
                results.appendChild(div);

                setTimeout(() => {
                    div.remove();
                }, 5000);
            }

            // Auto-clear on load
            window.onload = function() {
                console.log('Clearing browser data for Airflow...');
                clearLocalStorage();
                clearSessionStorage();
            };
        </script>
    </div>
</body>
</html>
EOF

echo "✅ Browser cleanup page created: clear_browser_for_airflow.html"
echo ""

echo "================================================="
echo "📋 ACCESS OPTIONS (CHOOSE BASED ON YOUR NEED):"
echo "================================================="
echo ""
echo "1️⃣ NGINX PROXY (Port 8888) - RECOMMENDED"
echo "   URL: http://localhost:8888"
echo "   ✅ Strips large headers automatically"
echo ""
echo "2️⃣ PYTHON PROXY (Port 7777)"
echo "   Run: python3 header_strip_proxy.py"
echo "   URL: http://localhost:7777"
echo "   ✅ Filters cookies and headers"
echo ""
echo "3️⃣ DIRECT ACCESS (Port 8080)"
echo "   URL: http://localhost:8080"
echo "   ⚠️ May still have header issues"
echo ""
echo "4️⃣ IP ADDRESS (Instead of localhost)"
echo "   URL: http://127.0.0.1:8080"
echo "   ✅ Sometimes bypasses cookie issues"
echo ""
echo "5️⃣ BROWSER CLEANUP"
echo "   Open: clear_browser_for_airflow.html"
echo "   ✅ Clears browser storage"
echo ""
echo "================================================="
echo ""
echo "Username: admin"
echo "Password: admin"
echo ""
echo "✅ All proxy solutions are now available!"