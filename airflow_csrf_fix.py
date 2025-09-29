#!/usr/bin/env python3
"""
Airflow CSRF Token Fix
Creates a session-aware proxy that handles CSRF tokens automatically
"""

import requests
from flask import Flask, request, Response, redirect, render_template_string
import re
import time

app = Flask(__name__)

# Session storage
session_cookies = {}
csrf_tokens = {}

AIRFLOW_URL = "http://localhost:8080"

def get_csrf_token():
    """Get CSRF token from Airflow login page"""
    try:
        response = requests.get(f"{AIRFLOW_URL}/login")
        # Look for CSRF token in the HTML
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        if csrf_match:
            return csrf_match.group(1)
    except:
        pass
    return None

def auto_login():
    """Automatically log in to Airflow and get session"""
    try:
        # Get login page and CSRF token
        login_response = requests.get(f"{AIRFLOW_URL}/login")
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_response.text)

        if not csrf_match:
            return None, None

        csrf_token = csrf_match.group(1)

        # Login with credentials
        login_data = {
            'username': 'admin',
            'password': 'admin',
            'csrf_token': csrf_token
        }

        # Use session cookies from login page
        cookies = login_response.cookies

        auth_response = requests.post(
            f"{AIRFLOW_URL}/login",
            data=login_data,
            cookies=cookies,
            allow_redirects=False
        )

        # Get session cookies from successful login
        if auth_response.status_code in [302, 200]:
            session_cookies.update(cookies)
            session_cookies.update(auth_response.cookies)
            return dict(session_cookies), csrf_token

    except Exception as e:
        print(f"Auto-login failed: {e}")

    return None, None

@app.route('/')
def index():
    """Landing page with auto-login"""
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Airflow CSRF Fix - Auto Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 100px auto;
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
        }
        .card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #28a745; }
        .btn {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px;
            font-size: 16px;
        }
        .btn:hover { background: #0056b3; }
        .success { background: #d4edda; padding: 15px; border-radius: 5px; color: #155724; margin: 20px 0; }
    </style>
    <script>
        function autoLogin() {
            document.getElementById('status').innerHTML = '🔄 Logging in automatically...';
            fetch('/auto-login')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('status').innerHTML = '✅ Logged in successfully! Redirecting...';
                    setTimeout(() => {
                        window.location.href = '/airflow/';
                    }, 2000);
                } else {
                    document.getElementById('status').innerHTML = '❌ Auto-login failed. Try manual access.';
                }
            });
        }

        // Auto-login on page load
        window.onload = autoLogin;
    </script>
</head>
<body>
    <div class="card">
        <h1>🔐 Airflow Access - CSRF Fixed</h1>

        <div id="status" class="success">
            🚀 Starting automatic login...
        </div>

        <p>This proxy automatically handles CSRF tokens and authentication.</p>

        <a href="/airflow/" class="btn">🌐 Access Airflow Dashboard</a>
        <a href="/dags" class="btn">📋 View DAGs Directly</a>

        <div style="margin-top: 30px; font-size: 14px; color: #666;">
            <strong>How it works:</strong><br>
            1. Automatically gets CSRF token from Airflow<br>
            2. Logs in with admin/admin credentials<br>
            3. Maintains session cookies<br>
            4. Forwards all requests with proper authentication
        </div>
    </div>
</body>
</html>
    """)

@app.route('/auto-login')
def auto_login_endpoint():
    """API endpoint for auto-login"""
    cookies, csrf = auto_login()
    if cookies:
        return {"success": True, "message": "Logged in successfully"}
    else:
        return {"success": False, "message": "Login failed"}

@app.route('/airflow/')
@app.route('/airflow/<path:path>')
def airflow_proxy(path=''):
    """Authenticated proxy to Airflow"""

    # Ensure we have valid session
    if not session_cookies:
        auto_login()

    # Build target URL
    url = f"{AIRFLOW_URL}/{path}"
    if request.query_string:
        url += f"?{request.query_string.decode()}"

    # Prepare headers
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'User-Agent': 'Mozilla/5.0 (compatible; AirflowProxy/1.0)',
        'Referer': f"{AIRFLOW_URL}/"
    }

    try:
        # Make authenticated request
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            cookies=session_cookies,
            allow_redirects=False,
            timeout=30
        )

        # Update session cookies if new ones are provided
        if response.cookies:
            session_cookies.update(response.cookies)

        # Handle redirects to login page (session expired)
        if response.status_code == 302 and 'login' in response.headers.get('Location', ''):
            # Re-authenticate and retry
            auto_login()
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                data=request.get_data(),
                cookies=session_cookies,
                allow_redirects=False,
                timeout=30
            )

        # Build response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(name, value) for (name, value) in response.headers.items()
                           if name.lower() not in excluded_headers]

        return Response(response.content, response.status_code, response_headers)

    except Exception as e:
        return f"""
        <h1>Connection Error</h1>
        <p>Could not connect to Airflow: {e}</p>
        <p><a href="/">← Back to Home</a></p>
        <p><strong>Try:</strong></p>
        <ul>
            <li>Check if Airflow is running: <code>docker ps | grep airflow</code></li>
            <li>Access directly: <a href="http://localhost:8080">http://localhost:8080</a></li>
            <li>Use command line: <code>python3 airflow_api_tool.py</code></li>
        </ul>
        """, 502

@app.route('/dags')
def dags_direct():
    """Direct access to DAGs page"""
    return airflow_proxy('dags')

@app.route('/health')
def health():
    """Health check"""
    try:
        response = requests.get(f"{AIRFLOW_URL}/health", timeout=5)
        return f"✅ Airflow: {response.status_code}, Proxy: OK", 200
    except:
        return "❌ Cannot reach Airflow", 503

if __name__ == '__main__':
    print("🚀 AIRFLOW CSRF TOKEN FIX")
    print("=" * 50)
    print("")
    print("This proxy automatically handles:")
    print("✅ CSRF token extraction")
    print("✅ Automatic login (admin/admin)")
    print("✅ Session management")
    print("✅ Request forwarding")
    print("")
    print("🌐 Access at: http://localhost:7777")
    print("")
    print("=" * 50)

    app.run(host='0.0.0.0', port=7777, debug=False)