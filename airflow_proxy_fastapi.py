#!/usr/bin/env python3
"""
FastAPI-based Airflow Proxy
High-performance proxy that strips large headers
Fixes 431 Request Header Fields Too Large error
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import httpx
import asyncio
from typing import Optional, Dict, Any
import re
import json
from datetime import datetime

app = FastAPI(title="Airflow Header-Fix Proxy")

# Configuration
AIRFLOW_BASE_URL = "http://localhost:8080"
MAX_HEADER_SIZE = 8192  # 8KB max per header
MAX_COOKIE_SIZE = 4096  # 4KB max per cookie
MAX_COOKIES = 10  # Maximum number of cookies to forward

# HTTP client with custom limits
client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=10)
)

def filter_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Filter and clean headers to prevent 431 errors"""
    filtered = {}

    for key, value in headers.items():
        key_lower = key.lower()

        # Skip connection-specific headers
        if key_lower in ['host', 'content-length', 'connection', 'transfer-encoding']:
            continue

        # Handle cookies specially
        if key_lower == 'cookie':
            cookies = value.split(';')
            filtered_cookies = []

            for cookie in cookies[:MAX_COOKIES]:  # Limit number of cookies
                cookie = cookie.strip()
                if len(cookie) < MAX_COOKIE_SIZE:
                    # Remove tracking and analytics cookies
                    if not any(track in cookie.lower() for track in ['_ga', '_gid', 'utm_', 'fbclid']):
                        filtered_cookies.append(cookie)

            if filtered_cookies:
                filtered['Cookie'] = '; '.join(filtered_cookies)

        # Skip oversized headers
        elif len(value) < MAX_HEADER_SIZE:
            filtered[key] = value

    return filtered

@app.get("/")
async def root():
    """Proxy root page"""
    return await proxy_request("")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_request(path: str, request: Request):
    """Proxy all requests to Airflow with header filtering"""

    # Build target URL
    url = f"{AIRFLOW_BASE_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Filter headers
    headers = filter_headers(dict(request.headers))

    # Get request body if present
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()

    try:
        # Make request to Airflow
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            follow_redirects=False
        )

        # Filter response headers
        response_headers = {}
        for key, value in response.headers.items():
            if key.lower() not in ['content-encoding', 'content-length', 'transfer-encoding']:
                response_headers[key] = value

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")

@app.get("/proxy/health")
async def health_check():
    """Health check endpoint"""
    try:
        response = await client.get(f"{AIRFLOW_BASE_URL}/health")
        airflow_healthy = response.status_code == 200
    except:
        airflow_healthy = False

    return {
        "proxy": "healthy",
        "airflow": "healthy" if airflow_healthy else "unreachable",
        "timestamp": datetime.now().isoformat(),
        "max_header_size": MAX_HEADER_SIZE,
        "max_cookie_size": MAX_COOKIE_SIZE
    }

@app.get("/proxy/clear-cookies", response_class=HTMLResponse)
async def clear_cookies_page():
    """Page to clear browser cookies"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clear Airflow Cookies</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #f3f4f6;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            h1 { color: #111827; }
            button {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                margin: 5px;
            }
            button:hover { background: #2563eb; }
            .status {
                padding: 12px;
                border-radius: 6px;
                margin: 15px 0;
                display: none;
            }
            .success { background: #d1fae5; color: #065f46; display: block; }
            .error { background: #fee2e2; color: #991b1b; display: block; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍪 Clear Airflow Cookies</h1>
            <p>Click the button below to clear cookies and fix header issues:</p>

            <button onclick="clearAll()">Clear All Browser Data</button>
            <button onclick="window.location.href='/'">Go to Airflow</button>

            <div id="status" class="status"></div>

            <script>
                function clearAll() {
                    // Clear cookies
                    document.cookie.split(";").forEach(function(c) {
                        document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" +
                            new Date(0).toUTCString() + ";path=/");
                    });

                    // Clear storage
                    localStorage.clear();
                    sessionStorage.clear();

                    // Show status
                    const status = document.getElementById('status');
                    status.className = 'status success';
                    status.innerHTML = '✅ Browser data cleared! You can now access Airflow.';

                    // Redirect after 2 seconds
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 2000);
                }

                // Auto-clear on load
                window.onload = clearAll;
            </script>
        </div>
    </body>
    </html>
    """
    return html

@app.get("/proxy/stats")
async def proxy_stats():
    """Get proxy statistics"""
    return {
        "proxy_type": "FastAPI",
        "airflow_url": AIRFLOW_BASE_URL,
        "limits": {
            "max_header_size": f"{MAX_HEADER_SIZE} bytes",
            "max_cookie_size": f"{MAX_COOKIE_SIZE} bytes",
            "max_cookies": MAX_COOKIES
        },
        "endpoints": [
            "/proxy/health - Health check",
            "/proxy/clear-cookies - Clear browser cookies",
            "/proxy/stats - This page",
            "/* - All other paths proxy to Airflow"
        ]
    }

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 FASTAPI AIRFLOW PROXY")
    print("=" * 60)
    print("\nThis proxy strips large headers to fix 431 errors")
    print("\nStarting on: http://localhost:9999")
    print("Proxying to: http://localhost:8080")
    print("\nFeatures:")
    print("  • Automatic header size limiting")
    print("  • Cookie filtering and reduction")
    print("  • Tracking cookie removal")
    print("  • Browser data clearing page")
    print("\n" + "=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=9999)