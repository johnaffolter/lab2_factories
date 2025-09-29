#!/bin/bash

echo "🔧 FIXING NGINX CONNECTION ISSUES"
echo "=================================="
echo ""

# Stop any existing nginx containers
echo "🛑 Cleaning up existing nginx containers..."
docker stop airflow-nginx airflow-nginx-proxy 2>/dev/null
docker rm airflow-nginx airflow-nginx-proxy 2>/dev/null

# Check if Airflow is running
echo "🔍 Checking Airflow status..."
if docker ps | grep -q airflow-standalone; then
    echo "✅ Airflow container is running"
else
    echo "❌ Airflow container not found - starting it..."
    # Add restart logic here if needed
fi

# Test direct Airflow connection
echo "🔍 Testing direct Airflow connection..."
if curl -s -I http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Direct Airflow connection working"
else
    echo "❌ Direct Airflow connection failed"
fi

# Create working nginx config
echo "📝 Creating nginx configuration..."
cat > nginx_simple.conf << 'EOF'
server {
    listen 8888;
    server_name localhost;

    # Basic proxy settings
    location / {
        proxy_pass http://host.docker.internal:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for real-time response
        proxy_buffering off;
        proxy_cache off;

        # Connection settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        return 200 "nginx proxy healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Start new nginx proxy
echo "🚀 Starting nginx proxy..."
docker run -d \
    --name airflow-nginx-proxy \
    -p 8888:8888 \
    -v ${PWD}/nginx_simple.conf:/etc/nginx/conf.d/default.conf:ro \
    --add-host=host.docker.internal:host-gateway \
    nginx:alpine

sleep 5

# Test nginx proxy
echo "🧪 Testing nginx proxy..."
if curl -s -I http://localhost:8888/health > /dev/null 2>&1; then
    echo "✅ Nginx proxy health check passed"
else
    echo "❌ Nginx proxy health check failed"
fi

# Test full proxy connection
echo "🧪 Testing full proxy connection to Airflow..."
if curl -s -I http://localhost:8888 > /dev/null 2>&1; then
    echo "✅ Nginx → Airflow proxy working!"
else
    echo "❌ Nginx → Airflow proxy failed"
    echo "📋 Checking nginx logs..."
    docker logs --tail 10 airflow-nginx-proxy
fi

echo ""
echo "=================================="
echo "✅ CONNECTION FIX COMPLETE"
echo "=================================="
echo ""
echo "🌐 Access Points:"
echo "   Primary: http://localhost:8888"
echo "   Direct:  http://localhost:8080"
echo "   Health:  http://localhost:8888/health"
echo ""
echo "🔧 If still having issues:"
echo "   1. Try: curl http://localhost:8888/health"
echo "   2. Check: docker logs airflow-nginx-proxy"
echo "   3. Direct: curl http://localhost:8080/health"
echo ""

# Show container status
echo "📊 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|airflow|nginx)"