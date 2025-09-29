#!/bin/bash

# Airbyte Setup Script
# Sets up Airbyte with Docker and configures connectors

echo "🚀 Setting up Airbyte for MLOps Platform"
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create directory for Airbyte
AIRBYTE_DIR="$HOME/airbyte"
mkdir -p $AIRBYTE_DIR
cd $AIRBYTE_DIR

# Download Airbyte
echo "📥 Downloading Airbyte..."
if [ ! -f "docker-compose.yaml" ]; then
    curl -sOO https://raw.githubusercontent.com/airbytehq/airbyte/master/{.env,docker-compose.yaml}

    # Modify .env for our setup
    echo "Configuring Airbyte..."
    sed -i.bak 's/BASIC_AUTH_USERNAME=.*/BASIC_AUTH_USERNAME=admin/' .env
    sed -i.bak 's/BASIC_AUTH_PASSWORD=.*/BASIC_AUTH_PASSWORD=mlops2024/' .env
    sed -i.bak 's/WEBAPP_URL=.*/WEBAPP_URL=http:\/\/localhost:8000/' .env
fi

# Start Airbyte
echo "🐳 Starting Airbyte containers..."
docker-compose up -d

# Wait for Airbyte to be ready
echo "⏳ Waiting for Airbyte to start (this may take a few minutes)..."
sleep 30

# Check if Airbyte is running
if curl -s http://localhost:8000/api/v1/health | grep -q "available"; then
    echo "✅ Airbyte is running!"
    echo "📊 Airbyte UI: http://localhost:8000"
    echo "   Username: admin"
    echo "   Password: mlops2024"
else
    echo "⚠️ Airbyte may still be starting. Check http://localhost:8000 in a few minutes."
fi

# Show running containers
echo ""
echo "🐳 Airbyte containers:"
docker-compose ps

echo ""
echo "✅ Airbyte setup complete!"
echo ""
echo "To stop Airbyte: cd $AIRBYTE_DIR && docker-compose down"
echo "To view logs: cd $AIRBYTE_DIR && docker-compose logs -f"