#!/bin/bash
# EC2 Deployment Script for Email Classification System
# MLOps Lab 2 - St. Thomas University

echo "================================================"
echo "Email Classification System - EC2 Deployment"
echo "================================================"

# Configuration
EC2_USER="ubuntu"
EC2_IP="YOUR_EC2_PUBLIC_IP"  # Update with actual EC2 IP
KEY_PATH="~/.ssh/your-key.pem"  # Update with actual key path

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if EC2_IP is set
if [ "$EC2_IP" = "YOUR_EC2_PUBLIC_IP" ]; then
    print_error "Please update EC2_IP in this script with your actual EC2 public IP"
    exit 1
fi

# Step 1: Package the application
print_status "Packaging application..."
tar -czf email_classification.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='.git' \
    --exclude='*.tar.gz' \
    .

# Step 2: Copy to EC2
print_status "Copying application to EC2 instance..."
scp -i $KEY_PATH email_classification.tar.gz $EC2_USER@$EC2_IP:~/

# Step 3: SSH and deploy
print_status "Connecting to EC2 and deploying..."
ssh -i $KEY_PATH $EC2_USER@$EC2_IP << 'ENDSSH'
    # Colors for remote output
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'

    echo -e "${GREEN}[EC2]${NC} Starting deployment on EC2..."

    # Update system
    echo -e "${GREEN}[EC2]${NC} Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y

    # Install Python and dependencies
    echo -e "${GREEN}[EC2]${NC} Installing Python 3.11..."
    sudo apt-get install -y python3.11 python3.11-venv python3-pip

    # Install Neo4j client dependencies
    echo -e "${GREEN}[EC2]${NC} Installing Neo4j dependencies..."
    sudo apt-get install -y libssl-dev

    # Extract application
    echo -e "${GREEN}[EC2]${NC} Extracting application..."
    rm -rf email_classification
    mkdir email_classification
    tar -xzf email_classification.tar.gz -C email_classification
    cd email_classification

    # Create virtual environment
    echo -e "${GREEN}[EC2]${NC} Creating virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate

    # Install Python dependencies
    echo -e "${GREEN}[EC2]${NC} Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt

    # Install additional dependencies for Neo4j
    pip install neo4j python-dotenv

    # Create systemd service
    echo -e "${GREEN}[EC2]${NC} Creating systemd service..."
    sudo tee /etc/systemd/system/email-classification.service > /dev/null << EOF
[Unit]
Description=Email Classification API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/email_classification
Environment="PATH=/home/ubuntu/email_classification/venv/bin"
ExecStart=/home/ubuntu/email_classification/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and start service
    echo -e "${GREEN}[EC2]${NC} Starting email classification service..."
    sudo systemctl daemon-reload
    sudo systemctl enable email-classification
    sudo systemctl restart email-classification

    # Check service status
    echo -e "${GREEN}[EC2]${NC} Checking service status..."
    sudo systemctl status email-classification --no-pager

    # Configure firewall
    echo -e "${GREEN}[EC2]${NC} Configuring firewall..."
    sudo ufw allow 8000/tcp
    sudo ufw allow 22/tcp
    sudo ufw --force enable

    # Create data directories
    echo -e "${GREEN}[EC2]${NC} Creating data directories..."
    mkdir -p data reports

    # Show deployment info
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Access the API at:"
    echo "  http://$EC2_IP:8000"
    echo "  http://$EC2_IP:8000/docs (Swagger UI)"
    echo ""
    echo "Check logs with:"
    echo "  sudo journalctl -u email-classification -f"
    echo ""
    echo "Restart service with:"
    echo "  sudo systemctl restart email-classification"
ENDSSH

print_status "Deployment complete!"
print_status "Testing API connection..."

# Test the API
sleep 5
if curl -s -o /dev/null -w "%{http_code}" http://$EC2_IP:8000/health | grep -q "200"; then
    print_status "API is running successfully!"
    echo ""
    echo "================================================"
    echo "Access your API at:"
    echo "  Main API: http://$EC2_IP:8000"
    echo "  Swagger Docs: http://$EC2_IP:8000/docs"
    echo "  ReDoc: http://$EC2_IP:8000/redoc"
    echo "================================================"
else
    print_warning "API may still be starting. Please check in a few moments."
fi

# Clean up local tar file
rm -f email_classification.tar.gz