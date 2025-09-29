# MLOps Email Classifier - Setup Guide

Complete setup guide for all environments using UV package manager.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development](#local-development)
4. [FastAPI Web Server](#fastapi-web-server)
5. [Docker/Airflow](#dockerairflow)
6. [Neo4j Knowledge Graph](#neo4j-knowledge-graph)
7. [Environment Variables](#environment-variables)

---

## Prerequisites

### Install UV Package Manager
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Verify Installation
```bash
uv --version
```

---

## Environment Setup

### 1. Local Development Environment

Complete environment with all features for local development.

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install all dependencies
uv pip install -e ".[local]"

# Or install from requirements file
uv pip install -r requirements-local.txt
```

**Includes:**
- FastAPI web server
- Neo4j database client
- AWS S3 integration
- OpenAI LLM integration
- ML libraries (numpy, pandas, scikit-learn)
- Development tools (pytest, black, ruff, mypy)

**Usage:**
```bash
# Run FastAPI server
uvicorn app.main:app --reload --port 8000

# Run tests
pytest

# Format code
black .
ruff check .
```

---

### 2. FastAPI API Server Environment

Production API server with database and cloud integrations.

```bash
# Create virtual environment
uv venv api-env

# Activate
source api-env/bin/activate

# Install API dependencies
uv pip install -e ".[api]"

# Or from requirements
uv pip install -r requirements-api.txt
```

**Includes:**
- FastAPI + Uvicorn
- Neo4j client
- AWS S3
- OpenAI
- ML libraries

**Excludes:**
- Development tools
- Airflow

**Usage:**
```bash
# Run production server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs  # Swagger UI
open http://localhost:8000/redoc  # ReDoc
```

---

### 3. Docker/Airflow Environment

Minimal dependencies for Airflow DAGs in Docker containers.

```bash
# In Docker container (Airflow pre-installed)
uv pip install -e ".[docker]"

# Or from requirements
uv pip install -r requirements-docker.txt
```

**Includes:**
- Neo4j client
- AWS S3 (boto3)
- python-dotenv

**Excludes:**
- FastAPI/Uvicorn (not needed in Airflow)
- Development tools
- apache-airflow (pre-installed in container)

**Docker Setup:**
```bash
# Install dependencies in Airflow container
docker exec airflow-standalone pip install neo4j boto3 python-dotenv

# Or copy requirements file
docker cp requirements-docker.txt airflow-standalone:/tmp/
docker exec airflow-standalone pip install -r /tmp/requirements-docker.txt
```

---

## Local Development

### Quick Start
```bash
# 1. Clone and navigate
cd /Users/johnaffolter/lab_2_homework/lab2_factories

# 2. Create environment
uv venv

# 3. Activate
source .venv/bin/activate

# 4. Install dependencies
uv pip install -e ".[local]"

# 5. Copy environment template
cp .env.example .env

# 6. Edit .env with your credentials
nano .env  # or vim, code, etc.

# 7. Run FastAPI server
uvicorn app.main:app --reload

# 8. Run tests
pytest
```

### Development Workflow
```bash
# Install package in editable mode
uv pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=app tests/

# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type check
mypy app/
```

---

## FastAPI Web Server

### Starting the Server

**Development:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Production:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

**Swagger UI (Interactive API Docs):**
```
http://localhost:8000/docs
```

**ReDoc (Alternative API Docs):**
```
http://localhost:8000/redoc
```

**OpenAPI JSON Schema:**
```
http://localhost:8000/openapi.json
```

### Key Endpoints

```python
# Email Classification
POST /api/classify
{
    "subject": "string",
    "body": "string",
    "use_learning": false
}

# Feature Generation
POST /api/features/generate
{
    "subject": "string",
    "body": "string"
}

# Get Available Feature Generators
GET /api/features/generators

# Neo4j Statistics
GET /api/neo4j/statistics

# ML Model Endpoints
GET /api/ml/models
POST /api/ml/predict
```

---

## Docker/Airflow

### Airflow Setup

**1. Start Airflow:**
```bash
docker run -d \
  --name airflow-standalone \
  -p 8080:8080 \
  -v /Users/johnaffolter/lab_2_homework/lab2_factories/dags:/opt/airflow/dags \
  -e LOAD_EX=n \
  apache/airflow:latest-python3.8 standalone
```

**2. Install Dependencies in Container:**
```bash
# Copy .env file to container
docker cp .env airflow-standalone:/opt/airflow/

# Install Python dependencies
docker exec airflow-standalone pip install neo4j boto3 python-dotenv

# Verify installation
docker exec airflow-standalone pip list | grep neo4j
```

**3. Access Airflow UI:**
```
http://localhost:8080
Username: admin
Password: admin
```

**4. Trigger DAG:**
```bash
# Via CLI
docker exec airflow-standalone airflow dags trigger mlops_neo4j_simple

# Via UI - Go to http://localhost:8080 and click "Trigger DAG"
```

### Available DAGs

1. **mlops_neo4j_simple** - Simple pipeline with Neo4j integration
   - Generates sample emails
   - Classifies using rule-based model
   - Stores results in Neo4j
   - Generates summary

2. **mlops_s3_simple** - ML pipeline with S3 storage
   - Generates training data
   - Trains multiple models
   - Uploads to S3
   - Validates results

3. **s3_lab3_dag** - Lab 3 S3 operations
   - Checks AWS credentials
   - Uploads files to S3
   - Downloads from S3
   - Verifies operations

---

## Neo4j Knowledge Graph

### Connection Setup

**Environment Variables (.env):**
```bash
NEO4J_URI=neo4j+s://e0753253.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=OlKIfoBwlcR3Vm3Jwtu1EhFu0WutvdVibDHY7hvkn_M
NEO4J_DATABASE=neo4j
```

### Using the Knowledge Graph

**Python:**
```python
from app.services.mlops_neo4j_integration import get_knowledge_graph

# Get singleton instance
kg = get_knowledge_graph()

# Store email classification
email_data = {
    'subject': 'Meeting tomorrow',
    'body': 'Team standup at 10am',
    'sender': 'boss@company.com'
}

classification_result = {
    'predicted_topic': 'work',
    'confidence_scores': {'work': 0.95, 'personal': 0.05},
    'features': {...},
    'model_type': 'EmailClassifierModel'
}

email_id = kg.store_email_with_classification(email_data, classification_result)

# Store ground truth
kg.store_ground_truth(email_id, 'work', annotator='human')

# Get training examples
training_data = kg.generate_training_examples(count=100)

# Get system overview
overview = kg.get_mlops_system_overview()
```

### Graph Schema

**Nodes:**
- `:Email` - Email messages
- `:Topic` - Classification topics
- `:MLModel` - Machine learning models
- `:FeatureGenerator` - Feature extraction components
- `:TrainingFlow` - MLOps pipelines
- `:Dataset` - Training datasets
- `:DesignPattern` - Software patterns used

**Relationships:**
- `(:Email)-[:CLASSIFIED_AS]->(:Topic)`
- `(:Email)-[:HAS_GROUND_TRUTH]->(:Topic)`
- `(:MLModel)-[:PREDICTED]->(:Email)`
- `(:FeatureGenerator)-[:EXTRACTED_FROM]->(:Email)`
- `(:TrainingFlow)-[:PRODUCED]->(:MLModel)`

---

## Environment Variables

### Complete .env File

```bash
# Neo4j Aura Configuration (Free instance - safe to use)
NEO4J_URI=neo4j+s://e0753253.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=OlKIfoBwlcR3Vm3Jwtu1EhFu0WutvdVibDHY7hvkn_M
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=e0753253
AURA_INSTANCENAME=Instance01

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-west-2

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development

# EC2 Configuration (update when deploying)
EC2_PUBLIC_IP=
EC2_INSTANCE_ID=

# Airflow Configuration
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
```

### Loading Environment Variables

**In Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

neo4j_uri = os.getenv('NEO4J_URI')
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
openai_key = os.getenv('OPENAI_API_KEY')
```

**In Docker:**
```bash
# Pass .env file to container
docker run --env-file .env your_image

# Or individual variables
docker run -e NEO4J_URI=$NEO4J_URI your_image
```

---

## UV Command Reference

### Installation Commands

```bash
# Install from pyproject.toml (local development)
uv pip install -e ".[local]"

# Install specific environment
uv pip install -e ".[api]"      # API server
uv pip install -e ".[docker]"   # Docker/Airflow
uv pip install -e ".[dev]"      # Development only

# Install from requirements file
uv pip install -r requirements-local.txt
uv pip install -r requirements-api.txt
uv pip install -r requirements-docker.txt

# Install single dependency
uv pip install neo4j
uv pip install fastapi uvicorn

# Upgrade all packages
uv pip install --upgrade -r requirements-local.txt

# Generate lock file
uv pip compile pyproject.toml -o requirements-lock.txt
```

### Virtual Environment Management

```bash
# Create new virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.10

# Create with custom name
uv venv my-env

# Activate
source .venv/bin/activate

# Deactivate
deactivate

# Remove environment
rm -rf .venv
```

### Package Management

```bash
# List installed packages
uv pip list

# Show package info
uv pip show neo4j

# Freeze current environment
uv pip freeze > requirements-freeze.txt

# Uninstall package
uv pip uninstall fastapi

# Check for outdated packages
uv pip list --outdated
```

---

## Troubleshooting

### Common Issues

**1. UV not found:**
```bash
# Reinstall UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**2. Python version mismatch:**
```bash
# Create environment with specific version
uv venv --python 3.10

# Or use pyenv
pyenv install 3.10.0
pyenv local 3.10.0
uv venv
```

**3. Neo4j connection errors:**
```bash
# Verify credentials
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USERNAME', '$NEO4J_PASSWORD')); print('Connected!')"
```

**4. AWS credentials not found:**
```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Or use AWS CLI
aws configure
```

**5. Airflow DAG not showing:**
```bash
# Check for import errors
docker exec airflow-standalone airflow dags list-import-errors

# Restart Airflow
docker restart airflow-standalone
```

---

## Next Steps

1. **Set up local development:** Follow [Local Development](#local-development)
2. **Start FastAPI server:** See [FastAPI Web Server](#fastapi-web-server)
3. **Configure Neo4j:** Set up [Neo4j Knowledge Graph](#neo4j-knowledge-graph)
4. **Run Airflow DAGs:** Deploy [Docker/Airflow](#dockerairflow)
5. **Review type mappings:** Check `NEO4J_TYPE_MAPPINGS.md`

---

## Resources

- **UV Documentation:** https://github.com/astral-sh/uv
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Neo4j Python Driver:** https://neo4j.com/docs/api/python-driver/
- **Apache Airflow:** https://airflow.apache.org/docs/
- **AWS Boto3:** https://boto3.amazonaws.com/v1/documentation/api/latest/index.html