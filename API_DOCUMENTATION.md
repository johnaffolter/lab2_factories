# Composable MLOps Platform - API Documentation

**Version:** 2.0.0
**Base URL:** `http://localhost:8000`
**Author:** John Affolter

---

## Overview

The Composable MLOps Platform provides REST APIs for managing composable ML components, building pipelines, and executing email classification workflows.

### Key Features

- 🔍 **Component Discovery** - Browse and search composable components
- 🔗 **Pipeline Management** - Create, save, and execute pipelines
- 📊 **Real-time Monitoring** - WebSocket-based system status
- 🎯 **Email Classification** - Execute pipelines on email data
- 📦 **Component Marketplace** - Discover community components

---

## Authentication

Currently no authentication required. In production, implement JWT or API keys.

---

## Endpoints

### Health Check

**GET** `/health`

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "ML Server"
}
```

---

## Component Management

### List All Components

**GET** `/api/composable/components`

Get all registered composable components.

**Query Parameters:**
- `type_filter` (optional) - Filter by component type

**Response:**
```json
{
  "total": 5,
  "components": [
    {
      "name": "Spam Detector",
      "version": "1.0.0",
      "type": "feature_generator",
      "description": "Detects spam keywords and patterns in email content",
      "author": "John Affolter",
      "tags": ["spam", "detection", "nlp", "keywords"],
      "icon": "🚫",
      "color": "#FF6B6B",
      "inputs": [
        {"name": "email", "type": "Email", "required": true}
      ],
      "outputs": [
        {"name": "spam_score", "type": "float"},
        {"name": "spam_keywords", "type": "List[str]"},
        {"name": "has_spam", "type": "bool"}
      ],
      "config_schema": {
        "type": "object",
        "properties": {
          "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "default": ["free", "winner", "urgent"]
          },
          "threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.3
          }
        }
      }
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/api/composable/components
```

---

### Get Component Details

**GET** `/api/composable/components/{component_name}`

Get detailed information about a specific component.

**Response:**
```json
{
  "name": "Spam Detector",
  "version": "1.0.0",
  "type": "feature_generator",
  "description": "Detects spam keywords and patterns in email content",
  "author": "John Affolter",
  "tags": ["spam", "detection", "nlp", "keywords"],
  "icon": "🚫",
  "color": "#FF6B6B",
  "inputs": [...],
  "outputs": [...],
  "config_schema": {...},
  "default_config": {}
}
```

**Example:**
```bash
curl http://localhost:8000/api/composable/components/Spam%20Detector
```

---

### Search Components

**POST** `/api/composable/components/search`

Search components by name, description, or tags.

**Request Body:**
```json
{
  "query": "sentiment",
  "type_filter": "feature_generator"
}
```

**Response:**
```json
{
  "query": "sentiment",
  "total_results": 1,
  "results": [
    {
      "name": "Sentiment Analyzer",
      "version": "1.0.0",
      "description": "Analyzes sentiment and tone of email content",
      "tags": ["sentiment", "emotion", "nlp", "analysis"],
      "icon": "😊",
      "color": "#F38181"
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/composable/components/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sentiment"}'
```

---

## Pipeline Management

### Create Pipeline

**POST** `/api/composable/pipelines`

Create a new composable pipeline.

**Request Body:**
```json
{
  "name": "Spam and Sentiment Analysis",
  "components": [
    {
      "name": "Spam Detector",
      "config": {
        "threshold": 0.5,
        "keywords": ["free", "winner", "urgent"]
      }
    },
    {
      "name": "Sentiment Analyzer",
      "config": {}
    }
  ]
}
```

**Response:**
```json
{
  "id": "pipeline_20250929_143000",
  "name": "Spam and Sentiment Analysis",
  "components": 2,
  "status": "created"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/composable/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Pipeline",
    "components": [
      {"name": "Spam Detector", "config": {}},
      {"name": "Word Length Analyzer", "config": {}}
    ]
  }'
```

---

### List Pipelines

**GET** `/api/composable/pipelines`

Get all saved pipelines.

**Response:**
```json
{
  "total": 3,
  "pipelines": [
    {
      "id": "pipeline_20250929_143000",
      "name": "Spam and Sentiment Analysis",
      "components": [...],
      "created_at": "2025-09-29T14:30:00",
      "executions": 5
    }
  ]
}
```

---

### Get Pipeline

**GET** `/api/composable/pipelines/{pipeline_id}`

Get details of a specific pipeline.

**Response:**
```json
{
  "id": "pipeline_20250929_143000",
  "name": "Spam and Sentiment Analysis",
  "components": [...],
  "created_at": "2025-09-29T14:30:00",
  "executions": 5
}
```

---

### Execute Pipeline

**POST** `/api/composable/pipelines/{pipeline_id}/execute`

Execute a pipeline on an email.

**Request Body:**
```json
{
  "email_subject": "URGENT: Limited Time Offer",
  "email_body": "Act now to get 50% off! This is a limited time offer for winners only."
}
```

**Response:**
```json
{
  "pipeline_id": "pipeline_20250929_143000",
  "pipeline_name": "Spam and Sentiment Analysis",
  "email_subject": "URGENT: Limited Time Offer",
  "results": [
    {
      "component": "Spam Detector",
      "output": {
        "spam_score": 0.8,
        "spam_keywords": ["urgent", "limited", "winner"],
        "has_spam": true
      }
    },
    {
      "component": "Sentiment Analyzer",
      "output": {
        "sentiment": "neutral",
        "confidence": 0.5,
        "tone": "professional"
      }
    }
  ],
  "execution_count": 6
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/composable/pipelines/pipeline_20250929_143000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "email_subject": "Team Meeting",
    "email_body": "Reminder about our weekly team meeting tomorrow at 10am."
  }'
```

---

### Delete Pipeline

**DELETE** `/api/composable/pipelines/{pipeline_id}`

Delete a saved pipeline.

**Response:**
```json
{
  "status": "deleted",
  "pipeline_id": "pipeline_20250929_143000"
}
```

---

## System Status

### Get System Status

**GET** `/api/composable/status`

Get overall system status and statistics.

**Response:**
```json
{
  "status": "operational",
  "timestamp": "2025-09-29T14:35:24",
  "components": {
    "total": 5,
    "types": {
      "feature_generators": 5
    }
  },
  "pipelines": {
    "total": 3,
    "total_executions": 15
  },
  "neo4j": {
    "emails": 55,
    "labeled": 55,
    "models": 1
  }
}
```

---

### Get Marketplace Data

**GET** `/api/composable/marketplace`

Get data for component marketplace view.

**Response:**
```json
{
  "total_components": 5,
  "categories": {
    "feature_generator": [
      {
        "name": "Spam Detector",
        "version": "1.0.0",
        "description": "Detects spam keywords and patterns",
        "author": "John Affolter",
        "tags": ["spam", "detection"],
        "icon": "🚫",
        "color": "#FF6B6B",
        "downloads": 0,
        "rating": 4.5
      }
    ]
  },
  "featured": [...],
  "recently_added": []
}
```

---

## WebSocket - Real-Time Updates

### System Status Stream

**WebSocket** `/api/composable/ws/status`

Connect for real-time system status updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/composable/ws/status');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status update:', data);
};
```

**Message Format:**
```json
{
  "type": "status_update",
  "data": {
    "status": "operational",
    "timestamp": "2025-09-29T14:35:26",
    "components": {...},
    "pipelines": {...},
    "neo4j": {...}
  },
  "timestamp": "2025-09-29T14:35:26"
}
```

**Updates:** Every 2 seconds

---

## Available Components

### 1. 🚫 Spam Detector

**Type:** feature_generator
**Version:** 1.0.0

Detects spam keywords and patterns in email content.

**Configuration:**
```json
{
  "keywords": ["free", "winner", "urgent", "click", "limited"],
  "threshold": 0.3
}
```

**Output:**
- `spam_score`: float (0-1)
- `spam_keywords`: List[str]
- `has_spam`: bool

---

### 2. 📏 Word Length Analyzer

**Type:** feature_generator
**Version:** 1.0.0

Analyzes average word length and vocabulary richness.

**Configuration:**
```json
{
  "include_subject": true
}
```

**Output:**
- `avg_word_length`: float
- `total_words`: int
- `unique_words`: int
- `vocabulary_richness`: float

---

### 3. 🔢 Email Embedder

**Type:** feature_generator
**Version:** 1.0.0

Generates numerical embeddings from email content.

**Configuration:**
```json
{
  "normalize": true
}
```

**Output:**
- `embedding_vector`: float
- `total_chars`: int

---

### 4. 😊 Sentiment Analyzer

**Type:** feature_generator
**Version:** 1.0.0

Analyzes sentiment and tone of email content.

**Configuration:**
```json
{
  "method": "lexicon"
}
```

**Output:**
- `sentiment`: str (positive/negative/neutral)
- `confidence`: float
- `tone`: str

---

### 5. ⚡ Urgency Detector

**Type:** feature_generator
**Version:** 1.0.0

Detects urgency and priority indicators in emails.

**Configuration:**
```json
{
  "urgency_keywords": ["urgent", "asap", "immediately", "critical", "emergency"]
}
```

**Output:**
- `urgency_score`: float
- `urgency_level`: str (high/medium/low)
- `urgency_keywords`: List[str]

---

## Web UI

### Access the UI

Navigate to: `http://localhost:8000/static/index.html`

The web UI provides:
- 📊 Real-time system statistics
- 📚 Component library browser
- 🧪 Interactive pipeline tester
- 🔗 Visual pipeline builder (click components to add)
- ⚡ Live execution and results

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (component/pipeline not found)
- `500` - Internal Server Error

---

## Rate Limiting

Currently no rate limiting. In production, implement:
- 100 requests per minute per IP
- 1000 pipeline executions per hour

---

## Examples

### Complete Workflow

```bash
# 1. List available components
curl http://localhost:8000/api/composable/components

# 2. Create a pipeline
curl -X POST http://localhost:8000/api/composable/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Email Analyzer",
    "components": [
      {"name": "Spam Detector", "config": {"threshold": 0.4}},
      {"name": "Sentiment Analyzer", "config": {}},
      {"name": "Urgency Detector", "config": {}}
    ]
  }'

# 3. Execute the pipeline
curl -X POST http://localhost:8000/api/composable/pipelines/pipeline_20250929_143000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "email_subject": "URGENT: Server Down",
    "email_body": "Critical issue detected. Immediate action required!"
  }'

# 4. Get system status
curl http://localhost:8000/api/composable/status
```

---

## Next Steps

1. **Authentication** - Add JWT-based auth
2. **Persistence** - Store pipelines in database
3. **Versioning** - Track component versions
4. **Monitoring** - Add Prometheus metrics
5. **Caching** - Redis for component registry
6. **GraphQL** - Alternative API interface

---

## Support

- **Documentation:** This file
- **Demo:** `demo_composable_system.py`
- **Architecture:** `COMPOSABILITY_ARCHITECTURE.md`
- **Author:** John Affolter <affo4353@stthomas.edu>

**Course:** MLOps - St. Thomas University
**Date:** 2025-09-29