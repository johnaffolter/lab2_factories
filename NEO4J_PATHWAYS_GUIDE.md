# Neo4j Integration Pathways Guide

**System:** MLOps Email Classification with Knowledge Graph
**Date:** 2025-09-29
**Author:** John Affolter

---

## Overview

This guide documents all pathways for data flow into and out of the Neo4j knowledge graph, including email storage, classification tracking, model versioning, and validation workflows.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEO4J KNOWLEDGE GRAPH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Node Types:                                                     │
│  • Email           - Email messages with metadata               │
│  • Topic           - Classification categories                  │
│  • MLModel         - Versioned ML models                        │
│  • FeatureGenerator- Feature extraction components             │
│  • TrainingFlow    - Pipeline execution records                │
│  • Dataset         - Training data collections                  │
│  • Experiment      - Training experiments                       │
│  • DesignPattern   - Software patterns used                     │
│                                                                  │
│  Relationships:                                                  │
│  • CLASSIFIED_AS   - Email → Topic (predicted)                  │
│  • HAS_GROUND_TRUTH- Email → Topic (actual)                     │
│  • PREDICTED       - MLModel → Email                            │
│  • EXTRACTED_FROM  - FeatureGenerator → Email                   │
│  • PRODUCED        - TrainingFlow → MLModel                     │
│  • USED_DATA       - TrainingFlow → Dataset                     │
│  • TRAINED         - Experiment → MLModel                       │
│  • IMPLEMENTS      - Component → DesignPattern                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Entry Points (15 Pathways)

### 1. **store_email_with_classification()**

**Purpose:** Primary pathway for storing emails with classification results

**Data Flow:**
```
Email Data + Classification Result
    ↓
1. Create Email node (with metadata)
    ↓
2. Create/Merge Topic node
    ↓
3. Create CLASSIFIED_AS relationship
    ↓
4. Create MLModel node
    ↓
5. Create PREDICTED relationship
    ↓
6. Create FeatureGenerator nodes
    ↓
7. Create EXTRACTED_FROM relationships
    ↓
Return: email_id
```

**Input Format:**
```python
email_data = {
    "subject": str,
    "body": str,
    "sender": str,
    "timestamp": str (ISO format)
}

classification_result = {
    "predicted_label": str,
    "confidence": float,
    "all_scores": Dict[str, float],
    "model_type": str,
    "features": Dict[str, Any]
}
```

**Usage Example:**
```python
from app.services.mlops_neo4j_integration import get_knowledge_graph

kg = get_knowledge_graph()
email_id = kg.store_email_with_classification(email_data, classification_result)
```

**Cypher Queries Executed:**
1. `MERGE (e:Email {id: $email_id}) SET e.subject = ..., e.body = ...`
2. `MERGE (t:Topic {name: $topic_name})`
3. `MERGE (e)-[r:CLASSIFIED_AS]->(t) SET r.confidence = ...`
4. `MERGE (m:MLModel {id: $model_id})`
5. `MERGE (m)-[r:PREDICTED]->(e)`

**Performance:** ~200ms per email (5 Cypher queries)

---

### 2. **store_ground_truth()**

**Purpose:** Store human-validated ground truth labels

**Data Flow:**
```
email_id + true_label + annotator
    ↓
1. Find Email node
    ↓
2. Find/Create Topic node (ground truth)
    ↓
3. Create HAS_GROUND_TRUTH relationship
    ↓
Return: success boolean
```

**Input Format:**
```python
email_id: str       # Email node ID
true_label: str     # Correct topic label
annotator: str      # Who provided the label (default: "human")
```

**Usage Example:**
```python
success = kg.store_ground_truth(
    email_id="abc123",
    true_label="work",
    annotator="human_annotator"
)
```

**Cypher Query:**
```cypher
MATCH (e:Email {id: $email_id})
MERGE (t:Topic {name: $true_label})
MERGE (e)-[r:HAS_GROUND_TRUTH]->(t)
SET r.annotator = $annotator,
    r.timestamp = datetime()
```

**Use Cases:**
- Manual email labeling
- Validation of predictions
- Training data curation
- Model evaluation

---

### 3. **store_model_version()**

**Purpose:** Register ML model versions with metadata

**Data Flow:**
```
model_data (metadata, performance metrics)
    ↓
Create/Update MLModel node with version info
    ↓
Return: model_id
```

**Input Format:**
```python
model_data = {
    "model_type": str,
    "version": str,
    "accuracy": float,
    "f1_score": float,
    "training_date": str,
    "hyperparameters": Dict,
    "features_used": List[str]
}
```

**Usage Example:**
```python
model_id = kg.store_model_version({
    "model_type": "EmailClassifierModel",
    "version": "1.2.0",
    "accuracy": 0.87,
    "f1_score": 0.85,
    "training_date": "2025-09-29",
    "hyperparameters": {"learning_rate": 0.001},
    "features_used": ["spam_keywords", "word_length", "embeddings"]
})
```

**Cypher Query:**
```cypher
MERGE (m:MLModel {id: $model_id})
SET m.model_type = $model_type,
    m.version = $version,
    m.accuracy = $accuracy,
    m.f1_score = $f1_score,
    m.training_date = datetime($training_date),
    m.hyperparameters = $hyperparameters,
    m.features_used = $features_used,
    m.updated_at = datetime()
```

---

### 4. **store_training_flow()**

**Purpose:** Record complete training pipeline executions

**Data Flow:**
```
flow_data (pipeline metadata, dataset info, models trained)
    ↓
1. Create TrainingFlow node
    ↓
2. Link to Dataset node
    ↓
3. Link to produced MLModel nodes
    ↓
Return: flow_id
```

**Input Format:**
```python
flow_data = {
    "flow_name": str,
    "dataset_id": str,
    "models_produced": List[str],
    "start_time": str,
    "end_time": str,
    "status": str,
    "metrics": Dict
}
```

**Usage Example:**
```python
flow_id = kg.store_training_flow({
    "flow_name": "daily_retrain_pipeline",
    "dataset_id": "dataset_20250929",
    "models_produced": ["model_v1.2", "model_v1.3"],
    "start_time": "2025-09-29T10:00:00Z",
    "end_time": "2025-09-29T10:15:00Z",
    "status": "completed",
    "metrics": {"total_samples": 180, "training_accuracy": 0.89}
})
```

---

### 5. **store_training_experiment()**

**Purpose:** Track ML training experiments with configurations

**Data Flow:**
```
experiment_data (config, results, model info)
    ↓
1. Create Experiment node
    ↓
2. Link to trained MLModel
    ↓
3. Link to Dataset used
    ↓
Return: experiment_id
```

**Input Format:**
```python
experiment_data = {
    "experiment_name": str,
    "model_id": str,
    "dataset_id": str,
    "config": Dict,
    "results": Dict,
    "timestamp": str
}
```

---

### 6. **generate_training_examples()**

**Purpose:** Query Neo4j to generate training datasets

**Data Flow:**
```
count parameter
    ↓
Query emails with ground truth labels
    ↓
Extract features from relationships
    ↓
Return: List of training examples
```

**Output Format:**
```python
[
    {
        "id": str,
        "text": str,
        "features": Dict,
        "label": str,
        "metadata": Dict
    },
    ...
]
```

**Usage Example:**
```python
training_data = kg.generate_training_examples(count=100)
# Returns 100 labeled emails with features for model training
```

**Cypher Query:**
```cypher
MATCH (e:Email)-[:HAS_GROUND_TRUTH]->(t:Topic)
OPTIONAL MATCH (g:FeatureGenerator)-[rf:EXTRACTED_FROM]->(e)
RETURN e.id as id,
       e.subject as subject,
       e.body as body,
       t.name as label,
       collect({generator: g.name, features: rf.features}) as features
LIMIT $count
```

---

### 7. **query_similar_emails()**

**Purpose:** Find emails similar to a given email

**Data Flow:**
```
email_id + similarity threshold
    ↓
Query for emails with same Topic
    ↓
Calculate feature similarity
    ↓
Return: List of similar emails
```

**Usage Example:**
```python
similar = kg.query_similar_emails(
    email_id="abc123",
    threshold=0.7,
    limit=10
)
```

---

### 8. **get_email_knowledge_graph()**

**Purpose:** Get complete graph structure for one email

**Data Flow:**
```
email_id
    ↓
Query all relationships:
  - Classifications
  - Ground truth
  - Predictions
  - Features extracted
    ↓
Return: Complete email graph
```

**Output Format:**
```python
{
    "email": {...},
    "classifications": [...],
    "ground_truth": {...},
    "predictions": [...],
    "features": [...],
    "similar_emails": [...]
}
```

---

### 9. **get_model_lineage()**

**Purpose:** Trace complete lineage of a model

**Data Flow:**
```
model_id
    ↓
Query:
  - Training flow that produced it
  - Dataset used
  - Experiments run
  - Performance metrics
  - Predictions made
    ↓
Return: Complete model lineage
```

**Usage Example:**
```python
lineage = kg.get_model_lineage(model_id="model_v1.2")
# Returns: training_flow, dataset, experiments, metrics, predictions_count
```

---

### 10. **export_training_dataset()**

**Purpose:** Export curated training dataset

**Data Flow:**
```
topic filter + confidence threshold
    ↓
Query emails with ground truth
    ↓
Filter by confidence (optional)
    ↓
Extract all features
    ↓
Return: Curated training dataset
```

**Usage Example:**
```python
dataset = kg.export_training_dataset(
    topic="work",
    min_confidence=0.8
)
# Returns all "work" emails with confidence >= 0.8
```

---

### 11. **compare_models()**

**Purpose:** Compare multiple model versions

**Data Flow:**
```
List of model_ids
    ↓
Query performance metrics for each
    ↓
Aggregate predictions and accuracy
    ↓
Return: Comparison table
```

**Output Format:**
```python
{
    "models": [
        {
            "model_id": str,
            "model_type": str,
            "accuracy": float,
            "f1_score": float,
            "predictions": int,
            "avg_confidence": float
        },
        ...
    ],
    "comparison_timestamp": str
}
```

---

### 12. **get_mlops_system_overview()**

**Purpose:** Get high-level system statistics

**Data Flow:**
```
No input required
    ↓
Query counts for all node types:
  - Total emails
  - Labeled emails
  - Models
  - Training flows
  - Topics
  - Experiments
    ↓
Return: System overview
```

**Output Format:**
```python
{
    "system_status": "operational",
    "emails": {"total": int, "labeled": int, "unlabeled": int},
    "models": {"total": int, "active": int},
    "training_flows": {"total": int, "recent": int},
    "topics": [...],
    "experiments": {"total": int},
    "timestamp": str
}
```

**Usage Example:**
```python
overview = kg.get_mlops_system_overview()
print(f"Total emails: {overview['emails']['total']}")
print(f"Labeled: {overview['emails']['labeled']}")
```

---

### 13. **get_mlops_dashboard_data()**

**Purpose:** Get comprehensive dashboard data

**Data Flow:**
```
No input
    ↓
Query multiple aggregations:
  - System overview
  - Recent activity
  - Model performance trends
  - Topic distribution
    ↓
Return: Dashboard data
```

---

### 14. **store_design_patterns()**

**Purpose:** Document design patterns used in system

**Data Flow:**
```
List of design patterns
    ↓
Create DesignPattern nodes
    ↓
Link to implementing components
    ↓
Return: success
```

**Input Format:**
```python
patterns = [
    {
        "name": "Factory Pattern",
        "type": "Creational",
        "implemented_by": ["FeatureGeneratorFactory"],
        "description": str
    },
    ...
]
```

---

### 15. **close()**

**Purpose:** Close Neo4j database connection

**Usage:**
```python
kg.close()
```

---

## Data Loading Workflow

### Complete Pipeline: Email → Neo4j

```python
from app.services.mlops_neo4j_integration import get_knowledge_graph
from app.features.factory import FeatureGeneratorFactory
from app.models.similarity_model import EmailClassifierModel
from llm_judge_validator import LLMJudge

# Initialize
kg = get_knowledge_graph()
factory = FeatureGeneratorFactory()
model = EmailClassifierModel()
judge = LLMJudge()

# 1. Extract features
email = Email(subject="Meeting tomorrow", body="Team standup at 10am")
features = factory.generate_all_features(email)

# 2. Classify
predicted_topic = model.predict(features)

# 3. Store in Neo4j
email_id = kg.store_email_with_classification(
    {"subject": email.subject, "body": email.body, "sender": "user@company.com"},
    {"predicted_label": predicted_topic, "features": features, "confidence": 0.8}
)

# 4. Add ground truth
kg.store_ground_truth(email_id, "work", "human")

# 5. Validate with LLM judge
validation = judge.validate_classification(
    email.subject, email.body, predicted_topic, "work"
)

print(f"Stored: {email_id}")
print(f"Validation quality: {validation['quality_score']}")
```

---

## Query Patterns

### Common Cypher Queries

**1. Find all emails classified as 'work':**
```cypher
MATCH (e:Email)-[:CLASSIFIED_AS]->(t:Topic {name: 'work'})
RETURN e.subject, e.body, e.timestamp
ORDER BY e.timestamp DESC
LIMIT 10
```

**2. Find misclassified emails:**
```cypher
MATCH (e:Email)-[:CLASSIFIED_AS]->(predicted:Topic)
MATCH (e)-[:HAS_GROUND_TRUTH]->(actual:Topic)
WHERE predicted.name <> actual.name
RETURN e.subject, predicted.name as predicted, actual.name as actual
```

**3. Model performance by topic:**
```cypher
MATCH (e:Email)-[:CLASSIFIED_AS]->(predicted:Topic)
MATCH (e)-[:HAS_GROUND_TRUTH]->(actual:Topic)
WITH actual.name as topic,
     COUNT(e) as total,
     SUM(CASE WHEN predicted.name = actual.name THEN 1 ELSE 0 END) as correct
RETURN topic,
       total,
       correct,
       toFloat(correct) / total as accuracy
ORDER BY accuracy DESC
```

**4. Feature generator usage:**
```cypher
MATCH (g:FeatureGenerator)-[:EXTRACTED_FROM]->(e:Email)
RETURN g.name as generator,
       COUNT(e) as emails_processed
ORDER BY emails_processed DESC
```

**5. Recent training flows:**
```cypher
MATCH (f:TrainingFlow)-[:PRODUCED]->(m:MLModel)
MATCH (f)-[:USED_DATA]->(d:Dataset)
RETURN f.flow_name,
       f.status,
       f.start_time,
       COUNT(m) as models_produced,
       d.name as dataset
ORDER BY f.start_time DESC
LIMIT 5
```

---

## Performance Considerations

### Optimizations

1. **Indexes Created:**
   - `email_id` (unique constraint)
   - `model_id` (unique constraint)
   - `topic_name` (unique constraint)
   - `email_timestamp` (range index)
   - `prediction_confidence` (relationship property index)

2. **Batch Operations:**
   - Use transactions for multiple emails
   - Batch feature extraction
   - Limit LLM validation to samples

3. **Query Optimization:**
   - Use `MERGE` instead of `CREATE` for idempotency
   - Index on frequently queried properties
   - Limit result sets with `LIMIT`

---

## Integration Points

### 1. Airflow DAGs

**File:** `dags/mlops_neo4j_complete_pipeline.py`

```python
def store_in_neo4j(**context):
    kg = get_knowledge_graph()
    emails = context["ti"].xcom_pull(task_ids="extract_features")

    for email_data in emails:
        kg.store_email_with_classification(...)
```

### 2. FastAPI Endpoints

**File:** `app/api/routes.py`

```python
@app.post("/api/classify")
async def classify_email(email: EmailInput):
    # Classify email
    result = model.predict(features)

    # Store in Neo4j
    kg.store_email_with_classification(email_data, result)

    return result
```

### 3. Training Scripts

**File:** `load_data_to_neo4j.py`

```python
def load_expanded_data_to_neo4j(limit: int = None):
    for email in emails:
        # Extract, classify, store
        kg.store_email_with_classification(...)
        kg.store_ground_truth(...)
```

---

## Testing Status

| Pathway | Status | Test Coverage |
|---------|--------|---------------|
| store_email_with_classification | ✅ TESTED | 55+ emails stored |
| store_ground_truth | ✅ TESTED | 55+ labels added |
| store_model_version | ⚠️ PARTIAL | Manual tests only |
| store_training_flow | ⚠️ PARTIAL | Schema created |
| generate_training_examples | ✅ TESTED | Working correctly |
| query_similar_emails | ⚠️ PARTIAL | Schema ready |
| get_email_knowledge_graph | ⚠️ PARTIAL | Schema ready |
| get_model_lineage | ⚠️ PARTIAL | Schema ready |
| export_training_dataset | ⚠️ PARTIAL | Schema ready |
| compare_models | ⚠️ PARTIAL | Schema ready |
| get_mlops_system_overview | ✅ TESTED | Working correctly |
| get_mlops_dashboard_data | ⚠️ PARTIAL | Schema ready |
| store_design_patterns | ⚠️ PARTIAL | Schema ready |

---

## Current Statistics

**As of 2025-09-29 14:45:**

- **Total Emails:** 55
- **Labeled Emails:** 55 (100%)
- **Topics Tracked:** 12 (work, personal, promotion, newsletter, support, travel, education, health, finance, shopping, social, entertainment)
- **Models:** 1 (EmailClassifierModel)
- **Prediction Accuracy:** 10% (simple rule-based model, expected to improve with ML)
- **Validation Quality:** 0.56 average (LLM judge with mock fallback)

---

## Future Enhancements

1. **Advanced Querying:**
   - Temporal queries (emails over time)
   - Graph algorithms (PageRank for important emails)
   - Recommendation system (similar email suggestions)

2. **Improved Performance:**
   - Batch inserts for large datasets
   - Async Neo4j operations
   - Connection pooling

3. **Enhanced Validation:**
   - Real-time LLM validation
   - Active learning loop
   - Confidence calibration

4. **Visualization:**
   - Neo4j Bloom integration
   - Custom dashboards
   - Interactive graph exploration

---

## References

- **Neo4j Python Driver:** https://neo4j.com/docs/python-manual/current/
- **Cypher Query Language:** https://neo4j.com/docs/cypher-manual/current/
- **Graph Data Science:** https://neo4j.com/docs/graph-data-science/current/

---

**Author:** John Affolter <affo4353@stthomas.edu>
**Course:** MLOps - St. Thomas University
**Date:** 2025-09-29