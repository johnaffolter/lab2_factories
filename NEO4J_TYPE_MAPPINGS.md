# Neo4j and Python Type Mappings for MLOps System

Complete mapping of all Python classes, data structures, and their Neo4j graph representations.

## Core Entities

### 1. Email (Python Dataclass → Neo4j Node)

**Python:**
```python
@dataclass
class Email:
    subject: str
    body: str
```

**Neo4j:**
```cypher
(:Email {
    id: String (MD5 hash),
    subject: String,
    body: String,
    sender: String,
    timestamp: DateTime,
    word_count: Integer,
    char_count: Integer
})
```

**Relationships:**
- `(:Email)-[:CLASSIFIED_AS {confidence: Float, all_scores: String}]->(:Topic)`
- `(:Email)-[:HAS_GROUND_TRUTH {annotator: String, verified: Boolean}]->(:Topic)`
- `(:MLModel)-[:PREDICTED {confidence: Float, processing_time_ms: Integer}]->(:Email)`
- `(:FeatureGenerator)-[:EXTRACTED_FROM {features: String (JSON)}]->(:Email)`

---

### 2. Topic (Implicit in Python → Neo4j Node)

**Python:**
```python
# Topics defined in data/topic_keywords.json
topics = ["work", "personal", "promotion", "newsletter", "support", "travel", "education", "health"]
```

**Neo4j:**
```cypher
(:Topic {
    name: String (unique),
    created_at: DateTime
})
```

**Relationships:**
- `(:Email)-[:CLASSIFIED_AS]->(:Topic)`
- `(:Email)-[:HAS_GROUND_TRUTH]->(:Topic)`
- `(:Trial)-[:PREDICTED]->(:Topic)`

---

### 3. MLModel (Python Class → Neo4j Node)

**Python:**
```python
class EmailClassifierModel:
    def __init__(self, use_email_similarity: bool = False):
        self.use_email_similarity = use_email_similarity
        self.topic_data = Dict[str, Dict[str, Any]]
        self.topics = List[str]
        self.stored_emails = List[Dict[str, Any]]

    def predict(self, features: Dict[str, Any]) -> str
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]
```

**Neo4j:**
```cypher
(:MLModel {
    id: String (unique),
    model_type: String,
    version: String,
    accuracy: Float,
    f1_score: Float,
    training_date: DateTime,
    hyperparameters: String (JSON),
    s3_path: String,
    training_size: Integer,
    status: String,
    created_at: DateTime
})
```

**Relationships:**
- `(:MLModel)-[:PREDICTED {confidence: Float, timestamp: DateTime}]->(:Email)`
- `(:TrainingFlow)-[:PRODUCED]->(:MLModel)`

---

### 4. FeatureGenerator (Python Classes → Neo4j Nodes)

**Python:**
```python
class BaseFeatureGenerator(ABC):
    @abstractmethod
    def generate(self, email: Email) -> Dict[str, Any]

# Concrete implementations:
class SpamFeatureGenerator(BaseFeatureGenerator)
class AverageWordLengthFeatureGenerator(BaseFeatureGenerator)
class EmailEmbeddingsFeatureGenerator(BaseFeatureGenerator)
class RawEmailFeatureGenerator(BaseFeatureGenerator)
class NonTextCharacterFeatureGenerator(BaseFeatureGenerator)
```

**Neo4j:**
```cypher
(:FeatureGenerator {
    name: String (unique),
    category: String,
    version: String
})
```

**Categories:**
- `content_analysis` - spam, non_text
- `linguistic_analysis` - word_length
- `ml_features` - email_embeddings
- `raw_features` - raw_email

**Relationships:**
- `(:FeatureGenerator)-[:EXTRACTED_FROM {features: String (JSON), timestamp: DateTime}]->(:Email)`
- `(:DesignPattern)-[:USED_BY]->(:FeatureGenerator)`

---

### 5. FeatureGeneratorFactory (Python Class → Neo4j Metadata)

**Python:**
```python
class FeatureGeneratorFactory:
    _generators: Dict[str, Type[BaseFeatureGenerator]]
    _cache: Dict[str, BaseFeatureGenerator]
    _statistics: Dict[str, Any]

    def create_generator(self, generator_type: str) -> BaseFeatureGenerator
    def generate_all_features(self, email: Email) -> Dict[str, Any]
    def get_available_generators(self) -> List[Dict[str, Any]]
```

**Neo4j:**
Represented through relationships between DesignPattern and FeatureGenerator nodes:
```cypher
(:DesignPattern {name: "Factory Pattern"})-[:IMPLEMENTS]->(:FeatureGenerator)
```

---

### 6. Experiment (Python Dict → Neo4j Node)

**Python:**
```python
experiment_data = {
    'experiment_id': str,
    'models': List[str],
    'dataset_size': int,
    'accuracy': float,
    's3_path': str,
    'timestamp': str,
    'hyperparameters': Dict[str, Any],
    'status': str
}
```

**Neo4j:**
```cypher
(:Experiment {
    id: String (unique),
    timestamp: DateTime,
    dataset_size: Integer,
    accuracy: Float,
    s3_path: String,
    hyperparameters: String (JSON),
    status: String
})
```

**Relationships:**
- `(:Experiment)-[:TRAINED]->(:MLModel)`

---

### 7. TrainingFlow (Python Dict → Neo4j Node)

**Python:**
```python
flow_data = {
    'flow_id': str,
    'name': str,
    'steps': List[str],
    'input_data': Dict,
    'output_models': List[str],
    'execution_time': int,
    'status': str,
    'airflow_dag_id': str,
    's3_artifacts': Dict
}
```

**Neo4j:**
```cypher
(:TrainingFlow {
    id: String (unique),
    name: String,
    steps: String (JSON array),
    execution_time_sec: Integer,
    status: String,
    airflow_dag_id: String,
    s3_artifacts: String (JSON),
    created_at: DateTime
})
```

**Relationships:**
- `(:TrainingFlow)-[:USED_DATA]->(:Dataset)`
- `(:TrainingFlow)-[:PRODUCED]->(:MLModel)`

---

### 8. Dataset (Python Dict → Neo4j Node)

**Python:**
```python
dataset = {
    'id': str,
    'size': int,
    'source': str,
    'samples': List[Dict]
}
```

**Neo4j:**
```cypher
(:Dataset {
    id: String (unique),
    size: Integer,
    source: String
})
```

**Relationships:**
- `(:TrainingFlow)-[:USED_DATA]->(:Dataset)`

---

### 9. DesignPattern (Metadata → Neo4j Node)

**Python:**
```python
patterns = [
    {
        "name": str,
        "component": str,
        "purpose": str,
        "benefits": List[str]
    }
]
```

**Neo4j:**
```cypher
(:DesignPattern {
    name: String (unique),
    component: String,
    purpose: String,
    benefits: String (JSON array),
    updated_at: DateTime
})
```

**Patterns in System:**
- Factory Pattern → FeatureGeneratorFactory
- Strategy Pattern → BaseFeatureGenerator
- Dataclass Pattern → Email
- Registry Pattern → GENERATORS Registry

**Relationships:**
- `(:DesignPattern)-[:USED_BY]->(:FeatureGenerator)`
- `(:DesignPattern)-[:IMPLEMENTS]->(:FeatureGenerator)`
- `(:DesignPattern)-[:CREATES]->(:MLModel)`

---

### 10. Trial (Python Dict → Neo4j Node)

**Python:**
```python
trial_data = {
    'input': {
        'subject': str,
        'body': str,
        'use_learning': bool
    },
    'ai_output': {
        'predicted_topic': str,
        'confidence_scores': Dict[str, float],
        'features_extracted': Dict[str, Any],
        'processing_time_ms': int
    }
}
```

**Neo4j:**
```cypher
(:Trial {
    id: String (unique),
    timestamp: DateTime,
    subject: String,
    predicted_topic: String,
    confidence: Float,
    processing_time: Integer,
    use_learning: Boolean,
    features: String (JSON),
    scores: String (JSON)
})
```

**Relationships:**
- `(:Trial)-[:PREDICTED {confidence: Float}]->(:Topic)`

---

## Feature Type Mappings

### Features Dictionary (Python → Neo4j)

**Python:**
```python
features: Dict[str, Any] = {
    # Spam features
    'spam_spam_keyword_count': int,
    'spam_has_urgent_words': bool,
    'spam_has_money_words': bool,
    'spam_spam_score': float,

    # Word length features
    'word_length_average_word_length': float,
    'word_length_long_word_count': int,
    'word_length_short_word_count': int,

    # Email embeddings
    'email_embeddings_average_embedding': float,

    # Raw email
    'raw_email_email_subject': str,
    'raw_email_email_body': str,

    # Non-text features
    'non_text_special_char_count': int,
    'non_text_exclamation_count': int
}
```

**Neo4j Storage:**
```cypher
(:FeatureGenerator)-[:EXTRACTED_FROM {
    features: "{
        'spam_spam_keyword_count': 3,
        'spam_spam_score': 0.85,
        ...
    }"
}]->(:Email)
```

---

## Complete Graph Schema

### Node Labels
```cypher
(:Email)              - Email messages
(:Topic)              - Classification topics
(:MLModel)            - Machine learning models
(:FeatureGenerator)   - Feature extraction components
(:Experiment)         - Training experiments
(:TrainingFlow)       - MLOps pipelines
(:Dataset)            - Training datasets
(:DesignPattern)      - Software design patterns
(:Trial)              - Classification attempts
```

### Relationship Types
```cypher
-[:CLASSIFIED_AS]->    Email to Topic (predicted)
-[:HAS_GROUND_TRUTH]-> Email to Topic (true label)
-[:PREDICTED]->        Model/Trial to Email/Topic
-[:EXTRACTED_FROM]->   FeatureGenerator to Email
-[:TRAINED]->          Experiment to Model
-[:PRODUCED]->         TrainingFlow to Model
-[:USED_DATA]->        TrainingFlow to Dataset
-[:USED_BY]->          DesignPattern to Component
-[:IMPLEMENTS]->       DesignPattern to Component
-[:CREATES]->          DesignPattern to Component
```

---

## Data Type Conversions

### Python → Neo4j Type Mapping

| Python Type | Neo4j Type | Storage Method |
|-------------|-----------|----------------|
| `str` | `String` | Direct |
| `int` | `Integer` | Direct |
| `float` | `Float` | Direct |
| `bool` | `Boolean` | Direct |
| `datetime` | `DateTime` | `datetime($iso_string)` |
| `Dict` | `String` | `json.dumps()` then store as String |
| `List` | `String` | `json.dumps()` then store as String |
| `dataclass` | `Node` | Map fields to properties |
| `None` | `null` | Omit property |

### Example Conversions

**Python Dict to Neo4j:**
```python
# Python
hyperparameters = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}

# Neo4j
SET m.hyperparameters = '{"learning_rate": 0.001, "batch_size": 32, "epochs": 10}'
```

**Python List to Neo4j:**
```python
# Python
steps = ["load_data", "preprocess", "train", "evaluate"]

# Neo4j
SET f.steps = '["load_data", "preprocess", "train", "evaluate"]'
```

**Python datetime to Neo4j:**
```python
# Python
timestamp = datetime.now().isoformat()  # "2025-09-29T10:30:00"

# Neo4j
SET e.timestamp = datetime("2025-09-29T10:30:00")
```

---

## Query Examples

### Get Email with All Context
```cypher
MATCH (e:Email {id: $email_id})
OPTIONAL MATCH (e)-[r1:CLASSIFIED_AS]->(t:Topic)
OPTIONAL MATCH (m:MLModel)-[r2:PREDICTED]->(e)
OPTIONAL MATCH (g:FeatureGenerator)-[r3:EXTRACTED_FROM]->(e)
OPTIONAL MATCH (e)-[r4:HAS_GROUND_TRUTH]->(gt:Topic)
RETURN e,
       collect(DISTINCT {topic: t.name, confidence: r1.confidence}) as predictions,
       collect(DISTINCT {model: m.model_type, confidence: r2.confidence}) as models,
       collect(DISTINCT {generator: g.name, features: r3.features}) as features,
       gt.name as ground_truth
```

### Get Model Lineage
```cypher
MATCH (m:MLModel {id: $model_id})
OPTIONAL MATCH (f:TrainingFlow)-[:PRODUCED]->(m)
OPTIONAL MATCH (f)-[:USED_DATA]->(d:Dataset)
OPTIONAL MATCH (m)-[p:PREDICTED]->(e:Email)
RETURN m, f, d, count(e) as predictions, avg(p.confidence) as avg_confidence
```

### Get Training Examples
```cypher
MATCH (e:Email)-[:HAS_GROUND_TRUTH]->(t:Topic)
OPTIONAL MATCH (g:FeatureGenerator)-[rf:EXTRACTED_FROM]->(e)
RETURN e.id, e.subject, e.body, t.name as label,
       collect({generator: g.name, features: rf.features}) as features
```

---

## System Overview

```
Python Application Layer           Neo4j Knowledge Graph
┌─────────────────────────┐       ┌─────────────────────────┐
│                         │       │                         │
│  Email (Dataclass)      │──────▶│  (:Email)              │
│  - subject: str         │       │  - id, subject, body    │
│  - body: str            │       │                         │
│                         │       │                         │
│  EmailClassifierModel   │──────▶│  (:MLModel)            │
│  - predict()            │       │  - model_type, version  │
│  - get_topic_scores()   │       │                         │
│                         │       │  (:MLModel)-[:PREDICTED]│
│  FeatureGenerator       │──────▶│  (:FeatureGenerator)   │
│  - generate()           │       │  - name, category       │
│                         │       │                         │
│  FeatureGeneratorFactory│──────▶│  (:DesignPattern)      │
│  - create_generator()   │       │  - Factory Pattern      │
│                         │       │                         │
│  MLOpsKnowledgeGraph    │◀─────▶│  Complete Graph        │
│  - store_*()            │       │  All Entities & Rels    │
│  - get_*()              │       │                         │
└─────────────────────────┘       └─────────────────────────┘
```

This comprehensive mapping ensures all Python types are properly represented in Neo4j, enabling full MLOps lineage tracking, feature traceability, and model governance.