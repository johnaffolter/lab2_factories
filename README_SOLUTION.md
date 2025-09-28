# Enhanced Email Classification System

## Assignment Solution for Lab 2 - Factory Pattern

### Student: John Affolter
### Repository: Private (Shared with @jhoward)

---

## 📋 Assignment Requirements Completed

✅ **Forked the lab2_factories repository**
✅ **Created endpoint to dynamically add new topics**
✅ **Created endpoint to store emails with optional ground truth**
✅ **Updated classifier to use topic or email similarity**
✅ **Demonstrated creating new topics**
✅ **Demonstrated inference on new topics**
✅ **Demonstrated adding new emails**
✅ **Demonstrated inference from email data**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit matplotlib  # For UI
```

### 2. Start the API Server
```bash
uvicorn app.main:app --reload
```

### 3. Run Streamlit UI (Optional)
```bash
streamlit run streamlit_app.py
```

### 4. Run Tests
```bash
python test_new_features.py
```

## 📦 Solution Components

### Core Implementation Files

1. **`app/features/generators.py`** - Added NonTextCharacterFeatureGenerator
2. **`app/features/factory.py`** - Updated factory with new generator
3. **`app/api/routes.py`** - Added new endpoints for topics and emails
4. **`app/models/similarity_model.py`** - Enhanced with dual classification modes
5. **`app/models/llm_classifier.py`** - Optional LLM classifier (commented)

### Additional Components

6. **`streamlit_app.py`** - Interactive web UI for demonstration
7. **`airflow_dags/email_classification_dag.py`** - Automation pipeline
8. **`test_new_features.py`** - Comprehensive test suite

## 🔧 New API Endpoints

### 1. Dynamic Topic Management
```http
POST /topics
Content-Type: application/json

{
  "topic_name": "finance",
  "description": "Financial emails, invoices, and banking"
}
```

### 2. Email Storage with Ground Truth
```http
POST /emails
Content-Type: application/json

{
  "subject": "Meeting Tomorrow",
  "body": "Join us at 2pm",
  "ground_truth": "work"  // Optional
}
```

### 3. Enhanced Classification
```http
POST /emails/classify
Content-Type: application/json

{
  "subject": "Your Order Shipped",
  "body": "Track your package...",
  "use_email_similarity": true  // New parameter
}
```

### 4. Feature Generators Info
```http
GET /features
```

### 5. Get Stored Emails
```http
GET /emails
```

## 📊 Classification Modes

### 1. Topic Similarity (Default)
- Uses cosine similarity with topic descriptions
- Best for general classification
- No training data required

### 2. Email Similarity
- Compares with stored emails having ground truth
- Learns from examples
- Improves over time

### 3. LLM Classification (Optional)
- Uses OpenAI GPT or Anthropic Claude
- Highest accuracy
- Requires API key

## 🎯 Test Results

```
✅ Feature Generators Test
   - Successfully retrieved 5 generators
   - New NonTextCharacterFeatureGenerator working

✅ Dynamic Topics Test
   - Added: travel, education, health
   - Topics persist to file

✅ Email Storage Test
   - Stored 5 emails with ground truth
   - Auto-incrementing IDs working

✅ Classification Modes Test
   - Topic similarity operational
   - Email similarity functional

✅ Non-Text Feature Test
   - Correctly counting special characters
   - Integration with factory pattern successful
```

## 📸 Demonstration Screenshots

### Streamlit UI - Main Classification Interface
The Streamlit app provides an intuitive interface for:
- Email classification with dual modes
- Dynamic topic management
- Email storage with ground truth
- Real-time analytics

### API Documentation (FastAPI)
Access interactive API docs at: `http://localhost:8000/docs`
- Try out endpoints directly
- View request/response schemas
- Test with sample data

## 🔄 Airflow Integration

The solution includes a production-ready Airflow DAG that:
- Fetches new emails hourly
- Classifies in batch
- Stores high-confidence predictions
- Generates analytics reports
- Performs maintenance tasks

## 🎓 Learning Objectives Achieved

1. ✅ **Factory Pattern Implementation**
   - Extended with new generator
   - Maintained SOLID principles
   - Clean abstraction

2. ✅ **Dynamic System Design**
   - Topics can be added at runtime
   - System learns from examples
   - No code changes needed

3. ✅ **REST API Design**
   - Proper HTTP methods
   - Consistent error handling
   - Well-structured responses

4. ✅ **Machine Learning Pipeline**
   - Multiple classification strategies
   - Feature generation pipeline
   - Training data management

## 🛠️ Technical Highlights

### Factory Pattern Extension
```python
# New generator added seamlessly
class NonTextCharacterFeatureGenerator(BaseFeatureGenerator):
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Implementation
        return {"non_text_char_count": count}
```

### Dual Classification Strategy
```python
# Flexible classification based on mode
if use_email_similarity:
    return self._predict_by_email_similarity(features)
else:
    return self._predict_by_topic_similarity(features)
```

### Dynamic Topic Addition
```python
# Runtime topic management
topics[request.topic_name] = {
    "description": request.description
}
```

## 📈 Future Enhancements

1. **Batch Classification API** - Process multiple emails
2. **Confidence Thresholds** - Configurable similarity thresholds
3. **Topic Analytics** - Track classification patterns
4. **Model Retraining** - Automated retraining pipeline
5. **Multi-Language Support** - Classify non-English emails

## 🔒 Repository Access

This repository is **private** and shared with:
- Professor: [@jhoward](https://github.com/jhoward)

## 📝 Conclusion

This solution successfully extends the original Factory Pattern email classification system with:
- Dynamic topic management
- Learning from labeled data
- Multiple classification strategies
- Production-ready UI and automation

The modular design ensures easy maintenance and future extensions while maintaining the elegance of the Factory Pattern.

---

**Submitted by:** John Affolter
**Course:** MLOps - St. Thomas
**Assignment:** Lab 2 - Factory Pattern Extensions