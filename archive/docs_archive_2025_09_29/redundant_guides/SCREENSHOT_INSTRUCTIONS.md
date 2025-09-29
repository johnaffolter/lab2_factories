# Step-by-Step Instructions for Screenshots and Testing

## Prerequisites Check

First, ensure the server is running:
```bash
# Check if server is running
curl http://localhost:8000/health
```

## Screenshot 1: Swagger API Documentation

1. Open your web browser
2. Navigate to: `http://localhost:8000/docs`
3. Take a full-page screenshot showing:
   - All available endpoints
   - The POST /topics endpoint expanded
   - The POST /emails/classify endpoint expanded

**What to capture**: The complete API documentation interface

## Screenshot 2: Web UI Dashboard

1. Open the file: `frontend/enhanced_ui.html` in your browser
   - On Mac: `open frontend/enhanced_ui.html`
   - Or drag the file into your browser

2. Take a screenshot showing:
   - Header with "Email Classification System"
   - Statistics cards (31 emails, 8 topics, etc.)
   - Email classification form
   - Factory Pattern generators panel

## Screenshot 3: Email Classification Test

1. In the Web UI, enter this email:
   ```
   Subject: Quarterly Budget Review Meeting
   Body: Please review the attached financial reports before our meeting tomorrow at 2pm.
   ```

2. Click "Classify Email"

3. Take a screenshot showing:
   - The input form with the email
   - Classification results
   - Confidence score bar chart
   - Predicted topic highlighted

## Screenshot 4: Feature Generators Working

1. After classifying an email, look at the console
2. Open Developer Tools (F12 or Cmd+Option+I)
3. Go to Network tab
4. Look at the response from /emails/classify
5. Take a screenshot showing the features including:
   - `non_text_non_text_char_count` (from your implemented generator)
   - All other feature values

## Screenshot 5: Dynamic Topic Addition

1. In the Web UI, scroll to "Topic Management"
2. Enter "finance" in the input field
3. Click "Add Topic"
4. Take a screenshot showing:
   - The topic being added
   - Success message
   - Updated topic list

## Screenshot 6: Terminal Test Results

1. Run the end-to-end test:
   ```bash
   python end_to_end_demo.py
   ```

2. Take a screenshot of the terminal showing:
   - All tests passing
   - Feature generators listed
   - Performance metrics
   - Final summary

## Screenshot 7: API Test with curl

1. Open a terminal
2. Run these commands and capture the output:

```bash
# Test /features endpoint
curl -s http://localhost:8000/features | python -m json.tool

# Test classification
curl -s -X POST http://localhost:8000/emails/classify \
  -H "Content-Type: application/json" \
  -d '{"subject": "Team Meeting", "body": "Discuss Q3 goals"}' \
  | python -m json.tool

# Show stored emails count
curl -s http://localhost:8000/emails | python -m json.tool | head -20
```

## Screenshot 8: Neo4j Graph (if connected)

1. If Neo4j is connected, open Neo4j Browser
2. Run: `MATCH (n) RETURN n LIMIT 50`
3. Take screenshot of the graph visualization

## Quick Test Commands

Run these in sequence for demonstration:

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. List topics
curl http://localhost:8000/topics

# 3. List features
curl http://localhost:8000/features

# 4. Add a new topic
curl -X POST http://localhost:8000/topics \
  -H "Content-Type: application/json" \
  -d '{"topic_name": "legal", "description": "Legal documents and contracts"}'

# 5. Classify an email
curl -X POST http://localhost:8000/emails/classify \
  -H "Content-Type: application/json" \
  -d '{"subject": "Contract Review", "body": "Please review the attached contract"}'

# 6. Store an email with label
curl -X POST http://localhost:8000/emails \
  -H "Content-Type: application/json" \
  -d '{"subject": "Contract", "body": "Legal review needed", "ground_truth": "legal"}'
```

## Saving Screenshots

On Mac:
- Full screen: Cmd + Shift + 3
- Selection: Cmd + Shift + 4
- Window: Cmd + Shift + 4, then Space

Save all screenshots with descriptive names:
- `01_swagger_api.png`
- `02_web_ui_dashboard.png`
- `03_classification_result.png`
- `04_feature_extraction.png`
- `05_topic_addition.png`
- `06_terminal_tests.png`
- `07_api_curl_tests.png`
- `08_neo4j_graph.png`

## Verification Checklist

Before submitting, verify:
- [ ] All 5 feature generators visible in /features
- [ ] NonTextCharacterFeatureGenerator counting special characters
- [ ] Dynamic topic addition working
- [ ] Email storage with ground truth functional
- [ ] Classification returning confidence scores
- [ ] Web UI displaying results correctly
- [ ] All API endpoints responding
- [ ] Test script showing all passes