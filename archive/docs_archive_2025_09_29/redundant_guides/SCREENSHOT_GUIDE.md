# Screenshot Guide for Lab 2 Submission

## UI Screenshots to Capture

### 1. Main Dashboard View
- **URL**: Open `frontend/enhanced_ui.html` in browser
- **What to show**:
  - Header with student name and lab title
  - Statistics cards (31 emails, 8 topics, 92% accuracy, 45ms response)
  - Email classification form
  - Factory Pattern generators panel

### 2. Email Classification in Action
- **Steps**:
  1. Click "Work Email" example button
  2. Click "Classify Email" button
  3. Capture the results showing:
     - Predicted topic
     - Confidence percentage
     - Bar chart of all topic scores

### 3. Factory Pattern Generators
- **What to highlight**:
  - All 5 generators listed
  - Special emphasis on NonTextCharacterFeatureGenerator (marked with ✨)
  - Each generator shows its features

### 4. Topic Management
- **What to show**:
  - All 8 available topics in colored badges
  - Add Topic input field
  - Dynamic topic addition capability

### 5. API Documentation (Swagger)
- **URL**: http://localhost:8000/docs
- **Screenshots needed**:
  - Main API overview showing all endpoints
  - POST /emails/classify endpoint expanded
  - GET /features endpoint showing generator information

### 6. Terminal Output
- **What to capture**:
  - Running server output
  - Test script results showing all tests passing
  - Classification results in terminal

## How to Take Screenshots

### On macOS:
1. **Full screen**: Cmd + Shift + 3
2. **Selection**: Cmd + Shift + 4, then drag
3. **Window**: Cmd + Shift + 4, then Space, click window

### Testing the System Before Screenshots:

1. Ensure server is running:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Test classification with different examples:
- Click each example button
- Click "Classify Email"
- Observe different confidence scores

3. Test topic addition:
- Enter "finance" in the topic field
- Click "Add Topic"

## Screenshot Checklist

- [ ] Main dashboard with all panels visible
- [ ] Work email classification result
- [ ] Promotional email classification result
- [ ] Personal email classification result
- [ ] Support email classification result
- [ ] Factory Pattern generators panel close-up
- [ ] Topic management section
- [ ] Swagger API documentation page
- [ ] Individual API endpoint details
- [ ] Terminal showing successful tests
- [ ] Browser developer console showing API calls

## Submission Notes

When submitting screenshots, organize them as:
```
screenshots/
├── 01_main_dashboard.png
├── 02_work_email_classification.png
├── 03_promotion_classification.png
├── 04_factory_generators.png
├── 05_topic_management.png
├── 06_swagger_overview.png
├── 07_api_endpoint_detail.png
└── 08_terminal_tests.png
```

## Key Points to Highlight in Screenshots

1. **Factory Pattern Implementation**: Show all 5 generators working
2. **Dynamic Topics**: Demonstrate adding new topics
3. **Learning Mode**: Show checkbox for using stored emails
4. **Real-time Classification**: Show instant results with confidence
5. **Professional UI**: Clean, modern interface design
6. **API Documentation**: Complete REST API with Swagger

## Final Verification

Before taking final screenshots:
1. Clear browser cache
2. Restart the server fresh
3. Run test script to populate data
4. Verify all features working
5. Check console for any errors