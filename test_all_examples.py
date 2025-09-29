#!/usr/bin/env python3
"""
Comprehensive testing of all API endpoints with real examples
"""
import json
import requests

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {method} {endpoint}")

    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)

        if response.status_code == 200:
            print(f"Status: ✅ Success ({response.status_code})")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
            return result
        else:
            print(f"Status: ❌ Failed ({response.status_code})")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Status: ❌ Exception")
        print(f"Error: {str(e)}")
        return None

# Test all endpoints
print("\n" + "="*80)
print("COMPREHENSIVE API TESTING")
print("="*80)

# 1. Test GET /topics
test_endpoint("Get Available Topics", "GET", "/topics")

# 2. Test GET /features
test_endpoint("Get Feature Generators", "GET", "/features")

# 3. Test GET /pipeline/info
test_endpoint("Get Pipeline Info", "GET", "/pipeline/info")

# 4. Test various email classifications
email_examples = [
    {
        "name": "Work Email - Meeting",
        "data": {
            "subject": "Quarterly Business Review Meeting",
            "body": "Please join us for the Q3 business review meeting where we will discuss revenue targets and KPIs."
        }
    },
    {
        "name": "Promotional Email",
        "data": {
            "subject": "50 Percent Off Everything - Limited Time!",
            "body": "Huge savings on all items in our store. Use code SAVE50 at checkout. This offer expires in 24 hours!"
        }
    },
    {
        "name": "Personal Email",
        "data": {
            "subject": "Happy Birthday!",
            "body": "Hey! Hope you have an amazing birthday today. Let's catch up soon over coffee!"
        }
    },
    {
        "name": "Support Email",
        "data": {
            "subject": "Issue with my account",
            "body": "I'm having trouble logging into my account. Could you please help reset my password?"
        }
    },
    {
        "name": "Newsletter Email",
        "data": {
            "subject": "Your Weekly Tech News Digest",
            "body": "This week in tech: AI advances, new smartphone releases, and cybersecurity updates."
        }
    },
    {
        "name": "Travel Email",
        "data": {
            "subject": "Your Flight Confirmation",
            "body": "Your flight to New York is confirmed for December 15th. Check-in opens 24 hours before departure."
        }
    },
    {
        "name": "Education Email",
        "data": {
            "subject": "Course Registration Reminder",
            "body": "Spring semester registration opens next week. Review the course catalog and prepare your schedule."
        }
    },
    {
        "name": "Health Email",
        "data": {
            "subject": "Annual Checkup Reminder",
            "body": "It's time for your annual health checkup. Please call our office to schedule an appointment."
        }
    }
]

print("\n" + "="*80)
print("EMAIL CLASSIFICATION RESULTS")
print("="*80)

classification_results = []
for example in email_examples:
    result = test_endpoint(
        f"Classify: {example['name']}",
        "POST",
        "/emails/classify",
        example['data']
    )
    if result:
        classification_results.append({
            "email_type": example['name'],
            "predicted": result['predicted_topic'],
            "confidence": result['topic_scores'][result['predicted_topic']],
            "subject": example['data']['subject']
        })

# Summary
print("\n" + "="*80)
print("CLASSIFICATION SUMMARY")
print("="*80)
print(f"{'Email Type':<25} {'Predicted':<12} {'Confidence':<10} {'Subject':<40}")
print("-" * 87)
for result in classification_results:
    print(f"{result['email_type']:<25} {result['predicted']:<12} {result['confidence']:.2%} {result['subject'][:40]:<40}")

# Test adding new topics
print("\n" + "="*80)
print("TESTING DYNAMIC TOPIC ADDITION")
print("="*80)

new_topic = {
    "topic": "finance",
    "description": "Financial emails about banking, investments, stocks, trading",
    "keywords": ["bank", "investment", "stock", "trading", "portfolio", "finance"]
}

test_endpoint("Add Finance Topic", "POST", "/topics", new_topic)

# Store emails with labels for learning
print("\n" + "="*80)
print("TESTING EMAIL STORAGE WITH LABELS")
print("="*80)

training_emails = [
    {
        "subject": "Your Portfolio Performance Report",
        "body": "Your investment portfolio gained 5% this month. See detailed breakdown of your stocks and bonds.",
        "label": "finance"
    },
    {
        "subject": "Team Standup at 10am",
        "body": "Daily standup meeting to discuss sprint progress and blockers.",
        "label": "work"
    }
]

for email in training_emails:
    test_endpoint(
        f"Store Training Email: {email['label']}",
        "POST",
        "/emails",
        email
    )

# Test classification with learning mode
print("\n" + "="*80)
print("TESTING LEARNING-BASED CLASSIFICATION")
print("="*80)

test_email = {
    "subject": "Stock Market Update",
    "body": "The S&P 500 closed up 2% today. Your tech stocks performed particularly well.",
    "use_learning": True
}

result = test_endpoint(
    "Classify with Learning Mode",
    "POST",
    "/emails/classify",
    test_email
)

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)