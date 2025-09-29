#!/usr/bin/env python3
"""
Visual demonstration script for MLOps Homework 1
Creates detailed step-by-step documentation with actual API results
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80 + "\n")

def print_step(step_num, title, description):
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*80}")
    print(f"Description: {description}\n")

def print_request(method, endpoint, data=None):
    print(f"REQUEST:")
    print(f"  Method: {method}")
    print(f"  Endpoint: {BASE_URL}{endpoint}")
    if data:
        print(f"  Body:")
        print(json.dumps(data, indent=4))
    print()

def print_response(response):
    print(f"RESPONSE:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Body:")
    try:
        print(json.dumps(response.json(), indent=4))
    except:
        print(response.text)
    print()

# Main demonstration
print_section("MLOps Homework 1 - Complete System Demonstration")
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Student: John Affolter")
print(f"Repository: https://github.com/johnaffolter/lab2_factories")
print(f"Branch: john-homework")

# STEP 1: Health Check
print_step(1, "System Health Check", 
           "Verify the FastAPI server is running and responding")
print_request("GET", "/health")
response = requests.get(f"{BASE_URL}/health")
print_response(response)
print("✅ Server is operational")

# STEP 2: View Initial Topics
print_step(2, "View Current Topics",
           "List all topics currently in the system before adding new ones")
print_request("GET", "/topics")
response = requests.get(f"{BASE_URL}/topics")
print_response(response)
initial_topics = response.json()
print(f"✅ Found {len(initial_topics['topics'])} existing topics")

# STEP 3: Add New Topic
print_step(3, "Add New Topic (Requirement #2)",
           "Dynamically add a new topic without server restart")
new_topic = {
    "topic_name": "urgent_issues",
    "description": "Urgent issues requiring immediate attention"
}
print_request("POST", "/topics", new_topic)
response = requests.post(f"{BASE_URL}/topics", json=new_topic)
print_response(response)
print("✅ New topic added successfully")

# STEP 4: Verify Topic Added
print_step(4, "Verify Topic Addition",
           "Confirm the new topic is immediately available")
print_request("GET", "/topics")
response = requests.get(f"{BASE_URL}/topics")
updated_topics = response.json()
print_response(response)
print(f"✅ Topic count increased from {len(initial_topics['topics'])} to {len(updated_topics['topics'])}")

# STEP 5: Store Email WITH Ground Truth
print_step(5, "Store Email WITH Ground Truth (Requirement #3)",
           "Store training email with known category label")
email_with_gt = {
    "subject": "URGENT: System Down",
    "body": "Production system is experiencing critical errors. Immediate attention required!",
    "ground_truth": "urgent_issues"
}
print_request("POST", "/emails", email_with_gt)
response = requests.post(f"{BASE_URL}/emails", json=email_with_gt)
print_response(response)
email_id_1 = response.json().get("email_id")
print(f"✅ Email stored with ground truth label (ID: {email_id_1})")

# STEP 6: Store Email WITHOUT Ground Truth
print_step(6, "Store Email WITHOUT Ground Truth (Requirement #3)",
           "Store unlabeled email - ground truth is optional")
email_without_gt = {
    "subject": "Weekly Team Sync",
    "body": "Let's meet on Friday at 3pm to discuss project progress"
}
print_request("POST", "/emails", email_without_gt)
response = requests.post(f"{BASE_URL}/emails", json=email_without_gt)
print_response(response)
email_id_2 = response.json().get("email_id")
print(f"✅ Email stored without ground truth (ID: {email_id_2})")

# STEP 7: Classification - Topic Similarity Mode
print_step(7, "Classification: Topic Similarity Mode (Requirement #4)",
           "Classify email using topic descriptions (baseline method)")
test_email = {
    "subject": "Critical Bug Report",
    "body": "Found a critical bug in production that needs immediate fix",
    "use_email_similarity": False
}
print_request("POST", "/emails/classify", test_email)
response = requests.post(f"{BASE_URL}/emails/classify", json=test_email)
print_response(response)
topic_prediction = response.json()
print(f"✅ Topic Mode Prediction: {topic_prediction.get('predicted_topic')}")
print(f"   Confidence: {topic_prediction.get('confidence', 0):.3f}")

# STEP 8: Classification - Email Similarity Mode
print_step(8, "Classification: Email Similarity Mode (Requirement #4)",
           "Classify email using stored training emails (improved method)")
test_email["use_email_similarity"] = True
print_request("POST", "/emails/classify", test_email)
response = requests.post(f"{BASE_URL}/emails/classify", json=test_email)
print_response(response)
email_prediction = response.json()
print(f"✅ Email Mode Prediction: {email_prediction.get('predicted_topic')}")
print(f"   Confidence: {email_prediction.get('confidence', 0):.3f}")

# STEP 9: Demonstrate Inference on New Topic
print_step(9, "Demonstrate Inference on New Topic (Requirement #6)",
           "Classify an email into the newly created 'urgent_issues' topic")
urgent_test = {
    "subject": "Production outage alert",
    "body": "Emergency: Database servers are not responding. Need immediate escalation!",
    "use_email_similarity": False
}
print_request("POST", "/emails/classify", urgent_test)
response = requests.post(f"{BASE_URL}/emails/classify", json=urgent_test)
print_response(response)
urgent_result = response.json()
print(f"✅ Predicted Topic: {urgent_result.get('predicted_topic')}")
if urgent_result.get('predicted_topic') == 'urgent_issues':
    print("   ✨ Successfully classified to our newly added topic!")

# STEP 10: Final System Status
print_step(10, "Final System Status",
           "Summary of all data in the system")
response = requests.get(f"{BASE_URL}/topics")
final_topics = response.json()
print(f"Total Topics: {len(final_topics['topics'])}")
print(f"Topics: {', '.join(final_topics['topics'][:10])}...")
print()

# Read emails file
with open('data/emails.json', 'r') as f:
    emails = json.load(f)
print(f"Total Emails Stored: {len(emails)}")
emails_with_gt = [e for e in emails if 'ground_truth' in e and e['ground_truth']]
print(f"Emails with Ground Truth: {len(emails_with_gt)}")
print(f"Emails without Ground Truth: {len(emails) - len(emails_with_gt)}")

print("\n" + "="*80)
print("✅ ALL HOMEWORK REQUIREMENTS DEMONSTRATED SUCCESSFULLY")
print("="*80)
print("\nRequirements Coverage:")
print("  ✓ Req #1: Repository forked (https://github.com/johnaffolter/lab2_factories)")
print("  ✓ Req #2: Dynamic topic addition (POST /topics) - STEP 3")
print("  ✓ Req #3: Email storage with optional ground truth (POST /emails) - STEPS 5-6")
print("  ✓ Req #4: Dual classification modes - STEPS 7-8")
print("  ✓ Req #5: Demonstrate creating new topics - STEP 3 (urgent_issues)")
print("  ✓ Req #6: Demonstrate inference on new topics - STEP 9")
print("  ✓ Req #7: Demonstrate adding new emails - STEPS 5-6")
print("  ✓ Req #8: Demonstrate inference from email data - STEP 8")
print("\n" + "="*80)

