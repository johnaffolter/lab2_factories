#!/usr/bin/env python3
"""
Test script to demonstrate all new functionality
"""
import requests
import json
import time

# API Base URL - update this for your environment
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def test_features_endpoint():
    """Test the /features endpoint"""
    print_section("Testing /features endpoint")

    response = requests.get(f"{BASE_URL}/features")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully retrieved {len(data['available_generators'])} feature generators:")
        for gen in data['available_generators']:
            print(f"  - {gen['name']}: {', '.join(gen['features'])}")
    else:
        print(f"✗ Failed to get features: {response.status_code}")
    return response.status_code == 200

def test_add_topics():
    """Test adding new custom topics"""
    print_section("Testing dynamic topic addition")

    new_topics = [
        {
            "topic_name": "travel",
            "description": "Travel bookings, itineraries, and vacation planning emails"
        },
        {
            "topic_name": "education",
            "description": "Educational content, course notifications, and learning materials"
        },
        {
            "topic_name": "health",
            "description": "Health updates, medical appointments, and wellness information"
        }
    ]

    for topic in new_topics:
        response = requests.post(f"{BASE_URL}/topics", json=topic)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Added topic '{topic['topic_name']}': {data['message']}")
        else:
            print(f"✗ Failed to add topic '{topic['topic_name']}': {response.text}")

    # Verify topics were added
    response = requests.get(f"{BASE_URL}/topics")
    if response.status_code == 200:
        topics = response.json()['topics']
        print(f"\n✓ Current topics: {', '.join(topics)}")
        return True
    return False

def test_store_emails():
    """Test storing emails with ground truth"""
    print_section("Testing email storage with ground truth")

    test_emails = [
        {
            "subject": "Flight Confirmation - NYC to Paris",
            "body": "Your flight booking is confirmed. Departure: May 15, 2024. Please arrive 2 hours early.",
            "ground_truth": "travel"
        },
        {
            "subject": "New Python Course Available",
            "body": "Enroll now in our advanced Python programming course. Limited seats available!",
            "ground_truth": "education"
        },
        {
            "subject": "Your Lab Results Are Ready",
            "body": "Your recent lab test results are now available in your patient portal.",
            "ground_truth": "health"
        },
        {
            "subject": "Team Meeting Tomorrow",
            "body": "Please join us for the quarterly review meeting at 2 PM in the conference room.",
            "ground_truth": "work"
        },
        {
            "subject": "50% Off Everything!",
            "body": "Don't miss our biggest sale of the year. Use code SAVE50 at checkout.",
            "ground_truth": "promotion"
        }
    ]

    stored_ids = []
    for email in test_emails:
        response = requests.post(f"{BASE_URL}/emails", json=email)
        if response.status_code == 200:
            data = response.json()
            stored_ids.append(data['email_id'])
            print(f"✓ Stored email #{data['email_id']}: '{email['subject']}' (truth: {email['ground_truth']})")
        else:
            print(f"✗ Failed to store email: {response.text}")

    # Get all stored emails
    response = requests.get(f"{BASE_URL}/emails")
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Total stored emails: {data['count']}")

    return len(stored_ids) > 0

def test_classification_modes():
    """Test classification with both topic and email similarity modes"""
    print_section("Testing classification modes")

    test_cases = [
        {
            "subject": "Booking Confirmation - Hotel in Rome",
            "body": "Your hotel reservation is confirmed for June 1-5, 2024.",
            "expected": "travel"
        },
        {
            "subject": "Learn Machine Learning Today",
            "body": "Start your journey in artificial intelligence with our comprehensive course.",
            "expected": "education"
        },
        {
            "subject": "Appointment Reminder",
            "body": "This is a reminder of your upcoming dental appointment on Monday at 10 AM.",
            "expected": "health"
        }
    ]

    for test_email in test_cases:
        print(f"\nClassifying: '{test_email['subject']}'")

        # Test with topic similarity (default)
        response = requests.post(f"{BASE_URL}/emails/classify", json={
            "subject": test_email["subject"],
            "body": test_email["body"],
            "use_email_similarity": False
        })

        if response.status_code == 200:
            data = response.json()
            print(f"  Topic similarity prediction: {data['predicted_topic']}")
            top_scores = sorted(data['topic_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3 scores: {', '.join([f'{t[0]}:{t[1]:.3f}' for t in top_scores])}")

        # Test with email similarity
        response = requests.post(f"{BASE_URL}/emails/classify", json={
            "subject": test_email["subject"],
            "body": test_email["body"],
            "use_email_similarity": True
        })

        if response.status_code == 200:
            data = response.json()
            print(f"  Email similarity prediction: {data['predicted_topic']}")
            print(f"  Expected: {test_email['expected']}")
            if data['predicted_topic'] == test_email['expected']:
                print("  ✓ Correct prediction with email similarity!")
            else:
                print("  ✗ Different prediction with email similarity")

def test_non_text_feature():
    """Test the NonTextCharacterFeatureGenerator"""
    print_section("Testing NonTextCharacterFeatureGenerator")

    test_emails = [
        {
            "subject": "Important!!! Act Now!!!",
            "body": "This is urgent$$$ Don't miss out!!!!"
        },
        {
            "subject": "Regular meeting update",
            "body": "The meeting is at 2pm tomorrow"
        }
    ]

    for email in test_emails:
        response = requests.post(f"{BASE_URL}/emails/classify", json={
            "subject": email["subject"],
            "body": email["body"]
        })

        if response.status_code == 200:
            data = response.json()
            non_text_count = data['features'].get('non_text_non_text_char_count', 0)
            print(f"Email: '{email['subject']}'")
            print(f"  Non-text character count: {non_text_count}")
            print(f"  Predicted topic: {data['predicted_topic']}")

def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*60)
    print(" Email Classification System - New Features Test Suite")
    print("="*60)

    # Run tests in sequence
    tests = [
        ("Feature Generators", test_features_endpoint),
        ("Dynamic Topics", test_add_topics),
        ("Email Storage", test_store_emails),
        ("Classification Modes", test_classification_modes),
        ("Non-Text Feature", test_non_text_feature)
    ]

    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✓ Passed" if success else "✗ Failed"
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results[name] = f"✗ Error: {e}"

    # Print summary
    print_section("Test Summary")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")

if __name__ == "__main__":
    print("Starting Email Classification Tests...")
    print(f"Testing API at: {BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  uvicorn app.main:app --reload")

    # Add a small delay to ensure server is ready
    time.sleep(2)

    run_all_tests()

    print("\n✅ All tests completed!")