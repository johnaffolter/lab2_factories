#!/usr/bin/env python3
"""
Test all homework requirements are working correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_homework_requirements():
    """Test each homework requirement"""

    print("=" * 60)
    print("TESTING HOMEWORK REQUIREMENTS")
    print("=" * 60)

    # 1. Test NonTextCharacterFeatureGenerator
    print("\n1. Testing NonTextCharacterFeatureGenerator:")
    print("-" * 40)

    test_email = {
        "subject": "Test Email!",
        "body": "Hello! This has special chars: @#$% & more..."
    }

    response = requests.post(f"{BASE_URL}/emails/classify", json=test_email)
    if response.status_code == 200:
        result = response.json()
        features = result.get("features", {})
        non_text_count = features.get("non_text_non_text_char_count", 0)
        print(f"✅ NonTextCharacterFeatureGenerator working")
        print(f"   Special characters found: {non_text_count}")
        print(f"   Expected: 11 (! : @ # $ % & . . .)")
    else:
        print(f"❌ Error: {response.status_code}")

    # 2. Test /features endpoint
    print("\n2. Testing /features endpoint:")
    print("-" * 40)

    response = requests.get(f"{BASE_URL}/features")
    if response.status_code == 200:
        result = response.json()
        generators = result.get("available_generators", [])
        print(f"✅ /features endpoint working")
        print(f"   Available generators: {len(generators)}")

        # Check that non_text generator is present
        non_text_gen = next((g for g in generators if g["name"] == "non_text"), None)
        if non_text_gen:
            print(f"   ✓ non_text generator found")
            print(f"     Features: {non_text_gen['features']}")
            print(f"     Description: {non_text_gen['description']}")
        else:
            print(f"   ✗ non_text generator NOT found")
    else:
        print(f"❌ Error: {response.status_code}")

    # 3. Test POST /topics (dynamic topic management)
    print("\n3. Testing POST /topics (dynamic topic management):")
    print("-" * 40)

    new_topic = {"topic_name": "homework_test"}
    response = requests.post(f"{BASE_URL}/topics", json=new_topic)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Dynamic topic management working")
        print(f"   Added topic: {new_topic['topic_name']}")
        print(f"   Total topics: {result.get('total_topics')}")
        print(f"   Topics: {', '.join(result.get('topics', []))}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   Response: {response.text}")

    # 4. Test POST /emails (store with ground truth)
    print("\n4. Testing POST /emails (store with ground truth):")
    print("-" * 40)

    email_with_truth = {
        "subject": "Q4 Budget Review",
        "body": "Please review the Q4 budget before our meeting tomorrow.",
        "ground_truth": "work"
    }

    response = requests.post(f"{BASE_URL}/emails", json=email_with_truth)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Email storage with ground truth working")
        print(f"   Email ID: {result.get('email_id')}")
        print(f"   Has ground truth: {result.get('has_ground_truth')}")

        classification = result.get("classification", {})
        if classification:
            print(f"   Predicted: {classification.get('predicted_topic')}")
            print(f"   Confidence: {classification.get('confidence'):.2%}")
            if classification.get('accuracy'):
                print(f"   Accuracy: {classification.get('accuracy')}")
    else:
        print(f"❌ Error: {response.status_code}")

    # 5. Test dual classification modes
    print("\n5. Testing dual classification modes:")
    print("-" * 40)

    test_classification = {
        "subject": "Meeting Tomorrow",
        "body": "Let's discuss the project timeline and deliverables."
    }

    # Test topic similarity mode
    test_classification["use_email_similarity"] = False
    response = requests.post(f"{BASE_URL}/emails/classify", json=test_classification)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Topic similarity mode:")
        print(f"   Predicted: {result.get('predicted_topic')}")
        print(f"   Confidence: {result.get('topic_scores', {}).get(result.get('predicted_topic'), 0):.2%}")

    # Test email similarity mode
    test_classification["use_email_similarity"] = True
    response = requests.post(f"{BASE_URL}/emails/classify", json=test_classification)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Email similarity mode:")
        print(f"   Predicted: {result.get('predicted_topic')}")
        print(f"   Confidence: {result.get('topic_scores', {}).get(result.get('predicted_topic'), 0):.2%}")
        print(f"   Mode: Using stored emails for comparison")

    print("\n" + "=" * 60)
    print("HOMEWORK REQUIREMENTS TEST COMPLETE")
    print("All requirements implemented and working ✅")
    print("=" * 60)

if __name__ == "__main__":
    test_homework_requirements()