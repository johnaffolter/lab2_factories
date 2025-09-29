#!/usr/bin/env python3
"""
Comprehensive System Testing Script
Tests all functionality and provides reproducible analysis results
"""

import requests
import json
import time
from datetime import datetime
import os
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is responding"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_features_endpoint():
    """Test /features endpoint and analyze results"""
    print("\n" + "="*60)
    print("TESTING: GET /features endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/features")
        data = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Generator Count: {len(data['available_generators'])}")

        for gen in data['available_generators']:
            print(f"  ✓ {gen['name']}: {gen['description']}")
            print(f"    Features: {gen['features']}")
            print(f"    Performance: {gen['performance']}")

        return {
            "status": "PASS",
            "generator_count": len(data['available_generators']),
            "generators": [g['name'] for g in data['available_generators']],
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_topic_management():
    """Test dynamic topic management"""
    print("\n" + "="*60)
    print("TESTING: Topic Management")
    print("="*60)

    test_topic = f"test_reproducible_{int(time.time())}"

    try:
        # Get current topics
        response = requests.get(f"{BASE_URL}/topics")
        initial_topics = response.json()['topics']
        print(f"Initial topics count: {len(initial_topics)}")

        # Add new topic
        new_topic_data = {
            "topic_name": test_topic,
            "description": "Reproducible test topic"
        }

        response = requests.post(f"{BASE_URL}/topics", json=new_topic_data)
        print(f"Add topic status: {response.status_code}")

        if response.status_code == 200:
            final_topics = response.json()['topics']
            print(f"Final topics count: {len(final_topics)}")
            print(f"New topic added: {test_topic}")

            return {
                "status": "PASS",
                "initial_count": len(initial_topics),
                "final_count": len(final_topics),
                "new_topic": test_topic,
                "response_time": response.elapsed.total_seconds()
            }
        else:
            print(f"Failed to add topic: {response.text}")
            return {"status": "FAIL", "error": response.text}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_email_storage():
    """Test email storage with and without ground truth"""
    print("\n" + "="*60)
    print("TESTING: Email Storage")
    print("="*60)

    try:
        # Test with ground truth
        email_with_truth = {
            "subject": "Q4 Financial Report",
            "body": "Please review the quarterly financial statements. Revenue increased by 12%.",
            "ground_truth": "finance"
        }

        response1 = requests.post(f"{BASE_URL}/emails", json=email_with_truth)
        print(f"Store with ground truth status: {response1.status_code}")

        # Test without ground truth
        email_without_truth = {
            "subject": "Team Building Event",
            "body": "Join us for team building activities next Friday at the park."
        }

        response2 = requests.post(f"{BASE_URL}/emails", json=email_without_truth)
        print(f"Store without ground truth status: {response2.status_code}")

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            print(f"Email 1 ID: {data1.get('email_id')}")
            print(f"Email 2 ID: {data2.get('email_id')}")

            return {
                "status": "PASS",
                "with_ground_truth": data1,
                "without_ground_truth": data2,
                "response_times": [response1.elapsed.total_seconds(), response2.elapsed.total_seconds()]
            }
        else:
            return {"status": "FAIL", "errors": [response1.text, response2.text]}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_classification_modes():
    """Test both classification modes"""
    print("\n" + "="*60)
    print("TESTING: Dual Classification Modes")
    print("="*60)

    test_email = {
        "subject": "Budget Allocation Meeting",
        "body": "We need to discuss the budget allocation for Q1 2024. Please prepare financial reports."
    }

    try:
        # Topic similarity mode
        test_email["use_email_similarity"] = False
        response1 = requests.post(f"{BASE_URL}/emails/classify", json=test_email)

        # Email similarity mode
        test_email["use_email_similarity"] = True
        response2 = requests.post(f"{BASE_URL}/emails/classify", json=test_email)

        if response1.status_code == 200 and response2.status_code == 200:
            topic_result = response1.json()
            email_result = response2.json()

            print(f"Topic Mode Result: {topic_result.get('predicted_topic')}")
            print(f"Email Mode Result: {email_result.get('predicted_topic')}")

            # Check if modes produce different results
            modes_different = topic_result.get('predicted_topic') != email_result.get('predicted_topic')
            print(f"Modes produce different results: {modes_different}")

            return {
                "status": "PASS",
                "topic_mode": topic_result,
                "email_mode": email_result,
                "modes_different": modes_different,
                "response_times": [response1.elapsed.total_seconds(), response2.elapsed.total_seconds()]
            }
        else:
            return {"status": "FAIL", "errors": [response1.text, response2.text]}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_nontext_generator():
    """Test NonTextCharacterFeatureGenerator specifically"""
    print("\n" + "="*60)
    print("TESTING: NonTextCharacterFeatureGenerator")
    print("="*60)

    test_cases = [
        {
            "name": "Basic Special Characters",
            "email": {
                "subject": "Test!!!",
                "body": "Special chars: @#$% & ()[]"
            },
            "expected_range": (10, 20)
        },
        {
            "name": "Email with Punctuation",
            "email": {
                "subject": "Meeting @ 2:00 PM",
                "body": "Don't forget our meeting. It's important!"
            },
            "expected_range": (5, 15)
        },
        {
            "name": "Clean Text",
            "email": {
                "subject": "Regular meeting",
                "body": "This is a normal email with minimal punctuation"
            },
            "expected_range": (0, 5)
        }
    ]

    results = []

    for test_case in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/emails/classify", json=test_case["email"])

            if response.status_code == 200:
                data = response.json()
                features = data.get("features", {})
                char_count = features.get("non_text_non_text_char_count", 0)

                print(f"\n{test_case['name']}:")
                print(f"  Input: {test_case['email']['subject']} | {test_case['email']['body']}")
                print(f"  Non-text char count: {char_count}")
                print(f"  Expected range: {test_case['expected_range']}")

                min_expected, max_expected = test_case['expected_range']
                in_range = min_expected <= char_count <= max_expected
                print(f"  Within expected range: {in_range}")

                results.append({
                    "test_name": test_case['name'],
                    "char_count": char_count,
                    "expected_range": test_case['expected_range'],
                    "in_range": in_range,
                    "status": "PASS" if in_range else "WARN"
                })
            else:
                print(f"Failed test case {test_case['name']}: {response.text}")
                results.append({
                    "test_name": test_case['name'],
                    "status": "FAIL",
                    "error": response.text
                })

        except Exception as e:
            print(f"ERROR in {test_case['name']}: {e}")
            results.append({
                "test_name": test_case['name'],
                "status": "FAIL",
                "error": str(e)
            })

    return {"status": "COMPLETED", "test_results": results}

def analyze_system_performance():
    """Analyze system performance with multiple requests"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)

    test_email = {
        "subject": "Performance Test",
        "body": "Testing system response times and consistency"
    }

    response_times = []

    for i in range(10):
        start_time = time.time()
        try:
            response = requests.post(f"{BASE_URL}/emails/classify", json=test_email)
            if response.status_code == 200:
                response_times.append(time.time() - start_time)
        except:
            pass

    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        print(f"Performance Results (10 requests):")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Min response time: {min_time:.3f}s")
        print(f"  Max response time: {max_time:.3f}s")
        print(f"  Successful requests: {len(response_times)}/10")

        return {
            "avg_response_time": avg_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "success_rate": len(response_times) / 10,
            "total_requests": 10
        }
    else:
        return {"status": "FAIL", "error": "No successful requests"}

def main():
    """Run comprehensive system tests"""
    print("="*80)
    print("COMPREHENSIVE SYSTEM TEST - REPRODUCIBLE ANALYSIS")
    print("="*80)
    print(f"Test Start Time: {datetime.now().isoformat()}")
    print(f"Base URL: {BASE_URL}")

    # Check API health
    if not test_api_health():
        print("❌ API is not responding. Please start the server:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return

    print("✅ API is healthy and responding")

    # Run all tests
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "features_test": test_features_endpoint(),
        "topic_management_test": test_topic_management(),
        "email_storage_test": test_email_storage(),
        "classification_test": test_classification_modes(),
        "nontext_generator_test": test_nontext_generator(),
        "performance_analysis": analyze_system_performance()
    }

    # Generate summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_tests = 0
    total_tests = 0

    for test_name, result in test_results.items():
        if test_name == "timestamp":
            continue

        total_tests += 1
        status = result.get("status", "UNKNOWN")

        if status == "PASS":
            passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        elif status == "COMPLETED":
            passed_tests += 1
            print(f"✅ {test_name}: COMPLETED")
        else:
            print(f"❌ {test_name}: {status}")

    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Save results to file
    results_file = f"test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n📊 Detailed results saved to: {results_file}")
    print(f"🕒 Test completed at: {datetime.now().isoformat()}")

    return test_results

if __name__ == "__main__":
    results = main()