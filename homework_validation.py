#!/usr/bin/env python3
"""
Lab 2 Homework Validation Script
Tests all required homework components
"""

import requests
import json
from typing import Dict, Any

class HomeworkValidator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}

    def test_features_endpoint(self) -> bool:
        """Test that features endpoint returns available generators including NonTextCharacterFeatureGenerator"""
        try:
            response = requests.get(f"{self.base_url}/features")
            if response.status_code != 200:
                self.results["features_endpoint"] = f"Failed: Status {response.status_code}"
                return False

            data = response.json()
            generators = data.get("available_generators", [])

            # Check for NonTextCharacterFeatureGenerator
            non_text_found = False
            for gen in generators:
                if gen.get("name") == "non_text":
                    non_text_found = True
                    if "non_text_char_count" in gen.get("features", []):
                        self.results["features_endpoint"] = "✓ PASSED: NonTextCharacterFeatureGenerator found"
                        return True
                    else:
                        self.results["features_endpoint"] = "Failed: non_text_char_count feature missing"
                        return False

            self.results["features_endpoint"] = "Failed: NonTextCharacterFeatureGenerator not found"
            return False

        except Exception as e:
            self.results["features_endpoint"] = f"Failed: {e}"
            return False

    def test_topic_management(self) -> bool:
        """Test dynamic topic management endpoint"""
        try:
            # Add a test topic
            response = requests.post(
                f"{self.base_url}/topics",
                json={"topic_name": "homework_test_topic", "description": "Test topic for homework validation"}
            )

            if response.status_code != 200:
                self.results["topic_management"] = f"Failed: Status {response.status_code}"
                return False

            data = response.json()
            if "homework_test_topic" in data.get("topics", []):
                self.results["topic_management"] = "✓ PASSED: Topic added successfully"
                return True
            else:
                self.results["topic_management"] = "Failed: Topic not found in response"
                return False

        except Exception as e:
            self.results["topic_management"] = f"Failed: {e}"
            return False

    def test_email_storage(self) -> bool:
        """Test email storage with ground truth"""
        try:
            response = requests.post(
                f"{self.base_url}/emails",
                json={
                    "subject": "Homework Validation Email",
                    "body": "Testing email storage with ground truth labeling @#$%^&*()",
                    "ground_truth": "homework_test_topic"
                }
            )

            if response.status_code != 200:
                self.results["email_storage"] = f"Failed: Status {response.status_code}"
                return False

            data = response.json()
            if "successfully" in data.get("message", "").lower():
                self.results["email_storage"] = "✓ PASSED: Email stored with ground truth"
                return True
            else:
                self.results["email_storage"] = "Failed: Email storage message unclear"
                return False

        except Exception as e:
            self.results["email_storage"] = f"Failed: {e}"
            return False

    def test_classification_modes(self) -> bool:
        """Test both topic similarity and email similarity classification modes"""
        try:
            test_email = {
                "subject": "Important Meeting Tomorrow",
                "body": "Please attend the quarterly review meeting at 2 PM. Agenda attached."
            }

            # Test topic mode
            topic_response = requests.post(
                f"{self.base_url}/emails/classify",
                json={**test_email, "mode": "topic"}
            )

            # Test email mode
            email_response = requests.post(
                f"{self.base_url}/emails/classify",
                json={**test_email, "mode": "email"}
            )

            if topic_response.status_code != 200 or email_response.status_code != 200:
                self.results["classification_modes"] = f"Failed: Status codes {topic_response.status_code}, {email_response.status_code}"
                return False

            topic_data = topic_response.json()
            email_data = email_response.json()

            # Check that both return predictions
            if "predicted_topic" in topic_data and "predicted_topic" in email_data:
                self.results["classification_modes"] = "✓ PASSED: Both classification modes working"
                return True
            else:
                self.results["classification_modes"] = "Failed: Missing predicted_topic in response"
                return False

        except Exception as e:
            self.results["classification_modes"] = f"Failed: {e}"
            return False

    def test_non_text_feature_generator(self) -> bool:
        """Test NonTextCharacterFeatureGenerator specifically"""
        try:
            # Email with known non-text characters
            test_email = {
                "subject": "Test@Email!",  # @ and ! = 2 chars
                "body": "Special chars: #$%^&*()_+-={}[]|\\:;\"'<>?,./"  # 24 chars
                # Total should be 26 non-text characters
            }

            response = requests.post(
                f"{self.base_url}/emails/classify",
                json={**test_email, "mode": "topic"}
            )

            if response.status_code != 200:
                self.results["non_text_generator"] = f"Failed: Status {response.status_code}"
                return False

            data = response.json()
            features = data.get("features", {})
            non_text_count = features.get("non_text_non_text_char_count")

            if non_text_count is not None:
                # Verify the count is reasonable (should be around 26)
                if 20 <= non_text_count <= 30:  # Allow some tolerance
                    self.results["non_text_generator"] = f"✓ PASSED: NonTextCharacterFeatureGenerator working (count: {non_text_count})"
                    return True
                else:
                    self.results["non_text_generator"] = f"Failed: Unexpected count {non_text_count} (expected ~26)"
                    return False
            else:
                self.results["non_text_generator"] = "Failed: non_text_char_count feature not found"
                return False

        except Exception as e:
            self.results["non_text_generator"] = f"Failed: {e}"
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all homework validation tests"""
        print("🔬 LAB 2 HOMEWORK VALIDATION")
        print("=" * 50)

        tests = [
            ("Features Endpoint", self.test_features_endpoint),
            ("Topic Management", self.test_topic_management),
            ("Email Storage", self.test_email_storage),
            ("Classification Modes", self.test_classification_modes),
            ("NonTextCharacterFeatureGenerator", self.test_non_text_feature_generator),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\nTesting {test_name}...")
            success = test_func()
            if success:
                passed += 1
            print(f"  {self.results.get(test_name, 'Unknown result')}")

        print("\n" + "=" * 50)
        print(f"HOMEWORK VALIDATION SUMMARY")
        print(f"Passed: {passed}/{total} tests")
        print(f"Score: {(passed/total)*100:.1f}%")

        if passed == total:
            print("🎉 ALL HOMEWORK REQUIREMENTS IMPLEMENTED SUCCESSFULLY!")
        else:
            print("⚠️  Some requirements need attention")

        return {
            "passed": passed,
            "total": total,
            "score": (passed/total)*100,
            "details": self.results
        }

if __name__ == "__main__":
    validator = HomeworkValidator()
    results = validator.run_all_tests()

    # Save results for reference
    with open("homework_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to homework_validation_results.json")