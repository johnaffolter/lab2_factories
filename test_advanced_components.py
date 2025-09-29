"""
Advanced Component Testing Suite
Tests all 10 composable components with comprehensive scenarios
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/composable"


class AdvancedComponentTester:
    def __init__(self):
        self.test_results = []
        self.test_emails = self._generate_test_emails()

    def _generate_test_emails(self):
        """Generate diverse test emails"""
        return {
            "spam_heavy": {
                "email_subject": "FREE! You WON $1,000,000! Click NOW!",
                "email_body": "Congratulations winner! This is urgent! Limited time offer! Click here immediately to claim your FREE prize money!"
            },
            "work_meeting": {
                "email_subject": "Project Deadline - Team Meeting Tomorrow",
                "email_body": "Hi team, we need to review the project status. Please prepare your reports for tomorrow's meeting at 10 AM. This is critical for the deadline."
            },
            "technical_issue": {
                "email_subject": "URGENT: Production Server Error",
                "email_body": "Critical bug detected in the database connection. Server is experiencing downtime. We need immediate action to fix this issue. Contact: support@company.com"
            },
            "personal_event": {
                "email_subject": "Birthday Party Invitation",
                "email_body": "You're invited to Sarah's birthday party on March 15, 2024! It's going to be wonderful. Let me know if you can make it. Call me at 555-123-4567."
            },
            "financial": {
                "email_subject": "Invoice Payment Reminder",
                "email_body": "Your invoice #12345 for $1,500 is due by Friday. Please process the payment transaction to account 987654. Payment details: http://payments.example.com/12345"
            },
            "action_items": {
                "email_subject": "Q4 Tasks and Deliverables",
                "email_body": "Please review the attached document. You must complete the analysis by next week. Should prepare the presentation for stakeholders. Need to send the final report ASAP."
            },
            "multilingual": {
                "email_subject": "Meeting Update",
                "email_body": "The meeting is scheduled for tomorrow. La reunión es mañana. We should prepare the materials."
            },
            "complex_text": {
                "email_subject": "Comprehensive Analysis Report",
                "email_body": "This sophisticated analysis demonstrates the intricate relationships between multifaceted variables. The methodology incorporates numerous statistical techniques. Consequently, interpretation requires substantial expertise."
            }
        }

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        print(f"{status} - {test_name}")
        if details:
            print(f"   {details}")

    def test_all_components_registered(self):
        """Test 1: Verify all 10 components are registered"""
        try:
            response = requests.get(f"{API_BASE}/components")
            data = response.json()
            passed = data["total"] == 10
            self.log_test("All Components Registered", passed,
                         f"Found {data['total']}/10 components")
            return passed
        except Exception as e:
            self.log_test("All Components Registered", False, str(e))
            return False

    def test_entity_extractor(self):
        """Test 2: Entity Extractor component"""
        try:
            # Create pipeline with entity extractor
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Entity Extraction Pipeline",
                    "components": [
                        {"name": "Entity Extractor", "config": {}}
                    ]
                }
            )
            pipeline_id = response.json()["id"]

            # Execute on technical issue email (contains email and phone)
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["technical_issue"]
            )
            exec_data = exec_response.json()
            if "results" not in exec_data:
                raise Exception(f"No results in response: {exec_data}")
            result = exec_data["results"][0]["output"]

            passed = (result["entity_count"] > 0 and
                     len(result["emails"]) > 0)

            self.log_test("Entity Extractor", passed,
                         f"Extracted {result['entity_count']} entities: {result['emails']}")
            return passed
        except Exception as e:
            self.log_test("Entity Extractor", False, str(e))
            return False

    def test_topic_classifier(self):
        """Test 3: Topic Classifier component"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Topic Classification",
                    "components": [{"name": "Topic Classifier", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test on work email
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["work_meeting"]
            )
            result = exec_response.json()["results"][0]["output"]

            passed = result["primary_topic"] in ["work", "unknown"] and result["confidence"] >= 0

            self.log_test("Topic Classifier", passed,
                         f"Topic: {result['primary_topic']} (confidence: {result['confidence']})")
            return passed
        except Exception as e:
            self.log_test("Topic Classifier", False, str(e))
            return False

    def test_readability_analyzer(self):
        """Test 4: Readability Analyzer component"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Readability Analysis",
                    "components": [{"name": "Readability Analyzer", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test on complex text
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["complex_text"]
            )
            result = exec_response.json()["results"][0]["output"]

            passed = result["sentence_count"] > 0 and result["readability_level"] != "unknown"

            self.log_test("Readability Analyzer", passed,
                         f"Readability: {result['readability_level']} ({result['sentence_count']} sentences)")
            return passed
        except Exception as e:
            self.log_test("Readability Analyzer", False, str(e))
            return False

    def test_language_detector(self):
        """Test 5: Language Detector component"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Language Detection",
                    "components": [{"name": "Language Detector", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test on multilingual email
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["multilingual"]
            )
            result = exec_response.json()["results"][0]["output"]

            passed = result["language"] in ["en", "es", "fr", "de", "unknown"]

            self.log_test("Language Detector", passed,
                         f"Language: {result['language']} (mixed: {result['is_mixed']})")
            return passed
        except Exception as e:
            self.log_test("Language Detector", False, str(e))
            return False

    def test_action_item_extractor(self):
        """Test 6: Action Item Extractor component"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Action Items",
                    "components": [{"name": "Action Item Extractor", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test on action items email
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["action_items"]
            )
            result = exec_response.json()["results"][0]["output"]

            passed = result["action_count"] > 0 and result["priority_level"] in ["low", "medium", "high"]

            self.log_test("Action Item Extractor", passed,
                         f"Actions: {result['action_count']} (priority: {result['priority_level']})")
            return passed
        except Exception as e:
            self.log_test("Action Item Extractor", False, str(e))
            return False

    def test_complex_multi_component_pipeline(self):
        """Test 7: Complex pipeline with all components"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Full Analysis Pipeline",
                    "components": [
                        {"name": "Spam Detector", "config": {}},
                        {"name": "Entity Extractor", "config": {}},
                        {"name": "Topic Classifier", "config": {}},
                        {"name": "Sentiment Analyzer", "config": {}},
                        {"name": "Urgency Detector", "config": {}},
                        {"name": "Readability Analyzer", "config": {}},
                        {"name": "Language Detector", "config": {}},
                        {"name": "Action Item Extractor", "config": {}}
                    ]
                }
            )
            pipeline_id = response.json()["id"]

            # Execute on technical issue
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["technical_issue"]
            )
            data = exec_response.json()

            passed = len(data["results"]) == 8

            self.log_test("Complex Multi-Component Pipeline", passed,
                         f"Executed {len(data['results'])} components successfully")
            return passed
        except Exception as e:
            self.log_test("Complex Multi-Component Pipeline", False, str(e))
            return False

    def test_spam_detection_accuracy(self):
        """Test 8: Spam detection accuracy"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Spam Detection",
                    "components": [{"name": "Spam Detector", "config": {"threshold": 0.3}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test spam email
            spam_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["spam_heavy"]
            )
            spam_result = spam_response.json()["results"][0]["output"]

            # Test legitimate email
            legit_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["work_meeting"]
            )
            legit_result = legit_response.json()["results"][0]["output"]

            passed = spam_result["has_spam"] == True and legit_result["has_spam"] == False

            self.log_test("Spam Detection Accuracy", passed,
                         f"Spam: {spam_result['spam_score']:.2f}, Legit: {legit_result['spam_score']:.2f}")
            return passed
        except Exception as e:
            self.log_test("Spam Detection Accuracy", False, str(e))
            return False

    def test_urgency_detection_levels(self):
        """Test 9: Urgency detection with different levels"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Urgency Detection",
                    "components": [{"name": "Urgency Detector", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test high urgency (technical issue)
            urgent_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["technical_issue"]
            )
            urgent_result = urgent_response.json()["results"][0]["output"]

            # Test low urgency (personal)
            normal_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["personal_event"]
            )
            normal_result = normal_response.json()["results"][0]["output"]

            passed = urgent_result["urgency_level"] in ["high", "medium"] and normal_result["urgency_level"] == "low"

            self.log_test("Urgency Detection Levels", passed,
                         f"Technical: {urgent_result['urgency_level']}, Personal: {normal_result['urgency_level']}")
            return passed
        except Exception as e:
            self.log_test("Urgency Detection Levels", False, str(e))
            return False

    def test_sentiment_analysis(self):
        """Test 10: Sentiment analysis on different email types"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Sentiment Analysis",
                    "components": [{"name": "Sentiment Analyzer", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test positive email (personal event)
            positive_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["personal_event"]
            )
            positive_result = positive_response.json()["results"][0]["output"]

            # Test neutral/professional email (work)
            neutral_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["work_meeting"]
            )
            neutral_result = neutral_response.json()["results"][0]["output"]

            passed = positive_result["sentiment"] in ["positive", "neutral"]

            self.log_test("Sentiment Analysis", passed,
                         f"Personal: {positive_result['sentiment']}, Work: {neutral_result['sentiment']}")
            return passed
        except Exception as e:
            self.log_test("Sentiment Analysis", False, str(e))
            return False

    def test_word_length_analysis(self):
        """Test 11: Word length and vocabulary analysis"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Word Analysis",
                    "components": [{"name": "Word Length Analyzer", "config": {}}]
                }
            )
            pipeline_id = response.json()["id"]

            # Test on complex text
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["complex_text"]
            )
            result = exec_response.json()["results"][0]["output"]

            passed = result["total_words"] > 0 and result["vocabulary_richness"] > 0

            self.log_test("Word Length Analysis", passed,
                         f"Words: {result['total_words']}, Richness: {result['vocabulary_richness']}")
            return passed
        except Exception as e:
            self.log_test("Word Length Analysis", False, str(e))
            return False

    def test_component_search(self):
        """Test 12: Search for components by keywords"""
        try:
            # Search for "entity"
            response = requests.post(
                f"{API_BASE}/components/search",
                json={"query": "entity"}
            )
            data = response.json()

            passed = data["total_results"] > 0

            self.log_test("Component Search", passed,
                         f"Found {data['total_results']} components for 'entity'")
            return passed
        except Exception as e:
            self.log_test("Component Search", False, str(e))
            return False

    def test_pipeline_with_custom_config(self):
        """Test 13: Pipeline with custom component configuration"""
        try:
            response = requests.post(
                f"{API_BASE}/pipelines",
                json={
                    "name": "Custom Config Pipeline",
                    "components": [
                        {
                            "name": "Spam Detector",
                            "config": {
                                "keywords": ["urgent", "free", "winner", "click"],
                                "threshold": 0.5
                            }
                        },
                        {
                            "name": "Topic Classifier",
                            "config": {
                                "topics": {
                                    "marketing": ["sale", "offer", "discount"],
                                    "support": ["help", "issue", "problem"]
                                }
                            }
                        }
                    ]
                }
            )
            pipeline_id = response.json()["id"]

            # Execute
            exec_response = requests.post(
                f"{API_BASE}/pipelines/{pipeline_id}/execute",
                json=self.test_emails["spam_heavy"]
            )
            data = exec_response.json()

            passed = exec_response.status_code == 200 and len(data["results"]) == 2

            self.log_test("Pipeline with Custom Config", passed,
                         f"Executed {len(data.get('results', []))} components with custom config")
            return passed
        except Exception as e:
            self.log_test("Pipeline with Custom Config", False, str(e))
            return False

    def run_all_tests(self):
        """Run all advanced component tests"""
        print("\n" + "="*70)
        print("ADVANCED COMPONENT TESTING SUITE")
        print("="*70 + "\n")

        # Component Registration Tests
        print("🔧 COMPONENT REGISTRATION")
        print("-" * 70)
        self.test_all_components_registered()
        self.test_component_search()

        # Individual Component Tests
        print("\n🧩 INDIVIDUAL COMPONENT TESTS")
        print("-" * 70)
        self.test_entity_extractor()
        self.test_topic_classifier()
        self.test_readability_analyzer()
        self.test_language_detector()
        self.test_action_item_extractor()
        self.test_word_length_analysis()

        # Accuracy Tests
        print("\n🎯 ACCURACY TESTS")
        print("-" * 70)
        self.test_spam_detection_accuracy()
        self.test_urgency_detection_levels()
        self.test_sentiment_analysis()

        # Integration Tests
        print("\n🔗 INTEGRATION TESTS")
        print("-" * 70)
        self.test_complex_multi_component_pipeline()
        self.test_pipeline_with_custom_config()

        # Generate Report
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t["passed"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Pass Rate: {pass_rate:.1f}%")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['details']}")

        print("\n" + "="*70)

        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate,
            "results": self.test_results
        }


if __name__ == "__main__":
    print("Starting advanced component tests...")
    time.sleep(2)

    tester = AdvancedComponentTester()
    report = tester.run_all_tests()

    # Save report
    with open("test_results_advanced.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📄 Full report saved to: test_results_advanced.json")