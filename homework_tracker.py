#!/usr/bin/env python3
"""
MLOps Homework 1 - Complete Tracking and Visualization System
Tracks all API calls, stores results, and generates visualizations
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import os

BASE_URL = "http://localhost:8000"
TRACKING_DIR = "homework_tracking"
RESULTS_FILE = f"{TRACKING_DIR}/test_results.json"
TIMELINE_FILE = f"{TRACKING_DIR}/test_timeline.json"

@dataclass
class TestResult:
    """Individual test result"""
    test_number: int
    test_name: str
    requirement: str
    timestamp: str
    request_method: str
    request_url: str
    request_body: Dict[str, Any]
    response_status: int
    response_body: Dict[str, Any]
    duration_ms: float
    success: bool
    notes: str

class HomeworkTracker:
    """Track and visualize homework requirement demonstrations"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()

        # Create tracking directory
        os.makedirs(TRACKING_DIR, exist_ok=True)

    def run_test(self, test_number: int, test_name: str, requirement: str,
                 method: str, endpoint: str, body: Dict = None, notes: str = "") -> TestResult:
        """Run a single test and track results"""

        url = f"{BASE_URL}{endpoint}"
        start = time.time()

        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=body)
            else:
                raise ValueError(f"Unsupported method: {method}")

            duration = (time.time() - start) * 1000  # Convert to ms

            result = TestResult(
                test_number=test_number,
                test_name=test_name,
                requirement=requirement,
                timestamp=datetime.now().isoformat(),
                request_method=method,
                request_url=url,
                request_body=body or {},
                response_status=response.status_code,
                response_body=response.json(),
                duration_ms=duration,
                success=200 <= response.status_code < 300,
                notes=notes
            )

            self.results.append(result)
            self._save_results()

            return result

        except Exception as e:
            result = TestResult(
                test_number=test_number,
                test_name=test_name,
                requirement=requirement,
                timestamp=datetime.now().isoformat(),
                request_method=method,
                request_url=url,
                request_body=body or {},
                response_status=0,
                response_body={"error": str(e)},
                duration_ms=0,
                success=False,
                notes=f"Error: {str(e)}"
            )

            self.results.append(result)
            return result

    def _save_results(self):
        """Save results to JSON file"""
        with open(RESULTS_FILE, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

    def generate_timeline(self):
        """Generate timeline visualization data"""
        timeline = {
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "requirements_tested": list(set(r.requirement for r in self.results)),
            "events": []
        }

        for result in self.results:
            timeline["events"].append({
                "timestamp": result.timestamp,
                "test_number": result.test_number,
                "test_name": result.test_name,
                "requirement": result.requirement,
                "success": result.success,
                "duration_ms": result.duration_ms,
                "method": result.request_method,
                "endpoint": result.request_url.replace(BASE_URL, "")
            })

        with open(TIMELINE_FILE, 'w') as f:
            json.dump(timeline, f, indent=2)

        return timeline

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 HOMEWORK TRACKING SUMMARY")
        print("="*80)

        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        print(f"\n✅ Total Tests: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(successful/total*100):.1f}%")

        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / total if total > 0 else 0
        print(f"⏱️  Total Duration: {total_duration:.2f}ms")
        print(f"⏱️  Average Duration: {avg_duration:.2f}ms")

        print("\n📋 Requirements Tested:")
        requirements = {}
        for result in self.results:
            if result.requirement not in requirements:
                requirements[result.requirement] = {"total": 0, "success": 0}
            requirements[result.requirement]["total"] += 1
            if result.success:
                requirements[result.requirement]["success"] += 1

        for req, stats in requirements.items():
            status = "✅" if stats["success"] == stats["total"] else "⚠️"
            print(f"  {status} {req}: {stats['success']}/{stats['total']} passed")

        print(f"\n📁 Results saved to: {RESULTS_FILE}")
        print(f"📁 Timeline saved to: {TIMELINE_FILE}")
        print("="*80)

def run_complete_homework_tracking():
    """Run complete homework demonstration with tracking"""

    tracker = HomeworkTracker()

    print("\n" + "="*80)
    print("🎓 MLOps HOMEWORK 1 - COMPLETE TRACKING SESSION")
    print("="*80)
    print(f"Start Time: {tracker.start_time.isoformat()}")
    print(f"Tracking Directory: {TRACKING_DIR}")
    print("="*80)

    # TEST 1: Health Check
    print("\n[TEST 1/10] Health Check...")
    result = tracker.run_test(
        test_number=1,
        test_name="System Health Check",
        requirement="Prerequisite",
        method="GET",
        endpoint="/health",
        notes="Verify server is running and responding"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    print(f"  Response: {result.response_body}")

    # TEST 2: List Initial Topics
    print("\n[TEST 2/10] List Initial Topics...")
    result = tracker.run_test(
        test_number=2,
        test_name="List Initial Topics",
        requirement="Baseline",
        method="GET",
        endpoint="/topics",
        notes="Get baseline topic count"
    )
    initial_topics_count = len(result.response_body.get("topics", []))
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    print(f"  Initial Topics: {initial_topics_count}")

    # TEST 3: Add New Topic
    print("\n[TEST 3/10] Add New Topic (Requirement #2)...")
    result = tracker.run_test(
        test_number=3,
        test_name="Add New Topic",
        requirement="Requirement #2: Dynamic Topic Addition",
        method="POST",
        endpoint="/topics",
        body={
            "topic_name": "homework_demo_complete",
            "description": "Complete homework demonstration topic"
        },
        notes="Test dynamic topic addition without server restart"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    new_topics_count = len(result.response_body.get("topics", []))
    print(f"  Topics increased: {initial_topics_count} → {new_topics_count}")

    # TEST 4: Verify Topic Persistence
    print("\n[TEST 4/10] Verify Topic Immediately Available...")
    result = tracker.run_test(
        test_number=4,
        test_name="Verify Topic Persistence",
        requirement="Requirement #2: Topic Persistence",
        method="GET",
        endpoint="/topics",
        notes="Confirm new topic persisted without restart"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    print(f"  Topic 'homework_demo_complete' present: {'homework_demo_complete' in result.response_body.get('topics', [])}")

    # TEST 5: Store Email WITH Ground Truth
    print("\n[TEST 5/10] Store Email WITH Ground Truth (Requirement #3)...")
    result = tracker.run_test(
        test_number=5,
        test_name="Store Labeled Email",
        requirement="Requirement #3: Email Storage with Ground Truth",
        method="POST",
        endpoint="/emails",
        body={
            "subject": "Q4 Financial Results",
            "body": "Revenue: $2.5M, Profit: $450K, Growth: 18% YoY",
            "ground_truth": "finance"
        },
        notes="Store training email with known category"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    email_id_1 = result.response_body.get("email_id")
    print(f"  Email ID: {email_id_1} | Total Emails: {result.response_body.get('total_emails')}")

    # TEST 6: Store Email WITHOUT Ground Truth
    print("\n[TEST 6/10] Store Email WITHOUT Ground Truth (Requirement #3)...")
    result = tracker.run_test(
        test_number=6,
        test_name="Store Unlabeled Email",
        requirement="Requirement #3: Optional Ground Truth",
        method="POST",
        endpoint="/emails",
        body={
            "subject": "Office Party Next Week",
            "body": "Join us for pizza and games in the break room"
        },
        notes="Demonstrate ground truth is optional"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    email_id_2 = result.response_body.get("email_id")
    print(f"  Email ID: {email_id_2} | Total Emails: {result.response_body.get('total_emails')}")

    # TEST 7: Classify - Topic Mode
    print("\n[TEST 7/10] Classify Email - Topic Similarity Mode (Requirement #4)...")
    result = tracker.run_test(
        test_number=7,
        test_name="Topic Similarity Classification",
        requirement="Requirement #4: Dual Mode #1 (Topic Similarity)",
        method="POST",
        endpoint="/emails/classify",
        body={
            "subject": "Urgent: Server Outage",
            "body": "Production database is down. Need immediate action!",
            "use_email_similarity": False
        },
        notes="Baseline classification using topic descriptions"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    predicted = result.response_body.get("predicted_topic")
    print(f"  Predicted Topic: {predicted}")

    # TEST 8: Classify - Email Mode
    print("\n[TEST 8/10] Classify Email - Email Similarity Mode (Requirement #4)...")
    result = tracker.run_test(
        test_number=8,
        test_name="Email Similarity Classification",
        requirement="Requirement #4: Dual Mode #2 (Email Similarity)",
        method="POST",
        endpoint="/emails/classify",
        body={
            "subject": "Urgent: Server Outage",
            "body": "Production database is down. Need immediate action!",
            "use_email_similarity": True
        },
        notes="Improved classification using stored emails"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    predicted = result.response_body.get("predicted_topic")
    print(f"  Predicted Topic: {predicted}")

    # TEST 9: Classify into New Topic
    print("\n[TEST 9/10] Demonstrate Inference on New Topic (Requirement #6)...")
    result = tracker.run_test(
        test_number=9,
        test_name="New Topic Inference",
        requirement="Requirement #6: Inference on New Topics",
        method="POST",
        endpoint="/emails/classify",
        body={
            "subject": "Homework submission complete",
            "body": "All requirements met and tested successfully",
            "use_email_similarity": False
        },
        notes="Verify new topic 'homework_demo_complete' is scored"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    topic_scores = result.response_body.get("topic_scores", {})
    new_topic_score = topic_scores.get("homework_demo_complete", 0)
    print(f"  New Topic Score: {new_topic_score:.4f}")
    print(f"  New Topic in Results: {'homework_demo_complete' in topic_scores}")

    # TEST 10: Final System Status
    print("\n[TEST 10/10] Final System Status...")
    result = tracker.run_test(
        test_number=10,
        test_name="Final System Status",
        requirement="Summary",
        method="GET",
        endpoint="/topics",
        notes="Final state of system after all demonstrations"
    )
    print(f"  Status: {result.response_status} | Duration: {result.duration_ms:.2f}ms")
    final_topics = result.response_body.get("topics", [])
    print(f"  Final Topic Count: {len(final_topics)}")

    # Read final email count
    if os.path.exists('data/emails.json'):
        with open('data/emails.json', 'r') as f:
            emails = json.load(f)
        labeled = [e for e in emails if 'ground_truth' in e and e['ground_truth']]
        print(f"  Final Email Count: {len(emails)}")
        print(f"  Labeled Emails: {len(labeled)}")

    # Generate timeline and summary
    print("\n" + "="*80)
    print("Generating timeline and visualizations...")
    timeline = tracker.generate_timeline()
    tracker.print_summary()

    return tracker, timeline

if __name__ == "__main__":
    tracker, timeline = run_complete_homework_tracking()