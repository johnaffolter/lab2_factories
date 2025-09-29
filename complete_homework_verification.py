#!/usr/bin/env python3
"""
Complete Homework Verification System
Tests Homework 1, Lab 2, and Lab 3 components with screenshot capture
"""

import os
import sys
import json
import subprocess
import requests
from datetime import datetime
from pathlib import Path
import time
import traceback

# Add paths
sys.path.append('/Users/johnaffolter/lab_2_homework/lab2_factories')

# Create directories
Path("verification_results").mkdir(exist_ok=True)
Path("screenshots").mkdir(exist_ok=True)

class HomeworkVerifier:
    """Complete verification system for all homework components"""

    def __init__(self):
        self.results = {
            "verification_timestamp": datetime.now().isoformat(),
            "homework_1": {},
            "lab_2": {},
            "lab_3": {},
            "screenshots": [],
            "overall_status": "unknown"
        }

    def capture_screenshot(self, name, description):
        """Capture screenshot using macOS screencapture"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshots/{name}_{timestamp}.png"

        try:
            # Use macOS screencapture command
            subprocess.run([
                "screencapture",
                "-x",  # Don't play camera sound
                "-t", "png",  # PNG format
                filename
            ], check=True)

            print(f"📸 Screenshot captured: {description}")
            print(f"   📁 Saved to: {filename}")

            self.results["screenshots"].append({
                "name": name,
                "description": description,
                "filename": filename,
                "timestamp": timestamp,
                "status": "success"
            })

            return filename

        except Exception as e:
            print(f"❌ Screenshot failed: {e}")
            self.results["screenshots"].append({
                "name": name,
                "description": description,
                "error": str(e),
                "status": "failed"
            })
            return None

    def verify_homework_1(self):
        """Verify Homework 1: Factory Pattern Implementation"""
        print("\n" + "="*60)
        print("📝 HOMEWORK 1: FACTORY PATTERN VERIFICATION")
        print("="*60)

        hw1_results = {
            "factory_pattern": "unknown",
            "email_processing": "unknown",
            "feature_generators": "unknown",
            "models": "unknown",
            "tests_passing": "unknown"
        }

        try:
            # Test factory pattern implementation
            print("\n1. Testing Factory Pattern...")
            from app.features.factory import FeatureGeneratorFactory

            # Test factory creation
            factory = FeatureGeneratorFactory()
            print("✅ Factory instantiated successfully")

            # Test generator creation
            generators = factory.get_all_generators()
            print(f"✅ Found {len(generators)} feature generators")

            hw1_results["factory_pattern"] = "pass"

        except Exception as e:
            print(f"❌ Factory pattern test failed: {e}")
            hw1_results["factory_pattern"] = "fail"

        try:
            # Test email processing
            print("\n2. Testing Email Processing...")
            from app.dataclasses import Email
            from app.features.generators import SpamFeatureGenerator

            # Create test email
            test_email = Email(
                subject="Test Subject",
                body="This is a test email body",
                sender="test@example.com",
                timestamp=datetime.now()
            )

            # Test feature generation
            spam_gen = SpamFeatureGenerator()
            features = spam_gen.generate_features(test_email)
            print(f"✅ Generated features: {features}")

            hw1_results["email_processing"] = "pass"

        except Exception as e:
            print(f"❌ Email processing test failed: {e}")
            hw1_results["email_processing"] = "fail"

        try:
            # Test models
            print("\n3. Testing ML Models...")
            from app.models.similarity_model import EmailClassifierModel

            model = EmailClassifierModel()
            topics = model.get_all_topics_with_descriptions()
            print(f"✅ Model loaded with {len(topics)} topics")

            hw1_results["models"] = "pass"

        except Exception as e:
            print(f"❌ Model test failed: {e}")
            hw1_results["models"] = "fail"

        # Take screenshot
        self.capture_screenshot("homework1_verification", "Homework 1 Factory Pattern Test Results")

        self.results["homework_1"] = hw1_results
        return hw1_results

    def verify_lab_2(self):
        """Verify Lab 2: Advanced MLOps System"""
        print("\n" + "="*60)
        print("🚀 LAB 2: MLOPS SYSTEM VERIFICATION")
        print("="*60)

        lab2_results = {
            "api_server": "unknown",
            "database_connections": "unknown",
            "airflow": "unknown",
            "monitoring": "unknown",
            "s3_integration": "unknown"
        }

        try:
            # Test API server
            print("\n1. Testing API Server...")

            # Try to start if not running
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                print(f"✅ API server responding: {response.status_code}")
                lab2_results["api_server"] = "pass"
            except:
                print("⚠️ API server not running, attempting to start...")
                # Could start server here if needed
                lab2_results["api_server"] = "not_running"

        except Exception as e:
            print(f"❌ API server test failed: {e}")
            lab2_results["api_server"] = "fail"

        try:
            # Test Airflow
            print("\n2. Testing Airflow...")

            result = subprocess.run([
                "docker", "exec", "airflow-standalone", "airflow", "dags", "list"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                dag_count = len([line for line in result.stdout.split('\n') if '|' in line and 'dag_id' not in line and line.strip()])
                print(f"✅ Airflow running with {dag_count} DAGs")
                lab2_results["airflow"] = "pass"
            else:
                print("❌ Airflow not responding")
                lab2_results["airflow"] = "fail"

        except Exception as e:
            print(f"❌ Airflow test failed: {e}")
            lab2_results["airflow"] = "fail"

        try:
            # Test S3 Integration
            print("\n3. Testing S3 Integration...")
            import boto3

            s3 = boto3.client('s3')
            buckets = s3.list_buckets()
            bucket_count = len(buckets.get('Buckets', []))
            print(f"✅ S3 connected with {bucket_count} buckets")
            lab2_results["s3_integration"] = "pass"

        except Exception as e:
            print(f"❌ S3 integration test failed: {e}")
            lab2_results["s3_integration"] = "fail"

        # Take screenshot
        self.capture_screenshot("lab2_verification", "Lab 2 MLOps System Status")

        self.results["lab_2"] = lab2_results
        return lab2_results

    def verify_lab_3(self):
        """Verify Lab 3: S3 Integration"""
        print("\n" + "="*60)
        print("☁️ LAB 3: S3 INTEGRATION VERIFICATION")
        print("="*60)

        lab3_results = {
            "s3_dags": "unknown",
            "ml_pipeline": "unknown",
            "data_upload": "unknown",
            "model_storage": "unknown"
        }

        try:
            # Check Lab 3 DAGs
            print("\n1. Checking Lab 3 DAGs...")

            result = subprocess.run([
                "docker", "exec", "airflow-standalone", "airflow", "dags", "list"
            ], capture_output=True, text=True, timeout=30)

            lab3_dags = [line for line in result.stdout.split('\n')
                        if any(keyword in line.lower() for keyword in ['lab3', 's3', 'mlops_s3'])]

            if lab3_dags:
                print(f"✅ Found {len(lab3_dags)} Lab 3 DAGs")
                for dag in lab3_dags[:3]:  # Show first 3
                    print(f"   📋 {dag.split('|')[0].strip()}")
                lab3_results["s3_dags"] = "pass"
            else:
                print("⚠️ No Lab 3 DAGs found")
                lab3_results["s3_dags"] = "fail"

        except Exception as e:
            print(f"❌ Lab 3 DAG check failed: {e}")
            lab3_results["s3_dags"] = "fail"

        try:
            # Test ML Pipeline DAG
            print("\n2. Testing ML Pipeline DAG...")

            result = subprocess.run([
                "docker", "exec", "airflow-standalone", "airflow", "dags", "trigger", "mlops_s3_pipeline"
            ], capture_output=True, text=True, timeout=30)

            if "queued" in result.stdout.lower() or result.returncode == 0:
                print("✅ ML Pipeline DAG triggered successfully")
                lab3_results["ml_pipeline"] = "pass"
            else:
                print("⚠️ ML Pipeline DAG trigger failed")
                lab3_results["ml_pipeline"] = "fail"

        except Exception as e:
            print(f"❌ ML Pipeline test failed: {e}")
            lab3_results["ml_pipeline"] = "fail"

        # Take screenshot
        self.capture_screenshot("lab3_verification", "Lab 3 S3 Integration Status")

        self.results["lab_3"] = lab3_results
        return lab3_results

    def test_screenshot_system(self):
        """Test the screenshot capture system"""
        print("\n" + "="*60)
        print("📸 SCREENSHOT SYSTEM VERIFICATION")
        print("="*60)

        # Take test screenshots
        screenshots_taken = []

        # Desktop screenshot
        filename = self.capture_screenshot("desktop_test", "Desktop Test Screenshot")
        if filename:
            screenshots_taken.append(filename)

        # Terminal screenshot (if applicable)
        filename = self.capture_screenshot("terminal_test", "Terminal Screenshot Test")
        if filename:
            screenshots_taken.append(filename)

        print(f"\n✅ Screenshot system test complete")
        print(f"📁 Captured {len(screenshots_taken)} test screenshots")

        return len(screenshots_taken) > 0

    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        # Calculate overall status
        all_tests = []

        # Count passes and fails
        for section in ["homework_1", "lab_2", "lab_3"]:
            section_results = self.results.get(section, {})
            for test_name, result in section_results.items():
                all_tests.append(result)

        pass_count = all_tests.count("pass")
        total_tests = len(all_tests)

        if total_tests == 0:
            overall_status = "no_tests"
        elif pass_count == total_tests:
            overall_status = "all_pass"
        elif pass_count > total_tests // 2:
            overall_status = "mostly_pass"
        else:
            overall_status = "mostly_fail"

        self.results["overall_status"] = overall_status
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": pass_count,
            "failed": total_tests - pass_count,
            "pass_rate": f"{(pass_count/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
        }

        # Save detailed results
        report_path = f"verification_results/complete_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Create markdown report
        md_report_path = f"verification_results/HOMEWORK_STATUS_REPORT.md"
        self.create_markdown_report(md_report_path)

        print(f"\n📄 Reports saved:")
        print(f"   📋 Detailed JSON: {report_path}")
        print(f"   📄 Summary Markdown: {md_report_path}")

        return self.results

    def create_markdown_report(self, path):
        """Create markdown status report"""
        content = f"""# MLOps Homework Complete Status Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Status: {self.results['overall_status'].upper()}

### Summary
- **Total Tests:** {self.results['summary']['total_tests']}
- **Passed:** {self.results['summary']['passed']}
- **Failed:** {self.results['summary']['failed']}
- **Pass Rate:** {self.results['summary']['pass_rate']}

---

## 📝 Homework 1: Factory Pattern
"""

        # Add Homework 1 results
        hw1 = self.results.get('homework_1', {})
        for test, status in hw1.items():
            emoji = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
            content += f"- {emoji} **{test.replace('_', ' ').title()}:** {status}\n"

        content += f"""
---

## 🚀 Lab 2: MLOps System
"""

        # Add Lab 2 results
        lab2 = self.results.get('lab_2', {})
        for test, status in lab2.items():
            emoji = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
            content += f"- {emoji} **{test.replace('_', ' ').title()}:** {status}\n"

        content += f"""
---

## ☁️ Lab 3: S3 Integration
"""

        # Add Lab 3 results
        lab3 = self.results.get('lab_3', {})
        for test, status in lab3.items():
            emoji = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
            content += f"- {emoji} **{test.replace('_', ' ').title()}:** {status}\n"

        content += f"""
---

## 📸 Screenshots Captured
"""

        # Add screenshot info
        screenshots = self.results.get('screenshots', [])
        for screenshot in screenshots:
            emoji = "✅" if screenshot.get('status') == 'success' else "❌"
            content += f"- {emoji} **{screenshot['description']}**\n"
            if screenshot.get('filename'):
                content += f"  - File: `{screenshot['filename']}`\n"

        content += f"""
---

## Next Steps

Based on the verification results:

"""

        if self.results['overall_status'] == 'all_pass':
            content += "🎉 **All systems operational!** Your MLOps homework is complete and working properly."
        elif self.results['overall_status'] == 'mostly_pass':
            content += "👍 **Mostly working!** Most components are operational. Review failed tests and address any issues."
        else:
            content += "⚠️ **Needs attention!** Several components need fixing. Review the failed tests above."

        content += f"""

## System Information

- **Airflow UI Access:** http://localhost:8888 (nginx proxy)
- **API Access:** http://localhost:8000
- **Screenshot Directory:** `screenshots/`
- **Verification Results:** `verification_results/`

---

*Report generated by Complete Homework Verification System*
"""

        with open(path, 'w') as f:
            f.write(content)

def main():
    """Main verification process"""
    print("🎯 COMPLETE MLOPS HOMEWORK VERIFICATION")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print()

    verifier = HomeworkVerifier()

    try:
        # Run all verifications
        verifier.verify_homework_1()
        verifier.verify_lab_2()
        verifier.verify_lab_3()
        verifier.test_screenshot_system()

        # Generate comprehensive report
        results = verifier.generate_comprehensive_report()

        print("\n" + "="*80)
        print("🏁 VERIFICATION COMPLETE!")
        print("="*80)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Pass Rate: {results['summary']['pass_rate']}")
        print(f"Screenshots: {len(results['screenshots'])}")
        print("\n📄 Check verification_results/ for detailed reports")

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()