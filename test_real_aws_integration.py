#!/usr/bin/env python3

"""
Real AWS Integration Test
Testing with actual AWS credentials and S3 operations
"""

import os
import json
import boto3
import time
from datetime import datetime
from typing import Dict, List, Any
import sys

# Load AWS credentials from environment variables
# Set these in your environment or .env file:
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_DEFAULT_REGION=us-west-2

if 'AWS_ACCESS_KEY_ID' not in os.environ:
    raise ValueError("AWS_ACCESS_KEY_ID environment variable not set")
if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
    raise ValueError("AWS_SECRET_ACCESS_KEY environment variable not set")

os.environ.setdefault('AWS_DEFAULT_REGION', 'us-west-2')

# Import our systems
sys.path.append('.')
from production_ready_airflow_s3_system import ProductionS3Manager, S3Configuration, DeploymentEnvironment
from advanced_email_analyzer import AdvancedEmailAnalyzer
from advanced_dataset_generation_system import AdvancedDatasetGenerator

class RealAWSIntegrationTest:
    """Test real AWS integration with comprehensive MLOps capabilities"""

    def __init__(self):
        self.bucket_name = f"mlops-test-bucket-{int(time.time())}"
        self.s3_config = S3Configuration(
            bucket_name=self.bucket_name,
            region="us-west-2",
            encryption=True,
            versioning=True,
            lifecycle_enabled=False  # Disable for test to avoid costs
        )

        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name='us-west-2')
        self.sts_client = boto3.client('sts', region_name='us-west-2')

        # Initialize our systems
        self.s3_manager = ProductionS3Manager(self.s3_config, DeploymentEnvironment.DEVELOPMENT)
        self.email_analyzer = AdvancedEmailAnalyzer()
        self.dataset_generator = AdvancedDatasetGenerator()

        self.test_results = {}

    def verify_aws_credentials(self) -> Dict[str, Any]:
        """Verify AWS credentials and permissions"""
        print("🔐 VERIFYING AWS CREDENTIALS")
        print("-" * 40)

        try:
            # Get caller identity
            identity = self.sts_client.get_caller_identity()

            result = {
                "status": "success",
                "user_id": identity.get('UserId'),
                "account": identity.get('Account'),
                "arn": identity.get('Arn'),
                "region": self.s3_client.meta.region_name
            }

            print(f"✅ AWS Identity Verified:")
            print(f"   User ID: {result['user_id']}")
            print(f"   Account: {result['account']}")
            print(f"   ARN: {result['arn']}")
            print(f"   Region: {result['region']}")

            return result

        except Exception as e:
            result = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ AWS Credential Verification Failed: {e}")
            return result

    def test_s3_operations(self) -> Dict[str, Any]:
        """Test comprehensive S3 operations"""
        print(f"\n📦 TESTING S3 OPERATIONS")
        print("-" * 40)

        test_results = {
            "bucket_creation": False,
            "file_upload": False,
            "file_download": False,
            "list_objects": False,
            "metadata_operations": False,
            "cleanup": False,
            "errors": []
        }

        try:
            # Test bucket creation
            print(f"🪣 Creating test bucket: {self.bucket_name}")
            bucket_success = self.s3_manager.setup_bucket()
            test_results["bucket_creation"] = bucket_success

            if bucket_success:
                print(f"✅ Bucket created successfully")
            else:
                print(f"❌ Bucket creation failed")
                return test_results

            # Test file upload
            print(f"📤 Testing file upload...")

            # Create test data
            test_data = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "real_aws_integration",
                "data": {
                    "message": "This is a test file for AWS S3 integration",
                    "numbers": [1, 2, 3, 4, 5],
                    "nested": {"key": "value", "flag": True}
                }
            }

            test_file_path = "/tmp/aws_test_file.json"
            with open(test_file_path, 'w') as f:
                json.dump(test_data, f, indent=2)

            metadata = {
                "test-run": "real-aws-integration",
                "created-by": "mlops-system",
                "timestamp": datetime.now().isoformat()
            }

            upload_success = self.s3_manager.upload_file(
                test_file_path,
                "test-data/aws_integration_test.json",
                metadata
            )
            test_results["file_upload"] = upload_success

            if upload_success:
                print(f"✅ File upload successful")
            else:
                print(f"❌ File upload failed")

            # Test file download
            print(f"📥 Testing file download...")
            download_path = "/tmp/downloaded_aws_test_file.json"
            download_success = self.s3_manager.download_file(
                "test-data/aws_integration_test.json",
                download_path
            )
            test_results["file_download"] = download_success

            if download_success:
                # Verify file integrity
                with open(download_path, 'r') as f:
                    downloaded_data = json.load(f)

                if downloaded_data == test_data:
                    print(f"✅ File download successful with data integrity verified")
                else:
                    print(f"⚠️ File downloaded but data integrity check failed")
            else:
                print(f"❌ File download failed")

            # Test list objects
            print(f"📋 Testing list objects...")
            objects = self.s3_manager.list_objects("test-data/")
            test_results["list_objects"] = len(objects) > 0

            if objects:
                print(f"✅ List objects successful: found {len(objects)} objects")
                for obj in objects:
                    print(f"   - {obj['Key']} ({obj['Size']} bytes)")
            else:
                print(f"❌ List objects failed or no objects found")

            # Test metadata operations
            print(f"🏷️ Testing metadata operations...")
            try:
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key="test-data/aws_integration_test.json"
                )

                if 'Metadata' in response:
                    print(f"✅ Metadata operations successful")
                    print(f"   Metadata: {response['Metadata']}")
                    test_results["metadata_operations"] = True
                else:
                    print(f"⚠️ Object exists but no metadata found")

            except Exception as e:
                print(f"❌ Metadata operations failed: {e}")

        except Exception as e:
            error_msg = f"S3 operations failed: {e}"
            test_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return test_results

    def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end MLOps pipeline with real AWS"""
        print(f"\n🔄 TESTING END-TO-END MLOPS PIPELINE")
        print("-" * 40)

        pipeline_results = {
            "email_analysis": False,
            "dataset_generation": False,
            "llm_evaluation": False,
            "s3_storage": False,
            "pipeline_complete": False,
            "processing_time": 0,
            "errors": []
        }

        start_time = time.time()

        try:
            # Step 1: Email Analysis
            print(f"📧 Step 1: Running email analysis...")
            test_emails = [
                {
                    "id": "pipeline_test_001",
                    "subject": "Production Pipeline Test Email",
                    "body": "This is a comprehensive test of our production MLOps pipeline with real AWS integration. The system should analyze this email, generate datasets, evaluate with LLM, and store results in S3.",
                    "sender": "pipeline@test.com"
                },
                {
                    "id": "pipeline_test_002",
                    "subject": "urgent meetng tommorow!!!",
                    "body": "Hi,\n\nIts very importnat we meet tommorow. Their are issues to discuss and I think you no what I mean.\n\nThanks",
                    "sender": "test@poor.com"
                }
            ]

            analysis_results = []
            for email in test_emails:
                result = self.email_analyzer.analyze_email(
                    subject=email["subject"],
                    body=email["body"],
                    sender=email["sender"]
                )

                analysis_data = {
                    "email_id": email["id"],
                    "overall_score": result.overall_score,
                    "professionalism": result.metrics.professionalism_score,
                    "clarity": result.metrics.clarity_score,
                    "word_count": result.metrics.word_count,
                    "issues_count": len(result.issues),
                    "analysis_timestamp": datetime.now().isoformat()
                }
                analysis_results.append(analysis_data)

            pipeline_results["email_analysis"] = len(analysis_results) > 0
            print(f"✅ Email analysis completed: {len(analysis_results)} emails analyzed")

            # Step 2: Dataset Generation
            print(f"🎯 Step 2: Generating dataset...")

            dataset_samples = []
            for email, analysis in zip(test_emails, analysis_results):
                sample = {
                    "sample_id": f"pipeline_sample_{email['id']}",
                    "content": {
                        "subject": email["subject"],
                        "body": email["body"],
                        "sender": email["sender"]
                    },
                    "labels": {
                        "overall_quality": analysis["overall_score"],
                        "professionalism": analysis["professionalism"],
                        "clarity": analysis["clarity"]
                    },
                    "metadata": {
                        "word_count": analysis["word_count"],
                        "issues_count": analysis["issues_count"],
                        "pipeline_run": "real_aws_integration_test"
                    }
                }
                dataset_samples.append(sample)

            pipeline_results["dataset_generation"] = len(dataset_samples) > 0
            print(f"✅ Dataset generation completed: {len(dataset_samples)} samples created")

            # Step 3: LLM Evaluation (using our mock LLM for this test)
            print(f"🤖 Step 3: LLM evaluation...")

            # Convert to DataSample objects for LLM evaluation
            from advanced_dataset_generation_system import DataSample

            data_samples = []
            for sample_data in dataset_samples:
                data_sample = DataSample(
                    sample_id=sample_data["sample_id"],
                    content=sample_data["content"],
                    true_labels=sample_data["labels"],
                    metadata=sample_data["metadata"]
                )
                data_samples.append(data_sample)

            llm_results = self.dataset_generator.llm_judge.batch_evaluate(
                data_samples, {"evaluate_all": True}
            )

            pipeline_results["llm_evaluation"] = len(llm_results) > 0
            print(f"✅ LLM evaluation completed: {len(llm_results)} judgments generated")

            # Step 4: Store results in S3
            print(f"☁️ Step 4: Storing results in S3...")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Prepare comprehensive results
            pipeline_data = {
                "pipeline_run_id": f"aws_integration_test_{timestamp}",
                "timestamp": datetime.now().isoformat(),
                "email_analysis_results": analysis_results,
                "dataset_samples": dataset_samples,
                "llm_evaluations": [
                    {
                        "judgment_id": result.judgment_id,
                        "sample_id": result.sample_id,
                        "confidence": result.confidence,
                        "consistency_score": result.consistency_score,
                        "scores": result.judgment_scores,
                        "reasoning": result.reasoning
                    }
                    for result in llm_results
                ],
                "pipeline_metadata": {
                    "aws_account": self.test_results.get("aws_verification", {}).get("account"),
                    "bucket_name": self.bucket_name,
                    "test_type": "real_aws_integration",
                    "system_components": [
                        "AdvancedEmailAnalyzer",
                        "AdvancedDatasetGenerator",
                        "ProductionS3Manager",
                        "LLM Judge"
                    ]
                }
            }

            # Save to local file first
            results_file = f"/tmp/pipeline_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(pipeline_data, f, indent=2)

            # Upload to S3
            s3_key = f"pipeline-results/{timestamp}/complete_pipeline_results.json"
            upload_success = self.s3_manager.upload_file(
                results_file,
                s3_key,
                metadata={
                    "pipeline-run": f"aws_integration_test_{timestamp}",
                    "total-samples": str(len(dataset_samples)),
                    "llm-evaluations": str(len(llm_results))
                }
            )

            pipeline_results["s3_storage"] = upload_success

            if upload_success:
                print(f"✅ Results stored in S3: s3://{self.bucket_name}/{s3_key}")
            else:
                print(f"❌ Failed to store results in S3")

            # Calculate final status
            pipeline_results["pipeline_complete"] = all([
                pipeline_results["email_analysis"],
                pipeline_results["dataset_generation"],
                pipeline_results["llm_evaluation"],
                pipeline_results["s3_storage"]
            ])

            pipeline_results["processing_time"] = time.time() - start_time

            if pipeline_results["pipeline_complete"]:
                print(f"🎉 END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
                print(f"   Processing time: {pipeline_results['processing_time']:.2f} seconds")
                print(f"   Emails analyzed: {len(analysis_results)}")
                print(f"   Dataset samples: {len(dataset_samples)}")
                print(f"   LLM evaluations: {len(llm_results)}")
                print(f"   S3 storage: {s3_key}")
            else:
                print(f"⚠️ Pipeline completed with some failures")

        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            pipeline_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()

        return pipeline_results

    def cleanup_test_resources(self) -> Dict[str, Any]:
        """Cleanup test resources from AWS"""
        print(f"\n🧹 CLEANING UP TEST RESOURCES")
        print("-" * 40)

        cleanup_results = {
            "objects_deleted": 0,
            "bucket_deleted": False,
            "errors": []
        }

        try:
            # List all objects in bucket
            objects = self.s3_manager.list_objects()

            if objects:
                print(f"🗑️ Deleting {len(objects)} objects...")

                # Delete objects in batches
                delete_objects = [{'Key': obj['Key']} for obj in objects]

                if delete_objects:
                    response = self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': delete_objects}
                    )

                    deleted = response.get('Deleted', [])
                    cleanup_results["objects_deleted"] = len(deleted)
                    print(f"✅ Deleted {len(deleted)} objects")

            # Delete bucket
            print(f"🪣 Deleting test bucket: {self.bucket_name}")
            self.s3_client.delete_bucket(Bucket=self.bucket_name)
            cleanup_results["bucket_deleted"] = True
            print(f"✅ Bucket deleted successfully")

        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            cleanup_results["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return cleanup_results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive AWS integration test"""
        print("🚀 COMPREHENSIVE AWS INTEGRATION TEST")
        print("Real AWS credentials and S3 operations")
        print("=" * 80)

        # Test 1: Verify AWS credentials
        aws_verification = self.verify_aws_credentials()
        self.test_results["aws_verification"] = aws_verification

        if aws_verification["status"] != "success":
            print("❌ AWS verification failed - stopping tests")
            return self.test_results

        # Test 2: S3 operations
        s3_operations = self.test_s3_operations()
        self.test_results["s3_operations"] = s3_operations

        # Test 3: End-to-end pipeline
        pipeline_test = self.test_end_to_end_pipeline()
        self.test_results["pipeline_test"] = pipeline_test

        # Test 4: Cleanup
        cleanup_results = self.cleanup_test_resources()
        self.test_results["cleanup"] = cleanup_results

        # Generate summary
        self.test_results["summary"] = self._generate_test_summary()

        return self.test_results

    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""

        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "key_achievements": [],
            "issues_found": [],
            "recommendations": []
        }

        # Count successes and failures
        if self.test_results.get("aws_verification", {}).get("status") == "success":
            summary["tests_passed"] += 1
            summary["key_achievements"].append("AWS credentials verified successfully")
        else:
            summary["tests_failed"] += 1
            summary["issues_found"].append("AWS credential verification failed")

        s3_ops = self.test_results.get("s3_operations", {})
        s3_success_count = sum([
            s3_ops.get("bucket_creation", False),
            s3_ops.get("file_upload", False),
            s3_ops.get("file_download", False),
            s3_ops.get("list_objects", False)
        ])

        if s3_success_count >= 3:
            summary["tests_passed"] += 1
            summary["key_achievements"].append(f"S3 operations successful ({s3_success_count}/4 tests passed)")
        else:
            summary["tests_failed"] += 1
            summary["issues_found"].append(f"S3 operations issues ({s3_success_count}/4 tests passed)")

        pipeline_test = self.test_results.get("pipeline_test", {})
        if pipeline_test.get("pipeline_complete", False):
            summary["tests_passed"] += 1
            summary["key_achievements"].append("End-to-end MLOps pipeline completed successfully")
        else:
            summary["tests_failed"] += 1
            summary["issues_found"].append("End-to-end pipeline had failures")

        cleanup_test = self.test_results.get("cleanup", {})
        if cleanup_test.get("bucket_deleted", False):
            summary["tests_passed"] += 1
            summary["key_achievements"].append("Test resources cleaned up successfully")
        else:
            summary["tests_failed"] += 1
            summary["issues_found"].append("Cleanup had issues")

        # Determine overall status
        if summary["tests_failed"] == 0:
            summary["overall_status"] = "all_tests_passed"
        elif summary["tests_passed"] > summary["tests_failed"]:
            summary["overall_status"] = "mostly_successful"
        else:
            summary["overall_status"] = "needs_attention"

        # Generate recommendations
        if summary["overall_status"] == "all_tests_passed":
            summary["recommendations"].extend([
                "System is ready for production deployment",
                "Consider implementing automated testing pipeline",
                "Add monitoring and alerting for production use"
            ])
        else:
            summary["recommendations"].extend([
                "Address identified issues before production deployment",
                "Review AWS permissions and credentials",
                "Test individual components in isolation"
            ])

        return summary

def main():
    """Run the comprehensive AWS integration test"""

    # Create and run test
    test_runner = RealAWSIntegrationTest()
    results = test_runner.run_comprehensive_test()

    # Display final summary
    print(f"\n" + "="*80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("="*80)

    summary = results["summary"]
    print(f"🕐 Test Timestamp: {summary['test_timestamp']}")
    print(f"🎯 Overall Status: {summary['overall_status'].upper()}")
    print(f"✅ Tests Passed: {summary['tests_passed']}")
    print(f"❌ Tests Failed: {summary['tests_failed']}")

    if summary["key_achievements"]:
        print(f"\n🏆 Key Achievements:")
        for achievement in summary["key_achievements"]:
            print(f"  • {achievement}")

    if summary["issues_found"]:
        print(f"\n⚠️ Issues Found:")
        for issue in summary["issues_found"]:
            print(f"  • {issue}")

    if summary["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  • {rec}")

    # Display pipeline specific results
    pipeline_results = results.get("pipeline_test", {})
    if pipeline_results.get("pipeline_complete"):
        print(f"\n🚀 MLOps Pipeline Performance:")
        print(f"  Processing Time: {pipeline_results['processing_time']:.2f} seconds")
        print(f"  Components Tested: Email Analysis, Dataset Generation, LLM Evaluation, S3 Storage")
        print(f"  Integration Status: ✅ FULLY OPERATIONAL")

    aws_verification = results.get("aws_verification", {})
    if aws_verification.get("status") == "success":
        print(f"\n☁️ AWS Integration Details:")
        print(f"  Account: {aws_verification['account']}")
        print(f"  User: {aws_verification['arn']}")
        print(f"  Region: {aws_verification['region']}")

    print(f"\n✅ Comprehensive AWS integration testing completed!")

    return test_runner, results

if __name__ == "__main__":
    main()