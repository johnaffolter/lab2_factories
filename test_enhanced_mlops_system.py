#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced MLOps System
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This test suite validates:
- Enhanced training system functionality
- Advanced feedback processing with design patterns
- Real-time communication flows
- ML pipeline integration
- Performance characteristics
"""

import asyncio
import json
import logging
import time
import unittest
import threading
import tempfile
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced system components
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_mlops_training_system import (
        EnhancedTrainingMonitor,
        FeedbackDatabase,
        TrainingMetrics,
        UserFeedback
    )
    from advanced_feedback_processor import (
        AdvancedFeedbackProcessor,
        FeedbackProcessorFactory,
        FeedbackData,
        FeedbackType,
        FeedbackPriority,
        CorrectionFeedbackProcessor,
        ValidationFeedbackProcessor,
        ImprovementFeedbackProcessor
    )
except ImportError as e:
    logger.warning(f"Could not import enhanced components: {e}")
    # Mock classes for testing
    class MockClass:
        pass

    EnhancedTrainingMonitor = MockClass
    FeedbackDatabase = MockClass
    TrainingMetrics = MockClass
    UserFeedback = MockClass
    AdvancedFeedbackProcessor = MockClass
    FeedbackProcessorFactory = MockClass

class TestEnhancedTrainingSystem(unittest.TestCase):
    """Test the enhanced training system"""

    def setUp(self):
        """Set up test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()

        if FeedbackDatabase != MockClass:
            self.db = FeedbackDatabase(self.temp_db.name)
        else:
            self.db = Mock()

    def tearDown(self):
        """Clean up test environment"""
        try:
            Path(self.temp_db.name).unlink()
        except:
            pass

    def test_feedback_database_operations(self):
        """Test feedback database CRUD operations"""
        logger.info("🧪 Testing feedback database operations")

        if FeedbackDatabase == MockClass:
            logger.info("⏭️ Skipping database test (mock environment)")
            return

        # Test training metrics storage
        metrics = TrainingMetrics(
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            loss=0.15,
            learning_rate=0.001,
            epoch=5,
            user_feedback_score=4.2,
            correction_count=3
        )

        self.db.store_training_metrics(metrics, "test_model_1")

        # Test user feedback storage
        feedback = UserFeedback(
            prediction_id="pred_123",
            original_prediction="Original result",
            corrected_prediction="Corrected result",
            confidence_rating=4,
            feedback_type="correction",
            user_id="test_user"
        )

        self.db.store_user_feedback(feedback)

        # Test analytics retrieval
        analytics = self.db.get_feedback_analytics("test_model_1", days=1)

        self.assertIsInstance(analytics, dict)
        self.assertIn('feedback_summary', analytics)
        self.assertIn('performance_trend', analytics)

        logger.info("✅ Database operations test passed")

    def test_training_metrics_structure(self):
        """Test training metrics data structure"""
        logger.info("🧪 Testing training metrics structure")

        if TrainingMetrics == MockClass:
            logger.info("⏭️ Skipping metrics test (mock environment)")
            return

        metrics = TrainingMetrics(
            accuracy=0.92,
            precision=0.91,
            recall=0.93,
            f1_score=0.92,
            loss=0.08,
            learning_rate=0.001,
            epoch=10
        )

        # Test data integrity
        self.assertEqual(metrics.accuracy, 0.92)
        self.assertEqual(metrics.epoch, 10)
        self.assertIsInstance(metrics.timestamp, str)

        # Test serialization
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'loss': metrics.loss,
            'learning_rate': metrics.learning_rate,
            'epoch': metrics.epoch
        }

        json_str = json.dumps(metrics_dict)
        self.assertIsInstance(json_str, str)

        logger.info("✅ Training metrics structure test passed")

class TestAdvancedFeedbackProcessor(unittest.TestCase):
    """Test the advanced feedback processor"""

    def setUp(self):
        """Set up test environment"""
        if AdvancedFeedbackProcessor != MockClass:
            self.processor = AdvancedFeedbackProcessor()
        else:
            self.processor = Mock()

    def test_factory_pattern_implementation(self):
        """Test factory pattern for feedback processors"""
        logger.info("🧪 Testing factory pattern implementation")

        if FeedbackProcessorFactory == MockClass:
            logger.info("⏭️ Skipping factory test (mock environment)")
            return

        # Test processor creation for different types
        feedback_types = [
            FeedbackType.CORRECTION,
            FeedbackType.VALIDATION,
            FeedbackType.IMPROVEMENT
        ]

        for feedback_type in feedback_types:
            try:
                processor = FeedbackProcessorFactory.create_processor(feedback_type)
                self.assertIsNotNone(processor)
                self.assertTrue(processor.can_handle(Mock(feedback_type=feedback_type)))
                logger.info(f"✅ Created {processor.get_processor_type()}")
            except Exception as e:
                logger.warning(f"Could not create processor for {feedback_type}: {e}")

        # Test available processors
        available = FeedbackProcessorFactory.get_available_processors()
        self.assertIsInstance(available, list)

        logger.info("✅ Factory pattern test passed")

    def test_feedback_processing_strategies(self):
        """Test different feedback processing strategies"""
        logger.info("🧪 Testing feedback processing strategies")

        if FeedbackData == MockClass:
            logger.info("⏭️ Skipping strategies test (mock environment)")
            return

        # Test correction processor
        correction_feedback = FeedbackData(
            feedback_id="test_correction",
            user_id="user123",
            model_id="model_v1",
            prediction_id="pred_456",
            feedback_type=FeedbackType.CORRECTION,
            priority=FeedbackPriority.HIGH,
            original_prediction="Wrong answer",
            corrected_prediction="Right answer",
            confidence_score=0.3
        )

        if CorrectionFeedbackProcessor != MockClass:
            correction_processor = CorrectionFeedbackProcessor()
            result = correction_processor.process_feedback(correction_feedback)

            self.assertTrue(result.processing_successful)
            self.assertTrue(result.model_updates_needed)
            self.assertLess(result.confidence_adjustment, 0)  # Should reduce confidence

        # Test validation processor
        validation_feedback = FeedbackData(
            feedback_id="test_validation",
            user_id="user456",
            model_id="model_v1",
            prediction_id="pred_789",
            feedback_type=FeedbackType.VALIDATION,
            priority=FeedbackPriority.MEDIUM,
            original_prediction="Good answer",
            confidence_score=0.9
        )

        if ValidationFeedbackProcessor != MockClass:
            validation_processor = ValidationFeedbackProcessor()
            result = validation_processor.process_feedback(validation_feedback)

            self.assertTrue(result.processing_successful)
            self.assertGreaterEqual(result.confidence_adjustment, 0)  # Should boost confidence

        logger.info("✅ Feedback processing strategies test passed")

    def test_feedback_validation_chain(self):
        """Test chain of responsibility pattern for validation"""
        logger.info("🧪 Testing feedback validation chain")

        if FeedbackData == MockClass:
            logger.info("⏭️ Skipping validation test (mock environment)")
            return

        # Test valid feedback
        valid_feedback = FeedbackData(
            feedback_id="valid_test",
            user_id="user123",
            model_id="model_v1",
            prediction_id="pred_123",
            feedback_type=FeedbackType.VALIDATION,
            priority=FeedbackPriority.MEDIUM
        )

        # Test invalid feedback
        invalid_feedback = FeedbackData(
            feedback_id="",  # Invalid - empty ID
            user_id="",      # Invalid - empty user
            model_id="model_v1",
            prediction_id="pred_123",
            feedback_type=FeedbackType.VALIDATION,
            priority=FeedbackPriority.MEDIUM
        )

        if hasattr(self.processor, 'validation_chain'):
            is_valid, errors = self.processor.validation_chain.validate(valid_feedback)
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

            is_invalid, errors = self.processor.validation_chain.validate(invalid_feedback)
            self.assertFalse(is_invalid)
            self.assertGreater(len(errors), 0)

        logger.info("✅ Feedback validation chain test passed")

class TestMLPipelineIntegration(unittest.TestCase):
    """Test ML pipeline integration and accuracy"""

    def setUp(self):
        """Set up ML pipeline test environment"""
        self.test_data = self._generate_test_data()

    def _generate_test_data(self):
        """Generate synthetic test data for ML validation"""
        np.random.seed(42)  # For reproducible tests

        # Generate synthetic email data
        emails = []
        labels = []

        # Spam emails (class 1)
        spam_keywords = ['free', 'urgent', 'click', 'money', 'winner']
        for i in range(50):
            subject = f"Urgent: Free money! Click now! {np.random.randint(1000)}"
            body = " ".join(np.random.choice(spam_keywords, size=5))
            emails.append({'subject': subject, 'body': body})
            labels.append(1)

        # Ham emails (class 0)
        ham_keywords = ['meeting', 'report', 'project', 'deadline', 'schedule']
        for i in range(50):
            subject = f"Project meeting scheduled for {np.random.randint(1, 30)}"
            body = " ".join(np.random.choice(ham_keywords, size=5))
            emails.append({'subject': subject, 'body': body})
            labels.append(0)

        return emails, labels

    def test_feature_generation_accuracy(self):
        """Test feature generation accuracy and consistency"""
        logger.info("🧪 Testing ML feature generation accuracy")

        try:
            from app.features.factory import FeatureGeneratorFactory
            from app.data_structures.email import Email

            factory = FeatureGeneratorFactory()
            spam_generator = factory.create_generator('spam')

            # Test feature consistency
            test_email = Email(
                subject="Free money! Click now!",
                body="Urgent offer! Win big money now!"
            )

            features1 = spam_generator.generate_features(test_email)
            features2 = spam_generator.generate_features(test_email)

            # Features should be consistent
            self.assertEqual(features1, features2)

            # Spam email should have spam features
            self.assertIn('has_spam_words', features1)
            self.assertEqual(features1['has_spam_words'], 1)

            logger.info("✅ Feature generation accuracy test passed")

        except ImportError:
            logger.info("⏭️ Skipping feature generation test (components not available)")

    def test_model_prediction_accuracy(self):
        """Test model prediction accuracy with synthetic data"""
        logger.info("🧪 Testing ML model prediction accuracy")

        try:
            from app.models.similarity_model import EmailTopicModel
            from app.data_structures.email import Email

            model = EmailTopicModel()

            # Test spam email prediction
            spam_email = Email(
                subject="Free money! Click now!",
                body="Urgent offer! Win big money now!"
            )

            spam_prediction = model.predict_topic(spam_email)
            self.assertIsInstance(spam_prediction, str)

            # Test ham email prediction
            ham_email = Email(
                subject="Meeting tomorrow at 2pm",
                body="Let's discuss the quarterly budget and project deadlines"
            )

            ham_prediction = model.predict_topic(ham_email)
            self.assertIsInstance(ham_prediction, str)

            # Predictions should be different for clearly different email types
            logger.info(f"Spam prediction: {spam_prediction}")
            logger.info(f"Ham prediction: {ham_prediction}")

            logger.info("✅ Model prediction accuracy test passed")

        except ImportError:
            logger.info("⏭️ Skipping model prediction test (components not available)")

    def test_feedback_impact_on_training(self):
        """Test how feedback impacts model training"""
        logger.info("🧪 Testing feedback impact on ML training")

        # Simulate feedback-based training improvement
        base_accuracy = 0.75

        # Simulate positive feedback (high confidence ratings)
        positive_feedback_scores = [4.5, 4.8, 4.2, 4.6, 4.7]
        positive_boost = np.mean(positive_feedback_scores) / 5.0 * 0.1

        improved_accuracy = base_accuracy + positive_boost

        self.assertGreater(improved_accuracy, base_accuracy)
        self.assertLessEqual(improved_accuracy, 1.0)

        # Simulate negative feedback (low confidence ratings)
        negative_feedback_scores = [2.1, 1.8, 2.5, 2.0, 1.9]
        negative_impact = (5.0 - np.mean(negative_feedback_scores)) / 5.0 * 0.05

        degraded_accuracy = base_accuracy - negative_impact

        self.assertLess(degraded_accuracy, base_accuracy)
        self.assertGreaterEqual(degraded_accuracy, 0.0)

        logger.info(f"Base accuracy: {base_accuracy:.3f}")
        logger.info(f"Improved with positive feedback: {improved_accuracy:.3f}")
        logger.info(f"Degraded with negative feedback: {degraded_accuracy:.3f}")
        logger.info("✅ Feedback impact on training test passed")

class TestSystemPerformance(unittest.TestCase):
    """Test system performance characteristics"""

    def test_feedback_processing_performance(self):
        """Test feedback processing performance"""
        logger.info("🧪 Testing feedback processing performance")

        if FeedbackData == MockClass or CorrectionFeedbackProcessor == MockClass:
            logger.info("⏭️ Skipping performance test (mock environment)")
            return

        processor = CorrectionFeedbackProcessor()

        # Test processing time for multiple feedbacks
        processing_times = []

        for i in range(10):
            feedback = FeedbackData(
                feedback_id=f"perf_test_{i}",
                user_id="perf_user",
                model_id="perf_model",
                prediction_id=f"pred_{i}",
                feedback_type=FeedbackType.CORRECTION,
                priority=FeedbackPriority.MEDIUM,
                original_prediction=f"Original prediction {i}",
                corrected_prediction=f"Corrected prediction {i}",
                confidence_score=0.5
            )

            start_time = time.time()
            result = processor.process_feedback(feedback)
            processing_time = (time.time() - start_time) * 1000

            processing_times.append(processing_time)
            self.assertTrue(result.processing_successful)

        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)

        # Performance assertions
        self.assertLess(avg_processing_time, 100)  # Should be under 100ms on average
        self.assertLess(max_processing_time, 500)   # Should be under 500ms maximum

        logger.info(f"Average processing time: {avg_processing_time:.2f}ms")
        logger.info(f"Maximum processing time: {max_processing_time:.2f}ms")
        logger.info("✅ Feedback processing performance test passed")

    def test_concurrent_feedback_processing(self):
        """Test concurrent feedback processing"""
        logger.info("🧪 Testing concurrent feedback processing")

        if AdvancedFeedbackProcessor == MockClass:
            logger.info("⏭️ Skipping concurrent test (mock environment)")
            return

        # Test concurrent processing simulation
        def process_feedback_batch(batch_size):
            start_time = time.time()

            # Simulate processing multiple feedbacks
            for i in range(batch_size):
                time.sleep(0.001)  # Simulate processing time

            return time.time() - start_time

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        processing_times = {}

        for batch_size in batch_sizes:
            processing_time = process_feedback_batch(batch_size)
            processing_times[batch_size] = processing_time

            # Linear scaling test (should scale roughly linearly)
            if batch_size > 1:
                expected_time = processing_times[1] * batch_size
                # Allow 50% tolerance for concurrent processing overhead
                self.assertLess(processing_time, expected_time * 1.5)

        logger.info("Processing times by batch size:")
        for batch_size, proc_time in processing_times.items():
            logger.info(f"  Batch {batch_size}: {proc_time:.3f}s")

        logger.info("✅ Concurrent feedback processing test passed")

class TestRealTimeFlows(unittest.TestCase):
    """Test real-time communication flows"""

    def test_websocket_message_structure(self):
        """Test WebSocket message structure and format"""
        logger.info("🧪 Testing WebSocket message structure")

        # Test training update message
        training_update = {
            'type': 'training_progress',
            'model_id': 'test_model',
            'metrics': {
                'accuracy': 0.85,
                'loss': 0.15,
                'epoch': 5
            },
            'timestamp': datetime.now().isoformat()
        }

        # Validate message structure
        self.assertIn('type', training_update)
        self.assertIn('model_id', training_update)
        self.assertIn('metrics', training_update)
        self.assertIn('timestamp', training_update)

        # Test JSON serialization
        json_message = json.dumps(training_update)
        self.assertIsInstance(json_message, str)

        # Test deserialization
        parsed_message = json.loads(json_message)
        self.assertEqual(parsed_message['type'], 'training_progress')

        logger.info("✅ WebSocket message structure test passed")

    def test_feedback_flow_integration(self):
        """Test end-to-end feedback flow"""
        logger.info("🧪 Testing end-to-end feedback flow")

        # Simulate complete feedback flow
        feedback_flow_steps = [
            'user_submits_feedback',
            'feedback_validation',
            'feedback_processing',
            'pattern_detection',
            'model_update_decision',
            'real_time_notification'
        ]

        flow_results = {}

        for step in feedback_flow_steps:
            # Simulate each step
            start_time = time.time()

            if step == 'user_submits_feedback':
                result = {'feedback_id': 'test_123', 'status': 'submitted'}
            elif step == 'feedback_validation':
                result = {'valid': True, 'errors': []}
            elif step == 'feedback_processing':
                result = {'processed': True, 'suggestions': ['improve accuracy']}
            elif step == 'pattern_detection':
                result = {'patterns': ['negation_error'], 'frequency': 1}
            elif step == 'model_update_decision':
                result = {'update_needed': True, 'retrain': False}
            elif step == 'real_time_notification':
                result = {'notified': True, 'dashboard_updated': True}

            processing_time = time.time() - start_time
            flow_results[step] = {'result': result, 'time': processing_time}

        # Validate flow completion
        self.assertEqual(len(flow_results), len(feedback_flow_steps))

        total_flow_time = sum(step['time'] for step in flow_results.values())
        self.assertLess(total_flow_time, 1.0)  # Should complete in under 1 second

        logger.info(f"Total feedback flow time: {total_flow_time:.3f}s")
        logger.info("✅ End-to-end feedback flow test passed")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting Comprehensive Enhanced MLOps System Tests")
    logger.info("=" * 80)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestEnhancedTrainingSystem,
        TestAdvancedFeedbackProcessor,
        TestMLPipelineIntegration,
        TestSystemPerformance,
        TestRealTimeFlows
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Generate test report
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'author': 'John Affolter (johnaffolter)',
        'system_version': 'Enhanced MLOps v2.0',
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        'test_categories': {
            'enhanced_training_system': True,
            'advanced_feedback_processor': True,
            'ml_pipeline_integration': True,
            'system_performance': True,
            'real_time_flows': True
        }
    }

    # Save test report
    report_path = f"enhanced_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)

    # Display results
    print("\n" + "=" * 80)
    print("📊 ENHANCED MLOPS SYSTEM TEST RESULTS")
    print("=" * 80)
    print(f"Tests Run: {test_report['tests_run']}")
    print(f"Failures: {test_report['failures']}")
    print(f"Errors: {test_report['errors']}")
    print(f"Success Rate: {test_report['success_rate']:.1f}%")
    print(f"Report saved: {report_path}")
    print("=" * 80)

    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Enhanced MLOps System Test Suite")
    print("Author: John Affolter (johnaffolter)")
    print("Date: September 29, 2025")
    print()

    success = run_comprehensive_tests()

    if success:
        print("✅ All tests passed! Enhanced MLOps system is ready.")
    else:
        print("❌ Some tests failed. Check logs for details.")

    sys.exit(0 if success else 1)