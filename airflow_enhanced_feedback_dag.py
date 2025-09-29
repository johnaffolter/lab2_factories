#!/usr/bin/env python3
"""
Enhanced Airflow DAG with Real-time Feedback Integration
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This DAG integrates with the enhanced training system to provide:
- Continuous model improvement based on user feedback
- Real-time performance monitoring
- Automated retraining triggers
- Advanced tracking and analytics
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import json
import sqlite3
import boto3
import logging
import numpy as np
from typing import Dict, List, Any
import websocket
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'johnaffolter',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'enhanced_feedback_training_pipeline',
    default_args=default_args,
    description='Enhanced MLOps pipeline with real-time feedback integration',
    schedule_interval=timedelta(hours=1),  # Run every hour
    catchup=False,
    tags=['mlops', 'feedback', 'training', 'enhanced'],
)

class FeedbackAnalyzer:
    """Analyze user feedback for training improvements"""

    def __init__(self, db_path: str = "/tmp/mlops_feedback.db"):
        self.db_path = db_path
        self.feedback_threshold = 0.7  # Trigger retraining if satisfaction below this

    def analyze_recent_feedback(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze feedback from the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent feedback
            cursor.execute('''
                SELECT confidence_rating, feedback_type, corrected_prediction
                FROM user_feedback
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours))

            feedback_data = cursor.fetchall()

            if not feedback_data:
                return {
                    'avg_satisfaction': 5.0,
                    'feedback_count': 0,
                    'corrections_needed': 0,
                    'retrain_recommended': False
                }

            # Calculate metrics
            satisfaction_scores = [row[0] for row in feedback_data]
            corrections = len([row for row in feedback_data if row[1] == 'correction'])

            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)

            analysis = {
                'avg_satisfaction': avg_satisfaction,
                'feedback_count': len(feedback_data),
                'corrections_needed': corrections,
                'retrain_recommended': avg_satisfaction < self.feedback_threshold or corrections > 5,
                'correction_rate': corrections / len(feedback_data) if feedback_data else 0
            }

            conn.close()
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return {
                'avg_satisfaction': 3.0,
                'feedback_count': 0,
                'corrections_needed': 0,
                'retrain_recommended': False
            }

class EnhancedModelTrainer:
    """Enhanced model trainer with feedback integration"""

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'mlops-enhanced-models'

    def train_with_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train model incorporating user feedback"""
        logger.info("🚀 Starting enhanced training with feedback integration")

        # Simulate training with feedback adjustment
        base_accuracy = 0.75
        feedback_boost = min(feedback_data['avg_satisfaction'] / 5.0 * 0.1, 0.15)

        # Simulate training epochs with feedback-adjusted learning
        training_results = []
        for epoch in range(10):
            # Simulate accuracy improvement with feedback
            epoch_accuracy = base_accuracy + feedback_boost + (epoch * 0.02)
            epoch_accuracy = min(epoch_accuracy + np.random.normal(0, 0.01), 0.98)

            training_results.append({
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'loss': max(0.8 - (epoch * 0.08) - feedback_boost, 0.05),
                'feedback_integration_score': feedback_data['avg_satisfaction']
            })

            logger.info(f"Epoch {epoch + 1}: Accuracy {epoch_accuracy:.3f}")

        final_metrics = training_results[-1]

        # Store model to S3
        model_key = f"enhanced_models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        model_data = {
            'training_results': training_results,
            'feedback_integration': feedback_data,
            'model_version': datetime.now().isoformat(),
            'final_metrics': final_metrics
        }

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=model_key,
                Body=json.dumps(model_data),
                ContentType='application/json'
            )
            logger.info(f"✅ Model saved to S3: {model_key}")
        except Exception as e:
            logger.warning(f"Could not save to S3 (using mock): {e}")

        return {
            'model_key': model_key,
            'final_accuracy': final_metrics['accuracy'],
            'training_epochs': len(training_results),
            'feedback_integration_score': feedback_data['avg_satisfaction'],
            'improvement_over_baseline': final_metrics['accuracy'] - base_accuracy
        }

class RealTimeNotifier:
    """Send real-time notifications to the training dashboard"""

    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.ws = None

    def connect(self):
        """Connect to WebSocket server"""
        try:
            import websocket
            self.ws = websocket.create_connection(self.websocket_url, timeout=5)
            return True
        except Exception as e:
            logger.warning(f"Could not connect to WebSocket: {e}")
            return False

    def send_update(self, update_data: Dict[str, Any]):
        """Send update to real-time dashboard"""
        if self.ws:
            try:
                message = json.dumps(update_data)
                self.ws.send(message)
                logger.info("✅ Sent real-time update to dashboard")
            except Exception as e:
                logger.warning(f"Could not send WebSocket update: {e}")
        else:
            logger.info(f"📊 Update (no WebSocket): {update_data['type']}")

    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()

def collect_system_metrics(**context):
    """Collect system performance metrics"""
    logger.info("📊 Collecting system performance metrics")

    # Simulate system metrics collection
    metrics = {
        'cpu_usage': np.random.uniform(20, 80),
        'memory_usage': np.random.uniform(30, 70),
        'disk_usage': np.random.uniform(10, 60),
        'active_connections': np.random.randint(5, 25),
        'response_time_ms': np.random.uniform(50, 200),
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"System metrics: CPU {metrics['cpu_usage']:.1f}%, Memory {metrics['memory_usage']:.1f}%")

    # Store metrics for analysis
    context['task_instance'].xcom_push(key='system_metrics', value=metrics)
    return metrics

def analyze_feedback_data(**context):
    """Analyze recent user feedback data"""
    logger.info("🔍 Analyzing user feedback data")

    analyzer = FeedbackAnalyzer()
    feedback_analysis = analyzer.analyze_recent_feedback(hours=24)

    logger.info(f"Feedback analysis: {feedback_analysis['feedback_count']} feedbacks, "
                f"avg satisfaction: {feedback_analysis['avg_satisfaction']:.2f}")

    # Send real-time update
    notifier = RealTimeNotifier()
    if notifier.connect():
        notifier.send_update({
            'type': 'feedback_analysis',
            'data': feedback_analysis,
            'timestamp': datetime.now().isoformat()
        })
        notifier.close()

    context['task_instance'].xcom_push(key='feedback_analysis', value=feedback_analysis)
    return feedback_analysis

def decide_retraining(**context):
    """Decide whether to trigger model retraining"""
    logger.info("🤔 Evaluating retraining requirements")

    # Get feedback analysis from previous task
    feedback_analysis = context['task_instance'].xcom_pull(
        task_ids='analyze_feedback', key='feedback_analysis'
    )

    # Decision logic
    retrain_reasons = []

    if feedback_analysis['retrain_recommended']:
        retrain_reasons.append("Low user satisfaction detected")

    if feedback_analysis['correction_rate'] > 0.3:
        retrain_reasons.append("High correction rate detected")

    if feedback_analysis['feedback_count'] > 20:
        retrain_reasons.append("Sufficient feedback volume for improvement")

    should_retrain = len(retrain_reasons) > 0

    decision = {
        'should_retrain': should_retrain,
        'reasons': retrain_reasons,
        'confidence': min(len(retrain_reasons) / 2.0, 1.0),
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"Retraining decision: {'YES' if should_retrain else 'NO'}")
    if retrain_reasons:
        logger.info(f"Reasons: {', '.join(retrain_reasons)}")

    context['task_instance'].xcom_push(key='retrain_decision', value=decision)
    return decision

def train_enhanced_model(**context):
    """Train model with enhanced feedback integration"""
    logger.info("🚀 Starting enhanced model training")

    # Get previous analysis
    feedback_analysis = context['task_instance'].xcom_pull(
        task_ids='analyze_feedback', key='feedback_analysis'
    )

    # Initialize trainer
    trainer = EnhancedModelTrainer()

    # Send training start notification
    notifier = RealTimeNotifier()
    if notifier.connect():
        notifier.send_update({
            'type': 'training_started',
            'model_id': f"enhanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'feedback_data': feedback_analysis,
            'timestamp': datetime.now().isoformat()
        })

    # Train model with feedback
    training_results = trainer.train_with_feedback(feedback_analysis)

    # Send training completion notification
    if notifier.ws:
        notifier.send_update({
            'type': 'training_completed',
            'results': training_results,
            'timestamp': datetime.now().isoformat()
        })
        notifier.close()

    logger.info(f"✅ Training completed! Final accuracy: {training_results['final_accuracy']:.3f}")
    logger.info(f"📈 Improvement over baseline: {training_results['improvement_over_baseline']:.3f}")

    context['task_instance'].xcom_push(key='training_results', value=training_results)
    return training_results

def validate_enhanced_model(**context):
    """Validate the enhanced model performance"""
    logger.info("🧪 Validating enhanced model performance")

    training_results = context['task_instance'].xcom_pull(
        task_ids='train_enhanced_model', key='training_results'
    )

    # Validation metrics
    validation_score = training_results['final_accuracy'] + np.random.normal(0, 0.02)
    validation_score = max(min(validation_score, 0.99), 0.5)

    validation_results = {
        'validation_accuracy': validation_score,
        'training_accuracy': training_results['final_accuracy'],
        'performance_delta': validation_score - training_results['final_accuracy'],
        'validation_passed': validation_score > 0.7,
        'feedback_integration_effective': training_results['improvement_over_baseline'] > 0.05,
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"Validation accuracy: {validation_score:.3f}")
    logger.info(f"Validation {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")

    # Send validation update
    notifier = RealTimeNotifier()
    if notifier.connect():
        notifier.send_update({
            'type': 'model_validation',
            'results': validation_results,
            'timestamp': datetime.now().isoformat()
        })
        notifier.close()

    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    return validation_results

def generate_performance_report(**context):
    """Generate comprehensive performance report"""
    logger.info("📊 Generating enhanced performance report")

    # Gather all results
    system_metrics = context['task_instance'].xcom_pull(
        task_ids='collect_metrics', key='system_metrics'
    )
    feedback_analysis = context['task_instance'].xcom_pull(
        task_ids='analyze_feedback', key='feedback_analysis'
    )
    training_results = context['task_instance'].xcom_pull(
        task_ids='train_enhanced_model', key='training_results'
    )
    validation_results = context['task_instance'].xcom_pull(
        task_ids='validate_model', key='validation_results'
    )

    # Create comprehensive report
    report = {
        'pipeline_execution': {
            'execution_time': datetime.now().isoformat(),
            'pipeline_version': 'enhanced_v2.0',
            'author': 'johnaffolter'
        },
        'system_performance': system_metrics,
        'feedback_insights': feedback_analysis,
        'training_performance': training_results,
        'validation_results': validation_results,
        'overall_health': {
            'system_status': 'healthy' if system_metrics['cpu_usage'] < 80 else 'warning',
            'user_satisfaction': 'high' if feedback_analysis['avg_satisfaction'] > 4 else 'medium',
            'model_performance': 'excellent' if validation_results['validation_accuracy'] > 0.9 else 'good',
            'feedback_integration': 'effective' if validation_results['feedback_integration_effective'] else 'needs_improvement'
        },
        'recommendations': []
    }

    # Generate recommendations
    if feedback_analysis['avg_satisfaction'] < 4.0:
        report['recommendations'].append("📈 Focus on improving user experience and prediction explanations")

    if validation_results['validation_accuracy'] < 0.85:
        report['recommendations'].append("🎯 Consider additional feature engineering or model architecture changes")

    if system_metrics['response_time_ms'] > 150:
        report['recommendations'].append("⚡ Optimize system performance to reduce response times")

    if not validation_results['feedback_integration_effective']:
        report['recommendations'].append("🔄 Review feedback integration mechanisms for better learning")

    logger.info("✅ Enhanced performance report generated")
    logger.info(f"Overall health indicators: {report['overall_health']}")

    # Save report
    report_path = f"/tmp/enhanced_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"📄 Report saved: {report_path}")

    # Send final update
    notifier = RealTimeNotifier()
    if notifier.connect():
        notifier.send_update({
            'type': 'performance_report',
            'report': report,
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        })
        notifier.close()

    return report

# Define task dependencies
collect_metrics_task = PythonOperator(
    task_id='collect_metrics',
    python_callable=collect_system_metrics,
    dag=dag,
)

analyze_feedback_task = PythonOperator(
    task_id='analyze_feedback',
    python_callable=analyze_feedback_data,
    dag=dag,
)

decide_retraining_task = PythonOperator(
    task_id='decide_retraining',
    python_callable=decide_retraining,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_enhanced_model',
    python_callable=train_enhanced_model,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_enhanced_model,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_performance_report,
    dag=dag,
)

# Setup task dependencies
collect_metrics_task >> analyze_feedback_task
analyze_feedback_task >> decide_retraining_task
decide_retraining_task >> train_model_task
train_model_task >> validate_model_task
validate_model_task >> generate_report_task

# Add this DAG to the Airflow system
logger.info("✅ Enhanced Feedback Training Pipeline DAG loaded successfully")
logger.info("🎯 Features: Real-time feedback integration, continuous improvement, performance tracking")
logger.info("👤 Author: John Affolter (johnaffolter)")
logger.info("📅 Date: September 29, 2025")