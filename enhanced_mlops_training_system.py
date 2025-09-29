#!/usr/bin/env python3
"""
Enhanced MLOps Training System with Feedback Loops and Real-time Tracking
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This system provides advanced training integration with:
- Real-time performance tracking
- Automatic feedback loops
- Interactive model improvement
- Continuous learning from user interactions
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import websockets
import sqlite3
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Enhanced training metrics with feedback tracking"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    learning_rate: float
    epoch: int
    user_feedback_score: Optional[float] = None
    correction_count: int = 0
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class UserFeedback:
    """User feedback for continuous improvement"""
    prediction_id: str
    original_prediction: str
    corrected_prediction: Optional[str]
    confidence_rating: int  # 1-5 scale
    feedback_type: str  # 'correction', 'validation', 'improvement'
    user_id: str
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ModelPerformanceSnapshot:
    """Real-time model performance snapshot"""
    model_id: str
    predictions_count: int
    accuracy_trend: List[float]
    user_satisfaction: float
    feedback_volume: int
    improvement_suggestions: List[str]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class FeedbackDatabase:
    """Database for storing and analyzing feedback"""

    def __init__(self, db_path: str = "mlops_feedback.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Training metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                loss REAL,
                learning_rate REAL,
                epoch INTEGER,
                user_feedback_score REAL,
                correction_count INTEGER,
                timestamp TEXT
            )
        ''')

        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                original_prediction TEXT,
                corrected_prediction TEXT,
                confidence_rating INTEGER,
                feedback_type TEXT,
                user_id TEXT,
                timestamp TEXT
            )
        ''')

        # Model performance snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                predictions_count INTEGER,
                accuracy_trend TEXT,
                user_satisfaction REAL,
                feedback_volume INTEGER,
                improvement_suggestions TEXT,
                timestamp TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_training_metrics(self, metrics: TrainingMetrics, model_id: str):
        """Store training metrics with feedback correlation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO training_metrics
            (model_id, accuracy, precision, recall, f1_score, loss, learning_rate,
             epoch, user_feedback_score, correction_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id, metrics.accuracy, metrics.precision, metrics.recall,
            metrics.f1_score, metrics.loss, metrics.learning_rate, metrics.epoch,
            metrics.user_feedback_score, metrics.correction_count, metrics.timestamp
        ))

        conn.commit()
        conn.close()

    def store_user_feedback(self, feedback: UserFeedback):
        """Store user feedback for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_feedback
            (prediction_id, original_prediction, corrected_prediction,
             confidence_rating, feedback_type, user_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.prediction_id, feedback.original_prediction,
            feedback.corrected_prediction, feedback.confidence_rating,
            feedback.feedback_type, feedback.user_id, feedback.timestamp
        ))

        conn.commit()
        conn.close()

    def get_feedback_analytics(self, model_id: str, days: int = 7) -> Dict:
        """Get comprehensive feedback analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent feedback trends
        cursor.execute('''
            SELECT AVG(confidence_rating), COUNT(*), feedback_type
            FROM user_feedback
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY feedback_type
        '''.format(days))

        feedback_summary = cursor.fetchall()

        # Get model performance trend
        cursor.execute('''
            SELECT accuracy, user_feedback_score, timestamp
            FROM training_metrics
            WHERE model_id = ? AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days), (model_id,))

        performance_trend = cursor.fetchall()

        conn.close()

        return {
            'feedback_summary': feedback_summary,
            'performance_trend': performance_trend,
            'total_feedback_count': sum([row[1] for row in feedback_summary])
        }

class EnhancedTrainingMonitor:
    """Enhanced training monitor with real-time feedback integration"""

    def __init__(self):
        self.db = FeedbackDatabase()
        self.active_connections = set()
        self.training_state = {}
        self.feedback_queue = asyncio.Queue()
        self.performance_cache = {}

    async def start_monitoring(self):
        """Start the enhanced monitoring system"""
        logger.info("🚀 Starting Enhanced MLOps Training Monitor")

        # Start WebSocket server for real-time updates
        server = await websockets.serve(
            self.handle_websocket_connection,
            "localhost",
            8765
        )

        # Start background tasks
        asyncio.create_task(self.process_feedback_queue())
        asyncio.create_task(self.update_performance_metrics())
        asyncio.create_task(self.generate_improvement_suggestions())

        logger.info("✅ Enhanced monitoring system started on ws://localhost:8765")
        await server.wait_closed()

    async def handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections for real-time updates"""
        self.active_connections.add(websocket)
        logger.info(f"🔗 New connection: {websocket.remote_address}")

        try:
            await websocket.send(json.dumps({
                "type": "connection_established",
                "message": "Connected to Enhanced MLOps Training Monitor",
                "timestamp": datetime.now().isoformat()
            }))

            async for message in websocket:
                await self.process_websocket_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Connection closed: {websocket.remote_address}")
        finally:
            self.active_connections.discard(websocket)

    async def process_websocket_message(self, websocket, message):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'training_update':
                await self.handle_training_update(data)
            elif message_type == 'user_feedback':
                await self.handle_user_feedback(data)
            elif message_type == 'request_analytics':
                await self.send_analytics(websocket, data)
            elif message_type == 'start_training':
                await self.start_enhanced_training(data)

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON format"
            }))

    async def handle_training_update(self, data):
        """Handle training progress updates with enhanced metrics"""
        model_id = data.get('model_id', 'default')
        metrics_data = data.get('metrics', {})

        # Create enhanced metrics object
        metrics = TrainingMetrics(
            accuracy=metrics_data.get('accuracy', 0.0),
            precision=metrics_data.get('precision', 0.0),
            recall=metrics_data.get('recall', 0.0),
            f1_score=metrics_data.get('f1_score', 0.0),
            loss=metrics_data.get('loss', 0.0),
            learning_rate=metrics_data.get('learning_rate', 0.001),
            epoch=metrics_data.get('epoch', 0),
            user_feedback_score=metrics_data.get('user_feedback_score'),
            correction_count=metrics_data.get('correction_count', 0)
        )

        # Store metrics
        self.db.store_training_metrics(metrics, model_id)

        # Update training state
        self.training_state[model_id] = metrics

        # Broadcast update to all connected clients
        await self.broadcast_update({
            "type": "training_progress",
            "model_id": model_id,
            "metrics": asdict(metrics),
            "improvement_suggestions": await self.get_improvement_suggestions(model_id)
        })

    async def handle_user_feedback(self, data):
        """Handle user feedback with immediate integration"""
        feedback = UserFeedback(
            prediction_id=data.get('prediction_id'),
            original_prediction=data.get('original_prediction'),
            corrected_prediction=data.get('corrected_prediction'),
            confidence_rating=data.get('confidence_rating', 3),
            feedback_type=data.get('feedback_type', 'validation'),
            user_id=data.get('user_id', 'anonymous')
        )

        # Store feedback
        self.db.store_user_feedback(feedback)

        # Add to processing queue
        await self.feedback_queue.put(feedback)

        # Immediate response to user
        await self.broadcast_update({
            "type": "feedback_received",
            "message": "Feedback received and will be integrated into training",
            "feedback_id": feedback.prediction_id,
            "impact_score": await self.calculate_feedback_impact(feedback)
        })

    async def process_feedback_queue(self):
        """Process feedback queue for continuous learning"""
        while True:
            try:
                feedback = await asyncio.wait_for(self.feedback_queue.get(), timeout=1.0)

                # Analyze feedback impact
                impact = await self.analyze_feedback_impact(feedback)

                # Update model suggestions
                suggestions = await self.generate_feedback_suggestions(feedback)

                # Broadcast insights
                await self.broadcast_update({
                    "type": "feedback_insights",
                    "feedback_type": feedback.feedback_type,
                    "impact_analysis": impact,
                    "suggestions": suggestions,
                    "timestamp": datetime.now().isoformat()
                })

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")

    async def update_performance_metrics(self):
        """Continuously update performance metrics"""
        while True:
            try:
                for model_id in self.training_state:
                    # Generate performance snapshot
                    snapshot = await self.create_performance_snapshot(model_id)

                    # Cache for quick access
                    self.performance_cache[model_id] = snapshot

                    # Broadcast to clients
                    await self.broadcast_update({
                        "type": "performance_snapshot",
                        "model_id": model_id,
                        "snapshot": asdict(snapshot)
                    })

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(60)

    async def create_performance_snapshot(self, model_id: str) -> ModelPerformanceSnapshot:
        """Create real-time performance snapshot"""
        analytics = self.db.get_feedback_analytics(model_id)

        # Calculate trends
        accuracy_trend = []
        for row in analytics['performance_trend'][-10:]:  # Last 10 data points
            accuracy_trend.append(row[0])

        # Calculate user satisfaction
        feedback_scores = [row[0] for row in analytics['feedback_summary']]
        user_satisfaction = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0

        # Generate improvement suggestions
        suggestions = await self.get_improvement_suggestions(model_id)

        return ModelPerformanceSnapshot(
            model_id=model_id,
            predictions_count=analytics.get('total_feedback_count', 0),
            accuracy_trend=accuracy_trend,
            user_satisfaction=user_satisfaction,
            feedback_volume=analytics.get('total_feedback_count', 0),
            improvement_suggestions=suggestions
        )

    async def get_improvement_suggestions(self, model_id: str) -> List[str]:
        """Generate AI-powered improvement suggestions"""
        suggestions = []

        if model_id in self.training_state:
            metrics = self.training_state[model_id]

            # Accuracy-based suggestions
            if metrics.accuracy < 0.8:
                suggestions.append("🎯 Consider increasing training epochs or adjusting learning rate")

            # Precision/Recall balance
            if metrics.precision > metrics.recall + 0.1:
                suggestions.append("⚖️ Model favors precision - consider class balancing")
            elif metrics.recall > metrics.precision + 0.1:
                suggestions.append("⚖️ Model favors recall - consider threshold tuning")

            # Loss trends
            if metrics.loss > 0.5:
                suggestions.append("📉 High loss detected - review feature engineering")

            # User feedback integration
            if metrics.user_feedback_score and metrics.user_feedback_score < 3.0:
                suggestions.append("👥 Low user satisfaction - review prediction explanations")

        return suggestions

    async def generate_improvement_suggestions(self):
        """Background task for generating continuous improvement suggestions"""
        while True:
            try:
                for model_id in self.training_state:
                    suggestions = await self.get_improvement_suggestions(model_id)

                    if suggestions:
                        await self.broadcast_update({
                            "type": "improvement_suggestions",
                            "model_id": model_id,
                            "suggestions": suggestions,
                            "priority": "medium",
                            "timestamp": datetime.now().isoformat()
                        })

                await asyncio.sleep(300)  # Generate suggestions every 5 minutes

            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")
                await asyncio.sleep(600)

    async def start_enhanced_training(self, config):
        """Start training with enhanced monitoring"""
        model_id = config.get('model_id', f'model_{int(time.time())}')

        logger.info(f"🚀 Starting enhanced training for model: {model_id}")

        # Simulate training with real-time feedback
        for epoch in range(config.get('epochs', 10)):
            # Simulate training metrics with some randomness
            import random

            base_accuracy = 0.6 + (epoch * 0.03) + random.uniform(-0.02, 0.02)
            metrics = TrainingMetrics(
                accuracy=min(base_accuracy, 0.95),
                precision=base_accuracy + random.uniform(-0.05, 0.05),
                recall=base_accuracy + random.uniform(-0.05, 0.05),
                f1_score=base_accuracy + random.uniform(-0.03, 0.03),
                loss=max(0.8 - (epoch * 0.08), 0.05) + random.uniform(-0.02, 0.02),
                learning_rate=config.get('learning_rate', 0.001),
                epoch=epoch + 1,
                correction_count=random.randint(0, 5)
            )

            # Send training update
            await self.handle_training_update({
                'model_id': model_id,
                'metrics': asdict(metrics)
            })

            await asyncio.sleep(2)  # Simulate training time

        logger.info(f"✅ Training completed for model: {model_id}")

        await self.broadcast_update({
            "type": "training_completed",
            "model_id": model_id,
            "final_metrics": asdict(metrics),
            "summary": f"Training completed with {metrics.accuracy:.3f} accuracy"
        })

    async def broadcast_update(self, message):
        """Broadcast update to all connected clients"""
        if self.active_connections:
            dead_connections = set()

            for websocket in self.active_connections:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    dead_connections.add(websocket)

            # Remove dead connections
            self.active_connections -= dead_connections

    async def calculate_feedback_impact(self, feedback: UserFeedback) -> float:
        """Calculate the impact score of user feedback"""
        base_impact = feedback.confidence_rating / 5.0

        # Weight by feedback type
        type_weights = {
            'correction': 1.0,
            'validation': 0.7,
            'improvement': 0.8
        }

        return base_impact * type_weights.get(feedback.feedback_type, 0.5)

    async def analyze_feedback_impact(self, feedback: UserFeedback) -> Dict:
        """Analyze the broader impact of feedback"""
        return {
            "immediate_impact": await self.calculate_feedback_impact(feedback),
            "pattern_detected": feedback.feedback_type == 'correction',
            "suggested_action": "Retrain model" if feedback.confidence_rating <= 2 else "Monitor trends",
            "confidence": feedback.confidence_rating / 5.0
        }

    async def generate_feedback_suggestions(self, feedback: UserFeedback) -> List[str]:
        """Generate suggestions based on feedback"""
        suggestions = []

        if feedback.feedback_type == 'correction':
            suggestions.append("🔄 Consider retraining with corrected examples")
            suggestions.append("📊 Analyze similar cases for systematic errors")

        if feedback.confidence_rating <= 2:
            suggestions.append("⚠️ Low confidence detected - review model predictions")
            suggestions.append("🎯 Focus on improving this prediction category")

        return suggestions

    async def send_analytics(self, websocket, request):
        """Send comprehensive analytics to requesting client"""
        model_id = request.get('model_id', 'default')
        analytics = self.db.get_feedback_analytics(model_id)

        response = {
            "type": "analytics_response",
            "model_id": model_id,
            "analytics": analytics,
            "performance_cache": self.performance_cache.get(model_id),
            "active_training": model_id in self.training_state,
            "timestamp": datetime.now().isoformat()
        }

        await websocket.send(json.dumps(response))

async def main():
    """Main function to start the enhanced training system"""
    monitor = EnhancedTrainingMonitor()
    await monitor.start_monitoring()

if __name__ == "__main__":
    print("🚀 Enhanced MLOps Training System")
    print("Author: John Affolter (johnaffolter)")
    print("Date: September 29, 2025")
    print("=" * 60)
    print()
    print("🌟 Features:")
    print("- Real-time training monitoring with WebSocket updates")
    print("- Continuous user feedback integration")
    print("- AI-powered improvement suggestions")
    print("- Performance trend analysis")
    print("- Interactive model optimization")
    print()
    print("🔗 Connect to: ws://localhost:8765")
    print("📊 Database: mlops_feedback.db")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Enhanced training system stopped")