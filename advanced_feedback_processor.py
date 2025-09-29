#!/usr/bin/env python3
"""
Advanced ML Feedback Processor with Factory Pattern
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This system implements advanced feedback processing mechanisms using design patterns:
- Factory Pattern for creating different feedback processors
- Strategy Pattern for various feedback analysis strategies
- Observer Pattern for real-time feedback notifications
- Chain of Responsibility for feedback validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import threading
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feedback Types Enum
class FeedbackType(Enum):
    CORRECTION = "correction"
    VALIDATION = "validation"
    IMPROVEMENT = "improvement"
    EXPLANATION_REQUEST = "explanation_request"
    CONFIDENCE_RATING = "confidence_rating"
    FEATURE_REQUEST = "feature_request"

# Feedback Priority Levels
class FeedbackPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class FeedbackData:
    """Enhanced feedback data structure"""
    feedback_id: str
    user_id: str
    model_id: str
    prediction_id: str
    feedback_type: FeedbackType
    priority: FeedbackPriority
    original_prediction: Any
    corrected_prediction: Optional[Any] = None
    confidence_score: float = 0.0
    explanation_request: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_status: str = "pending"
    impact_score: float = 0.0

@dataclass
class ProcessingResult:
    """Result of feedback processing"""
    feedback_id: str
    processing_successful: bool
    improvement_suggestions: List[str]
    model_updates_needed: bool
    retraining_recommended: bool
    confidence_adjustment: float
    error_patterns_detected: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# Abstract Base Classes

class BaseFeedbackProcessor(ABC):
    """Abstract base class for feedback processors"""

    @abstractmethod
    def process_feedback(self, feedback: FeedbackData) -> ProcessingResult:
        """Process feedback and return results"""
        pass

    @abstractmethod
    def get_processor_type(self) -> str:
        """Return the type of this processor"""
        pass

    @abstractmethod
    def can_handle(self, feedback: FeedbackData) -> bool:
        """Check if this processor can handle the given feedback"""
        pass

class BaseFeedbackAnalyzer(ABC):
    """Abstract base class for feedback analyzers"""

    @abstractmethod
    def analyze_patterns(self, feedback_batch: List[FeedbackData]) -> Dict[str, Any]:
        """Analyze patterns in a batch of feedback"""
        pass

class BaseFeedbackValidator(ABC):
    """Abstract base class for feedback validators"""

    @abstractmethod
    def validate_feedback(self, feedback: FeedbackData) -> bool:
        """Validate feedback data"""
        pass

    @abstractmethod
    def get_validation_errors(self, feedback: FeedbackData) -> List[str]:
        """Get validation errors for feedback"""
        pass

# Concrete Feedback Processors

class CorrectionFeedbackProcessor(BaseFeedbackProcessor):
    """Processor for correction feedback using strategy pattern"""

    def __init__(self):
        self.error_patterns = defaultdict(int)
        self.correction_history = []

    def process_feedback(self, feedback: FeedbackData) -> ProcessingResult:
        """Process correction feedback"""
        start_time = time.time()

        logger.info(f"🔧 Processing correction feedback: {feedback.feedback_id}")

        # Analyze the correction
        improvement_suggestions = []
        error_patterns = []

        if feedback.corrected_prediction:
            # Compare original vs corrected
            correction_analysis = self._analyze_correction(
                feedback.original_prediction,
                feedback.corrected_prediction
            )

            improvement_suggestions.extend(correction_analysis['suggestions'])
            error_patterns.extend(correction_analysis['patterns'])

            # Track error patterns
            for pattern in error_patterns:
                self.error_patterns[pattern] += 1

        # Determine if retraining is needed
        retraining_recommended = (
            feedback.priority == FeedbackPriority.CRITICAL or
            len(error_patterns) > 2 or
            feedback.confidence_score < 0.3
        )

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            feedback_id=feedback.feedback_id,
            processing_successful=True,
            improvement_suggestions=improvement_suggestions,
            model_updates_needed=True,
            retraining_recommended=retraining_recommended,
            confidence_adjustment=-0.1,  # Reduce confidence for incorrect predictions
            error_patterns_detected=error_patterns,
            processing_time_ms=processing_time,
            metadata={
                'correction_complexity': len(str(feedback.corrected_prediction)),
                'error_frequency': sum(self.error_patterns.values())
            }
        )

    def _analyze_correction(self, original: Any, corrected: Any) -> Dict[str, Any]:
        """Analyze the nature of the correction"""
        suggestions = []
        patterns = []

        # Simple analysis (would be more sophisticated in practice)
        if isinstance(original, str) and isinstance(corrected, str):
            if len(corrected) > len(original) * 1.5:
                patterns.append("insufficient_detail")
                suggestions.append("📝 Model needs to provide more detailed responses")
            elif len(corrected) < len(original) * 0.5:
                patterns.append("excessive_verbosity")
                suggestions.append("✂️ Model responses could be more concise")

            # Check for specific error types
            if "not" in corrected.lower() and "not" not in original.lower():
                patterns.append("negation_error")
                suggestions.append("⚠️ Review negation handling in model")

        return {
            'suggestions': suggestions,
            'patterns': patterns
        }

    def get_processor_type(self) -> str:
        return "correction_processor"

    def can_handle(self, feedback: FeedbackData) -> bool:
        return feedback.feedback_type == FeedbackType.CORRECTION

class ValidationFeedbackProcessor(BaseFeedbackProcessor):
    """Processor for validation feedback"""

    def __init__(self):
        self.validation_history = []
        self.confidence_tracker = defaultdict(list)

    def process_feedback(self, feedback: FeedbackData) -> ProcessingResult:
        """Process validation feedback"""
        start_time = time.time()

        logger.info(f"✅ Processing validation feedback: {feedback.feedback_id}")

        # Track confidence scores
        self.confidence_tracker[feedback.model_id].append(feedback.confidence_score)

        improvement_suggestions = []
        model_updates_needed = False

        # Analyze confidence trends
        if len(self.confidence_tracker[feedback.model_id]) >= 5:
            recent_scores = self.confidence_tracker[feedback.model_id][-5:]
            avg_confidence = np.mean(recent_scores)

            if avg_confidence < 0.6:
                improvement_suggestions.append("📊 Model confidence is declining - review recent changes")
                model_updates_needed = True
            elif avg_confidence > 0.9:
                improvement_suggestions.append("🎯 Excellent model performance - consider expanding to new domains")

        # Determine confidence adjustment
        confidence_adjustment = 0.0
        if feedback.confidence_score > 0.8:
            confidence_adjustment = 0.05  # Boost confidence for validated predictions
        elif feedback.confidence_score < 0.4:
            confidence_adjustment = -0.02  # Lower confidence for poorly validated predictions

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            feedback_id=feedback.feedback_id,
            processing_successful=True,
            improvement_suggestions=improvement_suggestions,
            model_updates_needed=model_updates_needed,
            retraining_recommended=False,
            confidence_adjustment=confidence_adjustment,
            error_patterns_detected=[],
            processing_time_ms=processing_time,
            metadata={
                'validation_trend': 'positive' if feedback.confidence_score > 0.7 else 'negative',
                'confidence_history_length': len(self.confidence_tracker[feedback.model_id])
            }
        )

    def get_processor_type(self) -> str:
        return "validation_processor"

    def can_handle(self, feedback: FeedbackData) -> bool:
        return feedback.feedback_type == FeedbackType.VALIDATION

class ImprovementFeedbackProcessor(BaseFeedbackProcessor):
    """Processor for improvement suggestions"""

    def __init__(self):
        self.improvement_categories = defaultdict(int)

    def process_feedback(self, feedback: FeedbackData) -> ProcessingResult:
        """Process improvement feedback"""
        start_time = time.time()

        logger.info(f"💡 Processing improvement feedback: {feedback.feedback_id}")

        improvement_suggestions = []

        # Categorize improvement suggestions
        if 'explanation' in str(feedback.metadata).lower():
            self.improvement_categories['explanation'] += 1
            improvement_suggestions.append("🔍 Enhance model explainability features")

        if 'speed' in str(feedback.metadata).lower():
            self.improvement_categories['performance'] += 1
            improvement_suggestions.append("⚡ Optimize model inference speed")

        if 'accuracy' in str(feedback.metadata).lower():
            self.improvement_categories['accuracy'] += 1
            improvement_suggestions.append("🎯 Focus on accuracy improvements in training")

        # Determine if model updates are needed based on suggestion frequency
        model_updates_needed = any(count > 3 for count in self.improvement_categories.values())

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            feedback_id=feedback.feedback_id,
            processing_successful=True,
            improvement_suggestions=improvement_suggestions,
            model_updates_needed=model_updates_needed,
            retraining_recommended=model_updates_needed,
            confidence_adjustment=0.0,
            error_patterns_detected=[],
            processing_time_ms=processing_time,
            metadata={
                'improvement_categories': dict(self.improvement_categories),
                'suggestion_priority': feedback.priority.value
            }
        )

    def get_processor_type(self) -> str:
        return "improvement_processor"

    def can_handle(self, feedback: FeedbackData) -> bool:
        return feedback.feedback_type == FeedbackType.IMPROVEMENT

# Factory for Feedback Processors

class FeedbackProcessorFactory:
    """Factory for creating feedback processors using the Factory Pattern"""

    _processors = {}

    @classmethod
    def register_processor(cls, feedback_type: FeedbackType, processor_class):
        """Register a processor for a feedback type"""
        cls._processors[feedback_type] = processor_class

    @classmethod
    def create_processor(cls, feedback_type: FeedbackType) -> BaseFeedbackProcessor:
        """Create a processor for the given feedback type"""
        processor_class = cls._processors.get(feedback_type)

        if not processor_class:
            raise ValueError(f"No processor registered for feedback type: {feedback_type}")

        return processor_class()

    @classmethod
    def get_available_processors(cls) -> List[FeedbackType]:
        """Get list of available processor types"""
        return list(cls._processors.keys())

# Register processors
FeedbackProcessorFactory.register_processor(FeedbackType.CORRECTION, CorrectionFeedbackProcessor)
FeedbackProcessorFactory.register_processor(FeedbackType.VALIDATION, ValidationFeedbackProcessor)
FeedbackProcessorFactory.register_processor(FeedbackType.IMPROVEMENT, ImprovementFeedbackProcessor)

# Validation Chain

class FeedbackValidationChain:
    """Chain of Responsibility pattern for feedback validation"""

    def __init__(self):
        self.validators = []

    def add_validator(self, validator: BaseFeedbackValidator):
        """Add a validator to the chain"""
        self.validators.append(validator)

    def validate(self, feedback: FeedbackData) -> tuple[bool, List[str]]:
        """Validate feedback through the chain"""
        all_errors = []

        for validator in self.validators:
            is_valid = validator.validate_feedback(feedback)
            if not is_valid:
                errors = validator.get_validation_errors(feedback)
                all_errors.extend(errors)

        return len(all_errors) == 0, all_errors

class BasicFeedbackValidator(BaseFeedbackValidator):
    """Basic feedback validator"""

    def validate_feedback(self, feedback: FeedbackData) -> bool:
        """Validate basic feedback requirements"""
        return (
            feedback.feedback_id and
            feedback.user_id and
            feedback.model_id and
            feedback.prediction_id and
            feedback.feedback_type in FeedbackType
        )

    def get_validation_errors(self, feedback: FeedbackData) -> List[str]:
        """Get validation errors"""
        errors = []

        if not feedback.feedback_id:
            errors.append("Missing feedback_id")
        if not feedback.user_id:
            errors.append("Missing user_id")
        if not feedback.model_id:
            errors.append("Missing model_id")
        if not feedback.prediction_id:
            errors.append("Missing prediction_id")
        if feedback.feedback_type not in FeedbackType:
            errors.append("Invalid feedback_type")

        return errors

# Observer Pattern for Real-time Notifications

class FeedbackObserver(ABC):
    """Abstract observer for feedback events"""

    @abstractmethod
    def on_feedback_processed(self, feedback: FeedbackData, result: ProcessingResult):
        """Called when feedback is processed"""
        pass

    @abstractmethod
    def on_pattern_detected(self, pattern: str, frequency: int):
        """Called when error pattern is detected"""
        pass

class RealTimeDashboardObserver(FeedbackObserver):
    """Observer that sends updates to real-time dashboard"""

    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        self.websocket_url = websocket_url

    def on_feedback_processed(self, feedback: FeedbackData, result: ProcessingResult):
        """Send feedback processing update"""
        update = {
            'type': 'feedback_processed',
            'feedback_id': feedback.feedback_id,
            'feedback_type': feedback.feedback_type.value,
            'processing_result': {
                'successful': result.processing_successful,
                'suggestions_count': len(result.improvement_suggestions),
                'retraining_recommended': result.retraining_recommended,
                'processing_time_ms': result.processing_time_ms
            },
            'timestamp': datetime.now().isoformat()
        }

        self._send_update(update)

    def on_pattern_detected(self, pattern: str, frequency: int):
        """Send pattern detection update"""
        update = {
            'type': 'pattern_detected',
            'pattern': pattern,
            'frequency': frequency,
            'severity': 'high' if frequency > 5 else 'medium' if frequency > 2 else 'low',
            'timestamp': datetime.now().isoformat()
        }

        self._send_update(update)

    def _send_update(self, update: Dict[str, Any]):
        """Send update via WebSocket (mock implementation)"""
        logger.info(f"📡 Dashboard update: {update['type']}")

# Main Feedback Processing System

class AdvancedFeedbackProcessor:
    """Main system for processing feedback with advanced patterns"""

    def __init__(self):
        self.factory = FeedbackProcessorFactory()
        self.validation_chain = FeedbackValidationChain()
        self.observers = []
        self.feedback_queue = asyncio.Queue()
        self.processing_stats = defaultdict(int)

        # Setup validation chain
        self.validation_chain.add_validator(BasicFeedbackValidator())

        # Add observers
        self.add_observer(RealTimeDashboardObserver())

    def add_observer(self, observer: FeedbackObserver):
        """Add an observer to the system"""
        self.observers.append(observer)

    async def process_feedback_async(self, feedback: FeedbackData) -> ProcessingResult:
        """Asynchronously process feedback"""
        logger.info(f"🔄 Processing feedback: {feedback.feedback_id}")

        # Validate feedback
        is_valid, errors = self.validation_chain.validate(feedback)
        if not is_valid:
            logger.error(f"❌ Invalid feedback: {errors}")
            return ProcessingResult(
                feedback_id=feedback.feedback_id,
                processing_successful=False,
                improvement_suggestions=[],
                model_updates_needed=False,
                retraining_recommended=False,
                confidence_adjustment=0.0,
                error_patterns_detected=errors,
                processing_time_ms=0.0
            )

        # Get appropriate processor
        try:
            processor = self.factory.create_processor(feedback.feedback_type)
        except ValueError as e:
            logger.error(f"❌ No processor for feedback type: {e}")
            return ProcessingResult(
                feedback_id=feedback.feedback_id,
                processing_successful=False,
                improvement_suggestions=[],
                model_updates_needed=False,
                retraining_recommended=False,
                confidence_adjustment=0.0,
                error_patterns_detected=[str(e)],
                processing_time_ms=0.0
            )

        # Process feedback
        result = processor.process_feedback(feedback)

        # Update stats
        self.processing_stats[feedback.feedback_type.value] += 1
        self.processing_stats['total_processed'] += 1

        # Notify observers
        for observer in self.observers:
            observer.on_feedback_processed(feedback, result)

        # Check for patterns
        for pattern in result.error_patterns_detected:
            for observer in self.observers:
                observer.on_pattern_detected(pattern, 1)

        logger.info(f"✅ Feedback processed: {feedback.feedback_id}")
        return result

    async def start_processing_service(self):
        """Start the asynchronous feedback processing service"""
        logger.info("🚀 Starting Advanced Feedback Processing Service")

        while True:
            try:
                # Wait for feedback with timeout
                feedback = await asyncio.wait_for(self.feedback_queue.get(), timeout=1.0)
                result = await self.process_feedback_async(feedback)

                # Mark task as done
                self.feedback_queue.task_done()

            except asyncio.TimeoutError:
                # No feedback to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in processing service: {e}")

    async def submit_feedback(self, feedback: FeedbackData):
        """Submit feedback for processing"""
        await self.feedback_queue.put(feedback)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'stats': dict(self.processing_stats),
            'available_processors': [t.value for t in self.factory.get_available_processors()],
            'queue_size': self.feedback_queue.qsize(),
            'observer_count': len(self.observers)
        }

# Test and Demo Functions

async def demo_advanced_feedback_processing():
    """Demonstrate the advanced feedback processing system"""
    logger.info("🎯 Starting Advanced Feedback Processing Demo")

    # Initialize system
    processor = AdvancedFeedbackProcessor()

    # Start processing service in background
    asyncio.create_task(processor.start_processing_service())

    # Create sample feedback data
    sample_feedbacks = [
        FeedbackData(
            feedback_id="fb001",
            user_id="user123",
            model_id="enhanced_model_v1",
            prediction_id="pred456",
            feedback_type=FeedbackType.CORRECTION,
            priority=FeedbackPriority.HIGH,
            original_prediction="This is incorrect",
            corrected_prediction="This is the correct answer",
            confidence_score=0.2
        ),
        FeedbackData(
            feedback_id="fb002",
            user_id="user456",
            model_id="enhanced_model_v1",
            prediction_id="pred789",
            feedback_type=FeedbackType.VALIDATION,
            priority=FeedbackPriority.MEDIUM,
            original_prediction="Good prediction",
            confidence_score=0.9
        ),
        FeedbackData(
            feedback_id="fb003",
            user_id="user789",
            model_id="enhanced_model_v1",
            prediction_id="pred012",
            feedback_type=FeedbackType.IMPROVEMENT,
            priority=FeedbackPriority.LOW,
            original_prediction="Could be better",
            metadata={"suggestion": "Add more explanation details"}
        )
    ]

    # Process sample feedback
    for feedback in sample_feedbacks:
        await processor.submit_feedback(feedback)
        await asyncio.sleep(0.5)  # Small delay for demo

    # Wait for processing to complete
    await processor.feedback_queue.join()

    # Show processing stats
    stats = processor.get_processing_stats()
    logger.info(f"📊 Processing completed: {stats}")

    return processor

if __name__ == "__main__":
    print("🤖 Advanced ML Feedback Processor")
    print("Author: John Affolter (johnaffolter)")
    print("Date: September 29, 2025")
    print("=" * 60)
    print()
    print("🌟 Design Patterns Implemented:")
    print("- Factory Pattern: FeedbackProcessorFactory")
    print("- Strategy Pattern: Different processing strategies")
    print("- Observer Pattern: Real-time notifications")
    print("- Chain of Responsibility: Validation chain")
    print()
    print("🎯 Features:")
    print("- Asynchronous feedback processing")
    print("- Real-time pattern detection")
    print("- Configurable validation chains")
    print("- Extensible processor types")
    print("- Performance tracking and analytics")
    print()

    # Run demo
    asyncio.run(demo_advanced_feedback_processing())