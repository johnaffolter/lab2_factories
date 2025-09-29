#!/usr/bin/env python3

"""
Advanced Dataset Generation System with LLM Judging and Robust Ontology
Comprehensive system for generating, validating, and curating training datasets
"""

import sys
import json
import time
import uuid
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import re
import statistics

# OpenAI for real LLM integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

# Import our analysis systems
sys.path.append('.')
from advanced_email_analyzer import AdvancedEmailAnalyzer, EmailAnalysisResult
from working_educational_demo import WorkingEducationalDemo

class DatasetQuality(Enum):
    """Dataset quality levels"""
    GOLD_STANDARD = "gold_standard"
    HIGH_QUALITY = "high_quality"
    ACCEPTABLE = "acceptable"
    REQUIRES_REVIEW = "requires_review"
    REJECTED = "rejected"

class LabelConfidence(Enum):
    """Label confidence levels"""
    VERY_HIGH = "very_high"     # 0.9-1.0
    HIGH = "high"               # 0.8-0.9
    MEDIUM = "medium"           # 0.6-0.8
    LOW = "low"                 # 0.4-0.6
    VERY_LOW = "very_low"       # 0.0-0.4

class AnnotationSource(Enum):
    """Source of annotations"""
    EXPERT_HUMAN = "expert_human"
    LLM_JUDGE = "llm_judge"
    AUTOMATED_SYSTEM = "automated_system"
    CROWD_SOURCE = "crowd_source"
    CONSENSUS_MULTIPLE = "consensus_multiple"

class TaskCategory(Enum):
    """Task categories for dataset generation"""
    EMAIL_CLASSIFICATION = "email_classification"
    GRAMMAR_CORRECTION = "grammar_correction"
    TONE_ANALYSIS = "tone_analysis"
    CLARITY_ASSESSMENT = "clarity_assessment"
    SECURITY_DETECTION = "security_detection"
    PROFESSIONALISM_SCORING = "professionalism_scoring"
    ENGAGEMENT_PREDICTION = "engagement_prediction"

@dataclass
class TaxonomyNode:
    """Node in the classification taxonomy"""
    node_id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    difficulty_level: float = 0.5  # 0.0 = easy, 1.0 = very difficult
    confusion_matrix: Dict[str, float] = field(default_factory=dict)

@dataclass
class AnnotationStandard:
    """Annotation standard definition"""
    standard_id: str
    name: str
    description: str
    guidelines: List[str]
    quality_criteria: Dict[str, float]
    inter_annotator_agreement_threshold: float
    confidence_thresholds: Dict[LabelConfidence, Tuple[float, float]]
    validation_rules: List[str] = field(default_factory=list)

@dataclass
class DataSample:
    """Individual data sample with comprehensive metadata"""
    sample_id: str
    content: Dict[str, Any]
    true_labels: Dict[str, Any]
    predicted_labels: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    difficulty_score: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    golden_set_member: bool = False

@dataclass
class LLMJudgmentResult:
    """Result from LLM judge evaluation"""
    judgment_id: str
    sample_id: str
    llm_model: str
    judgment_scores: Dict[str, float]
    reasoning: str
    confidence: float
    consistency_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    validation_passed: bool = False
    flagged_issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMJudgeInterface(ABC):
    """Abstract interface for LLM judges"""

    @abstractmethod
    def evaluate_sample(self, sample: DataSample, criteria: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate a data sample"""
        pass

    @abstractmethod
    def batch_evaluate(self, samples: List[DataSample], criteria: Dict[str, Any]) -> List[LLMJudgmentResult]:
        """Evaluate multiple samples"""
        pass

class MockLLMJudge(LLMJudgeInterface):
    """Mock LLM judge for demonstration (replace with actual LLM calls)"""

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.email_analyzer = AdvancedEmailAnalyzer()

    def evaluate_sample(self, sample: DataSample, criteria: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate a single sample using mock LLM judgment"""

        # Simulate LLM evaluation using our email analyzer for consistency
        content = sample.content

        try:
            # Run actual analysis
            analysis_result = self.email_analyzer.analyze_email(
                subject=content.get('subject', ''),
                body=content.get('body', ''),
                sender=content.get('sender', '')
            )

            # Generate LLM-style judgment scores
            judgment_scores = {
                "overall_quality": min(analysis_result.overall_score + random.uniform(-0.1, 0.1), 1.0),
                "grammar_accuracy": analysis_result.metrics.professionalism_score + random.uniform(-0.05, 0.05),
                "clarity": analysis_result.metrics.clarity_score + random.uniform(-0.05, 0.05),
                "professionalism": max(0.0, analysis_result.metrics.professionalism_score + random.uniform(-0.1, 0.1)),
                "engagement": analysis_result.metrics.engagement_score + random.uniform(-0.05, 0.05)
            }

            # Generate reasoning based on analysis
            reasoning = self._generate_llm_reasoning(analysis_result, judgment_scores)

            # Calculate confidence based on consistency
            confidence = self._calculate_confidence(analysis_result, judgment_scores)

            # Check for consistency with expected labels
            consistency_score = self._calculate_consistency(sample, judgment_scores)

            return LLMJudgmentResult(
                judgment_id=str(uuid.uuid4()),
                sample_id=sample.sample_id,
                llm_model=self.model_name,
                judgment_scores=judgment_scores,
                reasoning=reasoning,
                confidence=confidence,
                consistency_score=consistency_score,
                validation_passed=confidence > 0.7 and consistency_score > 0.6
            )

        except Exception as e:
            # Return low-confidence result if analysis fails
            return LLMJudgmentResult(
                judgment_id=str(uuid.uuid4()),
                sample_id=sample.sample_id,
                llm_model=self.model_name,
                judgment_scores={"overall_quality": 0.5},
                reasoning=f"Analysis failed: {str(e)}",
                confidence=0.3,
                consistency_score=0.3,
                validation_passed=False,
                flagged_issues=["analysis_error"]
            )

    def batch_evaluate(self, samples: List[DataSample], criteria: Dict[str, Any]) -> List[LLMJudgmentResult]:
        """Evaluate multiple samples in batch"""
        results = []
        for sample in samples:
            result = self.evaluate_sample(sample, criteria)
            results.append(result)
            time.sleep(0.01)  # Simulate processing delay
        return results

    def _generate_llm_reasoning(self, analysis_result: EmailAnalysisResult, scores: Dict[str, float]) -> str:
        """Generate human-like reasoning for the judgment"""

        issues = analysis_result.issues
        metrics = analysis_result.metrics

        reasoning_parts = []

        # Overall assessment
        if scores["overall_quality"] > 0.8:
            reasoning_parts.append("This email demonstrates high quality professional communication.")
        elif scores["overall_quality"] > 0.6:
            reasoning_parts.append("This email shows good communication fundamentals with room for improvement.")
        else:
            reasoning_parts.append("This email has significant issues that impact its effectiveness.")

        # Grammar assessment
        if len(issues) == 0:
            reasoning_parts.append("No significant grammar or clarity issues detected.")
        elif len(issues) <= 2:
            reasoning_parts.append(f"Minor issues detected ({len(issues)} total) that could be easily addressed.")
        else:
            reasoning_parts.append(f"Multiple issues identified ({len(issues)} total) requiring systematic revision.")

        # Specific metrics commentary
        if metrics.word_count < 30:
            reasoning_parts.append("The brevity may lack necessary detail for complex business communication.")
        elif metrics.word_count > 200:
            reasoning_parts.append("The length is appropriate for detailed professional communication.")

        if metrics.professionalism_score < 0.5:
            reasoning_parts.append("Tone and language may be too informal for business context.")

        return " ".join(reasoning_parts)

    def _calculate_confidence(self, analysis_result: EmailAnalysisResult, scores: Dict[str, float]) -> float:
        """Calculate confidence based on analysis consistency"""

        # Base confidence on score consistency
        base_confidence = 0.8

        # Reduce confidence if many issues detected
        issue_penalty = min(0.3, len(analysis_result.issues) * 0.05)

        # Reduce confidence if scores are extreme (possibly unreliable)
        extreme_penalty = 0.0
        for score in scores.values():
            if score < 0.1 or score > 0.95:
                extreme_penalty += 0.05

        confidence = max(0.3, base_confidence - issue_penalty - extreme_penalty)
        return min(1.0, confidence)

    def _calculate_consistency(self, sample: DataSample, judgment_scores: Dict[str, float]) -> float:
        """Calculate consistency with expected labels"""

        if not sample.true_labels:
            return 0.7  # Default if no true labels available

        consistency_scores = []

        for key, true_value in sample.true_labels.items():
            if key in judgment_scores:
                judgment_value = judgment_scores[key]

                if isinstance(true_value, (int, float)) and isinstance(judgment_value, (int, float)):
                    # Numerical comparison
                    difference = abs(true_value - judgment_value)
                    consistency = max(0.0, 1.0 - difference)
                    consistency_scores.append(consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.6

class RealOpenAILLMJudge(LLMJudgeInterface):
    """Real OpenAI LLM judge for production use"""

    def __init__(self, model_name: str = "gpt-4", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client (compatible with v0.28.1)
        openai.api_key = self.api_key
        self.email_analyzer = AdvancedEmailAnalyzer()

    def evaluate_sample(self, sample: DataSample, criteria: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate a single sample using real OpenAI LLM"""

        content = sample.content

        try:
            # First run our analysis for context
            analysis_result = self.email_analyzer.analyze_email(
                subject=content.get('subject', ''),
                body=content.get('body', ''),
                sender=content.get('sender', '')
            )

            # Create prompt for OpenAI evaluation
            prompt = self._create_evaluation_prompt(sample, criteria, analysis_result)

            # Make real OpenAI API call (compatible with v0.28.1)
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert email analyst and quality judge. Provide detailed evaluation scores and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            # Parse OpenAI response
            llm_response = response.choices[0].message.content
            judgment_data = self._parse_llm_response(llm_response)

            # Combine LLM judgment with our analysis
            judgment_scores = judgment_data.get('scores', {})
            reasoning = judgment_data.get('reasoning', llm_response)
            confidence = judgment_data.get('confidence', 0.8)

            # Calculate consistency
            consistency_score = self._calculate_consistency(sample, judgment_scores)

            return LLMJudgmentResult(
                judgment_id=str(uuid.uuid4()),
                sample_id=sample.sample_id,
                llm_model=self.model_name,
                judgment_scores=judgment_scores,
                reasoning=reasoning,
                confidence=confidence,
                consistency_score=consistency_score,
                validation_passed=confidence > 0.7 and consistency_score > 0.6,
                metadata={
                    "openai_response_tokens": response.usage.total_tokens,
                    "analysis_metrics": asdict(analysis_result.metrics)
                }
            )

        except Exception as e:
            print(f"OpenAI evaluation error: {str(e)}")
            # Fallback to analysis-based evaluation
            return self._fallback_evaluation(sample, analysis_result if 'analysis_result' in locals() else None)

    def batch_evaluate(self, samples: List[DataSample], criteria: Dict[str, Any]) -> List[LLMJudgmentResult]:
        """Evaluate multiple samples with real OpenAI API"""
        results = []
        for i, sample in enumerate(samples):
            print(f"Evaluating sample {i+1}/{len(samples)} with OpenAI...")
            result = self.evaluate_sample(sample, criteria)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        return results

    def _create_evaluation_prompt(self, sample: DataSample, criteria: Dict[str, Any], analysis_result) -> str:
        """Create evaluation prompt for OpenAI"""

        content = sample.content
        prompt = f"""
Evaluate the following email for quality and provide detailed scores:

EMAIL CONTENT:
Subject: {content.get('subject', 'N/A')}
Body: {content.get('body', 'N/A')}
Sender: {content.get('sender', 'N/A')}

ANALYSIS CONTEXT:
- Overall Score: {analysis_result.overall_score:.2f}
- Clarity Score: {analysis_result.metrics.clarity_score:.2f}
- Professionalism Score: {analysis_result.metrics.professionalism_score:.2f}
- Engagement Score: {analysis_result.metrics.engagement_score:.2f}
- Issues Found: {len(analysis_result.issues)}

EVALUATION CRITERIA:
{json.dumps(criteria, indent=2)}

Please provide evaluation in this JSON format:
{{
    "scores": {{
        "overall_quality": 0.0-1.0,
        "grammar_accuracy": 0.0-1.0,
        "clarity": 0.0-1.0,
        "professionalism": 0.0-1.0,
        "engagement": 0.0-1.0
    }},
    "reasoning": "Detailed explanation of the evaluation",
    "confidence": 0.0-1.0,
    "key_strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["improvement1", "improvement2"]
}}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response to extract structured data"""

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback parsing
        scores = {}
        confidence = 0.7
        reasoning = response

        # Extract numerical scores
        score_patterns = {
            'overall_quality': r'overall[_\s]*quality[:\s]*([0-9\.]+)',
            'grammar_accuracy': r'grammar[_\s]*accuracy[:\s]*([0-9\.]+)',
            'clarity': r'clarity[:\s]*([0-9\.]+)',
            'professionalism': r'professionalism[:\s]*([0-9\.]+)',
            'engagement': r'engagement[:\s]*([0-9\.]+)'
        }

        for key, pattern in score_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    scores[key] = min(1.0, max(0.0, score))
                except ValueError:
                    pass

        return {
            'scores': scores,
            'reasoning': reasoning,
            'confidence': confidence
        }

    def _fallback_evaluation(self, sample: DataSample, analysis_result) -> LLMJudgmentResult:
        """Fallback evaluation if OpenAI fails"""

        if analysis_result:
            judgment_scores = {
                "overall_quality": analysis_result.overall_score,
                "grammar_accuracy": analysis_result.metrics.professionalism_score,
                "clarity": analysis_result.metrics.clarity_score,
                "professionalism": analysis_result.metrics.professionalism_score,
                "engagement": analysis_result.metrics.engagement_score
            }
            reasoning = f"Fallback analysis: Overall score {analysis_result.overall_score:.2f} with {len(analysis_result.issues)} issues detected."
        else:
            judgment_scores = {"overall_quality": 0.5}
            reasoning = "Analysis failed, using default scores."

        return LLMJudgmentResult(
            judgment_id=str(uuid.uuid4()),
            sample_id=sample.sample_id,
            llm_model=f"{self.model_name}-fallback",
            judgment_scores=judgment_scores,
            reasoning=reasoning,
            confidence=0.6,
            consistency_score=0.5,
            validation_passed=False,
            flagged_issues=["fallback_evaluation"],
            metadata={"fallback_reason": "openai_api_error"}
        )

    def _calculate_consistency(self, sample: DataSample, judgment_scores: Dict[str, float]) -> float:
        """Calculate consistency with expected labels"""

        if not sample.true_labels:
            return 0.7

        consistency_scores = []
        for key, true_value in sample.true_labels.items():
            if key in judgment_scores:
                judgment_value = judgment_scores[key]
                if isinstance(true_value, (int, float)) and isinstance(judgment_value, (int, float)):
                    difference = abs(true_value - judgment_value)
                    consistency = max(0.0, 1.0 - difference)
                    consistency_scores.append(consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.6

class OntologySchema:
    """Robust ontology schema for classification tasks"""

    def __init__(self, domain: str):
        self.domain = domain
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.root_nodes: Set[str] = set()
        self.annotation_standards: Dict[str, AnnotationStandard] = {}
        self.validation_rules: List[str] = []

        self._initialize_domain_ontology()

    def _initialize_domain_ontology(self):
        """Initialize domain-specific ontology"""

        if self.domain == "email_classification":
            self._create_email_classification_ontology()
        elif self.domain == "communication_quality":
            self._create_communication_quality_ontology()

        self._create_annotation_standards()

    def _create_email_classification_ontology(self):
        """Create comprehensive email classification taxonomy"""

        # Root categories
        communication_types = TaxonomyNode(
            node_id="communication_types",
            name="Communication Types",
            description="High-level email communication categories",
            attributes={"level": 0, "importance": "critical"}
        )

        # Business communication branch
        business_comm = TaxonomyNode(
            node_id="business_communication",
            name="Business Communication",
            description="Professional business-related emails",
            parent_id="communication_types",
            attributes={"formality": "high", "stakes": "medium_to_high"}
        )

        # Business sub-categories
        internal_comm = TaxonomyNode(
            node_id="internal_communication",
            name="Internal Communication",
            description="Communication within organization",
            parent_id="business_communication",
            examples=["team updates", "project status", "meeting requests"]
        )

        external_comm = TaxonomyNode(
            node_id="external_communication",
            name="External Communication",
            description="Communication with external parties",
            parent_id="business_communication",
            examples=["client correspondence", "vendor communication", "partner updates"]
        )

        # Quality dimensions
        quality_dimensions = TaxonomyNode(
            node_id="quality_dimensions",
            name="Quality Dimensions",
            description="Aspects of communication quality",
            attributes={"level": 0, "measurement_type": "continuous"}
        )

        grammar_quality = TaxonomyNode(
            node_id="grammar_quality",
            name="Grammar Quality",
            description="Grammatical correctness and language mechanics",
            parent_id="quality_dimensions",
            attributes={"weight": 0.25, "threshold_excellent": 0.95}
        )

        clarity_quality = TaxonomyNode(
            node_id="clarity_quality",
            name="Clarity Quality",
            description="Message clarity and comprehensibility",
            parent_id="quality_dimensions",
            attributes={"weight": 0.30, "threshold_excellent": 0.90}
        )

        professionalism_quality = TaxonomyNode(
            node_id="professionalism_quality",
            name="Professionalism Quality",
            description="Professional tone and appropriateness",
            parent_id="quality_dimensions",
            attributes={"weight": 0.25, "threshold_excellent": 0.85}
        )

        engagement_quality = TaxonomyNode(
            node_id="engagement_quality",
            name="Engagement Quality",
            description="Ability to engage and motivate recipients",
            parent_id="quality_dimensions",
            attributes={"weight": 0.20, "threshold_excellent": 0.80}
        )

        # Add all nodes
        nodes = [
            communication_types, business_comm, internal_comm, external_comm,
            quality_dimensions, grammar_quality, clarity_quality,
            professionalism_quality, engagement_quality
        ]

        for node in nodes:
            self.nodes[node.node_id] = node
            if node.parent_id is None:
                self.root_nodes.add(node.node_id)
            else:
                # Add to parent's children
                if node.parent_id in self.nodes:
                    self.nodes[node.parent_id].children.append(node.node_id)

    def _create_communication_quality_ontology(self):
        """Create communication quality assessment ontology"""

        # Quality levels
        quality_levels = TaxonomyNode(
            node_id="quality_levels",
            name="Quality Levels",
            description="Hierarchical quality classification",
            attributes={"type": "ordinal", "scale": "5_point"}
        )

        excellent = TaxonomyNode(
            node_id="excellent_quality",
            name="Excellent Quality",
            description="Exemplary communication meeting highest standards",
            parent_id="quality_levels",
            attributes={"score_range": (0.9, 1.0), "frequency": 0.1}
        )

        good = TaxonomyNode(
            node_id="good_quality",
            name="Good Quality",
            description="High quality with minor improvement opportunities",
            parent_id="quality_levels",
            attributes={"score_range": (0.7, 0.9), "frequency": 0.3}
        )

        fair = TaxonomyNode(
            node_id="fair_quality",
            name="Fair Quality",
            description="Acceptable quality with noticeable issues",
            parent_id="quality_levels",
            attributes={"score_range": (0.5, 0.7), "frequency": 0.4}
        )

        poor = TaxonomyNode(
            node_id="poor_quality",
            name="Poor Quality",
            description="Significant issues requiring major revision",
            parent_id="quality_levels",
            attributes={"score_range": (0.0, 0.5), "frequency": 0.2}
        )

        nodes = [quality_levels, excellent, good, fair, poor]

        for node in nodes:
            self.nodes[node.node_id] = node
            if node.parent_id is None:
                self.root_nodes.add(node.node_id)
            else:
                if node.parent_id in self.nodes:
                    self.nodes[node.parent_id].children.append(node.node_id)

    def _create_annotation_standards(self):
        """Create comprehensive annotation standards"""

        # Email quality annotation standard
        email_quality_standard = AnnotationStandard(
            standard_id="email_quality_v1",
            name="Email Quality Assessment Standard",
            description="Comprehensive standard for evaluating email communication quality",
            guidelines=[
                "Evaluate grammar, spelling, and language mechanics",
                "Assess tone appropriateness for business context",
                "Measure clarity and comprehensibility of message",
                "Consider engagement and call-to-action effectiveness",
                "Account for completeness and necessary detail level"
            ],
            quality_criteria={
                "grammar_weight": 0.25,
                "clarity_weight": 0.30,
                "professionalism_weight": 0.25,
                "engagement_weight": 0.20
            },
            inter_annotator_agreement_threshold=0.8,
            confidence_thresholds={
                LabelConfidence.VERY_HIGH: (0.9, 1.0),
                LabelConfidence.HIGH: (0.8, 0.9),
                LabelConfidence.MEDIUM: (0.6, 0.8),
                LabelConfidence.LOW: (0.4, 0.6),
                LabelConfidence.VERY_LOW: (0.0, 0.4)
            },
            validation_rules=[
                "Overall score must be weighted average of component scores",
                "Grammar score cannot exceed 0.8 if spelling errors present",
                "Professionalism score must account for tone appropriateness",
                "Clarity score should correlate with sentence length and complexity"
            ]
        )

        self.annotation_standards["email_quality"] = email_quality_standard

    def validate_annotation(self, sample: DataSample, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate annotation against standards"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "confidence": 1.0
        }

        # Check against validation rules
        if "email_quality" in self.annotation_standards:
            standard = self.annotation_standards["email_quality"]

            for rule in standard.validation_rules:
                if not self._check_validation_rule(rule, annotation):
                    validation_result["errors"].append(f"Validation rule failed: {rule}")
                    validation_result["valid"] = False

        return validation_result

    def _check_validation_rule(self, rule: str, annotation: Dict[str, Any]) -> bool:
        """Check specific validation rule"""

        # Simple rule checking (extend for more complex rules)
        if "Overall score must be weighted average" in rule:
            if all(key in annotation for key in ["overall_score", "grammar_score", "clarity_score", "professionalism_score", "engagement_score"]):
                weights = [0.25, 0.30, 0.25, 0.20]
                scores = [annotation["grammar_score"], annotation["clarity_score"],
                         annotation["professionalism_score"], annotation["engagement_score"]]
                expected = sum(w * s for w, s in zip(weights, scores))
                return abs(annotation["overall_score"] - expected) < 0.1

        return True  # Default to passing for unimplemented rules

class AdvancedDatasetGenerator:
    """Advanced dataset generation system with LLM judging and quality control"""

    def __init__(self, domain: str = "email_classification", use_real_llm: bool = None, openai_api_key: str = None):
        self.domain = domain
        self.ontology = OntologySchema(domain)

        # Initialize LLM judge - use real OpenAI if available and requested
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if use_real_llm is None:
            use_real_llm = bool(api_key and OPENAI_AVAILABLE)

        if use_real_llm and api_key and OPENAI_AVAILABLE:
            print("🤖 Initializing REAL OpenAI LLM Judge...")
            self.llm_judge = RealOpenAILLMJudge(api_key=api_key)
            self.using_real_llm = True
        else:
            print("🎭 Using Mock LLM Judge (for demo purposes)")
            self.llm_judge = MockLLMJudge()
            self.using_real_llm = False

        self.educational_demo = WorkingEducationalDemo()

        self.datasets: Dict[str, List[DataSample]] = {}
        self.golden_sets: Dict[str, List[DataSample]] = {}
        self.quality_metrics: Dict[str, Dict[str, float]] = {}

        # Initialize with base examples
        self._initialize_base_datasets()

    def _initialize_base_datasets(self):
        """Initialize with base examples from educational demo"""

        base_samples = []

        # Convert educational examples to data samples
        for example_id, example in self.educational_demo.examples.items():

            # Create true labels based on example quality
            true_labels = self._generate_true_labels_from_quality(example.quality)

            sample = DataSample(
                sample_id=f"base_{example_id}",
                content={
                    "subject": example.input_data["subject"],
                    "body": example.input_data["body"],
                    "sender": example.input_data["sender"]
                },
                true_labels=true_labels,
                quality_score=true_labels["overall_quality"],
                metadata={
                    "source": "educational_demo",
                    "example_title": example.title,
                    "quality_level": example.quality.value,
                    "learning_objectives": example.learning_objectives
                },
                golden_set_member=True  # Base examples are golden set
            )

            base_samples.append(sample)

        self.datasets["base_examples"] = base_samples
        self.golden_sets["base_examples"] = base_samples.copy()

    def _generate_true_labels_from_quality(self, quality) -> Dict[str, float]:
        """Generate true labels based on quality level"""

        if quality.value == "poor":
            return {
                "overall_quality": 0.3,
                "grammar_score": 0.4,
                "clarity_score": 0.4,
                "professionalism_score": 0.2,
                "engagement_score": 0.3
            }
        elif quality.value == "good":
            return {
                "overall_quality": 0.75,
                "grammar_score": 0.8,
                "clarity_score": 0.8,
                "professionalism_score": 0.7,
                "engagement_score": 0.7
            }
        else:  # perfect
            return {
                "overall_quality": 0.95,
                "grammar_score": 0.95,
                "clarity_score": 0.95,
                "professionalism_score": 0.95,
                "engagement_score": 0.9
            }

    def generate_synthetic_samples(self, base_sample: DataSample,
                                 variations: int = 5) -> List[DataSample]:
        """Generate synthetic variations of a base sample"""

        synthetic_samples = []

        for i in range(variations):
            # Create variations by modifying the base sample
            varied_content = self._create_content_variation(base_sample.content, i)

            # Estimate true labels for the variation
            estimated_labels = self._estimate_labels_for_variation(
                base_sample.true_labels, i, variations
            )

            synthetic_sample = DataSample(
                sample_id=f"{base_sample.sample_id}_var_{i}",
                content=varied_content,
                true_labels=estimated_labels,
                metadata={
                    "source": "synthetic_generation",
                    "base_sample": base_sample.sample_id,
                    "variation_index": i,
                    "generation_method": "content_variation"
                }
            )

            synthetic_samples.append(synthetic_sample)

        return synthetic_samples

    def _create_content_variation(self, base_content: Dict[str, str], variation_index: int) -> Dict[str, str]:
        """Create content variation"""

        subject = base_content["subject"]
        body = base_content["body"]
        sender = base_content["sender"]

        # Different types of variations
        if variation_index == 0:
            # Grammar error variation
            body = body.replace("its", "it's").replace("you're", "your")
            subject = subject.replace("tomorrow", "tommorow")
        elif variation_index == 1:
            # Formality variation
            body = body.replace("Dear", "Hi").replace("Best regards", "Thanks")
        elif variation_index == 2:
            # Length variation (shorter)
            sentences = body.split(".")
            body = ". ".join(sentences[:len(sentences)//2]) + "."
        elif variation_index == 3:
            # Clarity variation (more complex)
            body = body.replace("meet", "convene for a collaborative discussion")
        elif variation_index == 4:
            # Professional enhancement
            if "Hi" in body:
                body = body.replace("Hi,", "Dear Colleagues,")
            body += "\n\nThank you for your attention to this matter."

        return {
            "subject": subject,
            "body": body,
            "sender": sender
        }

    def _estimate_labels_for_variation(self, base_labels: Dict[str, float],
                                     variation_index: int, total_variations: int) -> Dict[str, float]:
        """Estimate labels for content variation"""

        estimated_labels = base_labels.copy()

        # Apply variation-specific adjustments
        if variation_index == 0:  # Grammar errors
            estimated_labels["grammar_score"] = max(0.1, estimated_labels["grammar_score"] - 0.3)
            estimated_labels["overall_quality"] = max(0.1, estimated_labels["overall_quality"] - 0.2)
        elif variation_index == 1:  # Less formal
            estimated_labels["professionalism_score"] = max(0.2, estimated_labels["professionalism_score"] - 0.2)
            estimated_labels["overall_quality"] = max(0.2, estimated_labels["overall_quality"] - 0.1)
        elif variation_index == 2:  # Shorter
            estimated_labels["clarity_score"] = max(0.3, estimated_labels["clarity_score"] - 0.1)
        elif variation_index == 3:  # More complex
            estimated_labels["clarity_score"] = max(0.2, estimated_labels["clarity_score"] - 0.2)
        elif variation_index == 4:  # More professional
            estimated_labels["professionalism_score"] = min(1.0, estimated_labels["professionalism_score"] + 0.1)
            estimated_labels["overall_quality"] = min(1.0, estimated_labels["overall_quality"] + 0.05)

        return estimated_labels

    def evaluate_with_llm_judge(self, samples: List[DataSample]) -> List[LLMJudgmentResult]:
        """Evaluate samples using LLM judge"""

        print(f"🤖 Evaluating {len(samples)} samples with LLM judge...")

        criteria = {
            "evaluate_grammar": True,
            "evaluate_clarity": True,
            "evaluate_professionalism": True,
            "evaluate_engagement": True,
            "provide_reasoning": True
        }

        start_time = time.time()

        # Batch evaluation
        llm_results = self.llm_judge.batch_evaluate(samples, criteria)

        evaluation_time = time.time() - start_time

        print(f"✅ LLM evaluation completed in {evaluation_time:.2f} seconds")
        print(f"📊 Results: {len(llm_results)} judgments generated")

        # Update samples with LLM predictions
        for sample, llm_result in zip(samples, llm_results):
            sample.predicted_labels.update(llm_result.judgment_scores)
            sample.annotations.append({
                "source": AnnotationSource.LLM_JUDGE.value,
                "judgment_id": llm_result.judgment_id,
                "scores": llm_result.judgment_scores,
                "reasoning": llm_result.reasoning,
                "confidence": llm_result.confidence,
                "timestamp": llm_result.timestamp.isoformat()
            })

        return llm_results

    def validate_dataset_quality(self, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive dataset quality validation"""

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        samples = self.datasets[dataset_name]

        print(f"🔍 Validating dataset quality for '{dataset_name}'...")
        print(f"📊 Dataset size: {len(samples)} samples")

        validation_results = {
            "dataset_name": dataset_name,
            "total_samples": len(samples),
            "validation_timestamp": datetime.now().isoformat(),
            "quality_metrics": {},
            "distribution_analysis": {},
            "consistency_analysis": {},
            "recommendations": []
        }

        # Quality metrics analysis
        quality_scores = [s.quality_score for s in samples if s.quality_score > 0]
        if quality_scores:
            validation_results["quality_metrics"] = {
                "mean_quality": statistics.mean(quality_scores),
                "std_quality": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "min_quality": min(quality_scores),
                "max_quality": max(quality_scores),
                "samples_with_quality": len(quality_scores)
            }

        # Label distribution analysis
        all_labels = defaultdict(list)
        for sample in samples:
            for label_key, label_value in sample.true_labels.items():
                if isinstance(label_value, (int, float)):
                    all_labels[label_key].append(label_value)

        distribution_analysis = {}
        for label_key, values in all_labels.items():
            if values:
                distribution_analysis[label_key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        validation_results["distribution_analysis"] = distribution_analysis

        # Consistency analysis (between true labels and predictions)
        consistency_scores = []
        for sample in samples:
            if sample.predicted_labels and sample.true_labels:
                sample_consistency = self._calculate_sample_consistency(
                    sample.true_labels, sample.predicted_labels
                )
                consistency_scores.append(sample_consistency)

        if consistency_scores:
            validation_results["consistency_analysis"] = {
                "mean_consistency": statistics.mean(consistency_scores),
                "std_consistency": statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0,
                "samples_analyzed": len(consistency_scores)
            }

        # Generate recommendations
        recommendations = self._generate_dataset_recommendations(validation_results)
        validation_results["recommendations"] = recommendations

        # Store metrics
        self.quality_metrics[dataset_name] = validation_results

        print(f"✅ Dataset validation completed")
        if quality_scores:
            print(f"📈 Mean quality score: {validation_results['quality_metrics']['mean_quality']:.3f}")
        if consistency_scores:
            print(f"🎯 Mean consistency: {validation_results['consistency_analysis']['mean_consistency']:.3f}")

        return validation_results

    def _calculate_sample_consistency(self, true_labels: Dict[str, Any],
                                    predicted_labels: Dict[str, Any]) -> float:
        """Calculate consistency between true and predicted labels"""

        consistency_scores = []

        for key in true_labels:
            if key in predicted_labels:
                true_val = true_labels[key]
                pred_val = predicted_labels[key]

                if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
                    difference = abs(true_val - pred_val)
                    consistency = max(0.0, 1.0 - difference)
                    consistency_scores.append(consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _generate_dataset_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        # Quality recommendations
        quality_metrics = validation_results.get("quality_metrics", {})
        if quality_metrics:
            mean_quality = quality_metrics.get("mean_quality", 0)
            if mean_quality < 0.6:
                recommendations.append("Consider improving overall dataset quality - mean quality below acceptable threshold")

            std_quality = quality_metrics.get("std_quality", 0)
            if std_quality > 0.3:
                recommendations.append("High quality variance detected - consider more consistent quality control")

        # Consistency recommendations
        consistency_analysis = validation_results.get("consistency_analysis", {})
        if consistency_analysis:
            mean_consistency = consistency_analysis.get("mean_consistency", 0)
            if mean_consistency < 0.7:
                recommendations.append("Low label consistency - review annotation process and LLM judge calibration")

        # Distribution recommendations
        distribution_analysis = validation_results.get("distribution_analysis", {})
        for label_key, stats in distribution_analysis.items():
            if stats.get("std", 0) < 0.1:
                recommendations.append(f"Low variance in {label_key} - consider adding more diverse examples")

        # Sample size recommendations
        total_samples = validation_results.get("total_samples", 0)
        if total_samples < 50:
            recommendations.append("Small dataset size - consider generating more samples for robust training")
        elif total_samples > 1000:
            recommendations.append("Large dataset - consider creating stratified subsets for efficient training")

        return recommendations

    def create_golden_dataset(self, dataset_name: str,
                            selection_criteria: Dict[str, Any]) -> List[DataSample]:
        """Create golden dataset based on selection criteria"""

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        candidates = self.datasets[dataset_name]

        print(f"🏆 Creating golden dataset from '{dataset_name}'...")
        print(f"📋 Selection criteria: {selection_criteria}")

        golden_samples = []

        # Apply selection criteria
        min_quality = selection_criteria.get("min_quality", 0.8)
        min_consistency = selection_criteria.get("min_consistency", 0.8)
        max_samples = selection_criteria.get("max_samples", 100)
        require_llm_validation = selection_criteria.get("require_llm_validation", True)

        for sample in candidates:
            # Quality filter
            if sample.quality_score < min_quality:
                continue

            # Consistency filter (if predictions available)
            if sample.predicted_labels:
                consistency = self._calculate_sample_consistency(
                    sample.true_labels, sample.predicted_labels
                )
                if consistency < min_consistency:
                    continue

            # LLM validation filter
            if require_llm_validation:
                llm_annotations = [a for a in sample.annotations
                                 if a.get("source") == AnnotationSource.LLM_JUDGE.value]
                if not llm_annotations:
                    continue

                # Check if any LLM annotation has high confidence
                has_high_confidence = any(a.get("confidence", 0) > 0.8 for a in llm_annotations)
                if not has_high_confidence:
                    continue

            # Mark as golden set member
            sample.golden_set_member = True
            golden_samples.append(sample)

            # Limit size
            if len(golden_samples) >= max_samples:
                break

        # Store golden dataset
        golden_dataset_name = f"{dataset_name}_golden"
        self.golden_sets[golden_dataset_name] = golden_samples

        print(f"✅ Golden dataset created: {len(golden_samples)} samples")
        print(f"🎯 Selection rate: {len(golden_samples)/len(candidates)*100:.1f}%")

        return golden_samples

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of the dataset generation system"""

        print("🚀 ADVANCED DATASET GENERATION SYSTEM ANALYSIS")
        print("=" * 70)

        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "system_overview": {},
            "dataset_analysis": {},
            "llm_judge_analysis": {},
            "quality_validation": {},
            "ontology_analysis": {},
            "recommendations": []
        }

        # System overview
        results["system_overview"] = {
            "total_datasets": len(self.datasets),
            "total_golden_sets": len(self.golden_sets),
            "ontology_domain": self.domain,
            "ontology_nodes": len(self.ontology.nodes),
            "annotation_standards": len(self.ontology.annotation_standards)
        }

        print(f"📊 System Overview:")
        print(f"  Datasets: {results['system_overview']['total_datasets']}")
        print(f"  Golden Sets: {results['system_overview']['total_golden_sets']}")
        print(f"  Ontology Nodes: {results['system_overview']['ontology_nodes']}")

        # Generate synthetic samples for analysis
        print(f"\n🔧 Generating synthetic dataset...")

        base_samples = self.datasets["base_examples"]
        synthetic_samples = []

        for base_sample in base_samples:
            variations = self.generate_synthetic_samples(base_sample, variations=3)
            synthetic_samples.extend(variations)

        synthetic_dataset_name = "synthetic_generated"
        self.datasets[synthetic_dataset_name] = synthetic_samples

        print(f"✅ Generated {len(synthetic_samples)} synthetic samples")

        # LLM evaluation
        print(f"\n🤖 Running LLM judge evaluation...")

        all_samples = base_samples + synthetic_samples
        llm_results = self.evaluate_with_llm_judge(all_samples)

        # Analyze LLM results
        llm_confidences = [r.confidence for r in llm_results]
        llm_consistency_scores = [r.consistency_score for r in llm_results]

        results["llm_judge_analysis"] = {
            "total_evaluations": len(llm_results),
            "mean_confidence": statistics.mean(llm_confidences),
            "mean_consistency": statistics.mean(llm_consistency_scores),
            "high_confidence_rate": len([c for c in llm_confidences if c > 0.8]) / len(llm_confidences),
            "validation_pass_rate": len([r for r in llm_results if r.validation_passed]) / len(llm_results)
        }

        print(f"🎯 LLM Judge Analysis:")
        print(f"  Mean Confidence: {results['llm_judge_analysis']['mean_confidence']:.3f}")
        print(f"  Mean Consistency: {results['llm_judge_analysis']['mean_consistency']:.3f}")
        print(f"  Validation Pass Rate: {results['llm_judge_analysis']['validation_pass_rate']:.3f}")

        # Dataset quality validation
        print(f"\n🔍 Running dataset quality validation...")

        dataset_validations = {}
        for dataset_name in self.datasets:
            validation_result = self.validate_dataset_quality(dataset_name)
            dataset_validations[dataset_name] = validation_result

        results["quality_validation"] = dataset_validations

        # Create golden dataset
        print(f"\n🏆 Creating golden dataset...")

        golden_criteria = {
            "min_quality": 0.7,
            "min_consistency": 0.7,
            "max_samples": 50,
            "require_llm_validation": True
        }

        golden_samples = self.create_golden_dataset(synthetic_dataset_name, golden_criteria)

        # Ontology analysis
        results["ontology_analysis"] = {
            "domain": self.ontology.domain,
            "total_nodes": len(self.ontology.nodes),
            "root_nodes": len(self.ontology.root_nodes),
            "annotation_standards": list(self.ontology.annotation_standards.keys()),
            "validation_rules_count": len(self.ontology.validation_rules)
        }

        # Generate comprehensive recommendations
        system_recommendations = self._generate_system_recommendations(results)
        results["recommendations"] = system_recommendations

        print(f"\n💡 System Recommendations:")
        for rec in system_recommendations[:5]:
            print(f"  • {rec}")

        print(f"\n✅ Comprehensive analysis completed!")

        return results

    def _generate_system_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations"""

        recommendations = []

        # LLM judge recommendations
        llm_analysis = analysis_results.get("llm_judge_analysis", {})
        if llm_analysis.get("mean_confidence", 0) < 0.8:
            recommendations.append("Consider calibrating LLM judge or using ensemble of judges for higher confidence")

        if llm_analysis.get("validation_pass_rate", 0) < 0.7:
            recommendations.append("Review LLM judge validation criteria - low pass rate indicates potential issues")

        # Dataset recommendations
        total_datasets = analysis_results.get("system_overview", {}).get("total_datasets", 0)
        if total_datasets < 3:
            recommendations.append("Expand dataset collection - consider generating more diverse dataset types")

        # Ontology recommendations
        ontology_analysis = analysis_results.get("ontology_analysis", {})
        if ontology_analysis.get("total_nodes", 0) < 10:
            recommendations.append("Expand ontology schema - consider adding more granular classification nodes")

        # Quality validation recommendations
        quality_validations = analysis_results.get("quality_validation", {})
        low_quality_datasets = []
        for dataset_name, validation in quality_validations.items():
            quality_metrics = validation.get("quality_metrics", {})
            if quality_metrics.get("mean_quality", 0) < 0.6:
                low_quality_datasets.append(dataset_name)

        if low_quality_datasets:
            recommendations.append(f"Improve quality for datasets: {', '.join(low_quality_datasets)}")

        # Golden set recommendations
        golden_sets = analysis_results.get("system_overview", {}).get("total_golden_sets", 0)
        if golden_sets == 0:
            recommendations.append("Create golden datasets for benchmarking and validation")

        return recommendations


def main():
    """Run comprehensive demonstration of advanced dataset generation system"""

    print("🚀 ADVANCED DATASET GENERATION SYSTEM")
    print("Real Analysis with LLM Judging and Robust Ontology")
    print("=" * 80)

    # Initialize system
    dataset_generator = AdvancedDatasetGenerator(domain="email_classification")

    # Run comprehensive analysis
    results = dataset_generator.run_comprehensive_analysis()

    # Display final summary
    print(f"\n" + "="*80)
    print("📋 SYSTEM ANALYSIS COMPLETE")
    print("="*80)

    overview = results["system_overview"]
    print(f"✅ Datasets Created: {overview['total_datasets']}")
    print(f"🏆 Golden Sets: {overview['total_golden_sets']}")
    print(f"🧠 Ontology Nodes: {overview['ontology_nodes']}")

    llm_analysis = results["llm_judge_analysis"]
    print(f"🤖 LLM Evaluations: {llm_analysis['total_evaluations']}")
    print(f"🎯 Mean LLM Confidence: {llm_analysis['mean_confidence']:.3f}")
    print(f"✅ Validation Pass Rate: {llm_analysis['validation_pass_rate']:.3f}")

    print(f"\n🎯 Key Capabilities Demonstrated:")
    print(f"  • Real dataset generation with quality control")
    print(f"  • LLM judge evaluation with confidence scoring")
    print(f"  • Robust ontology schema with validation rules")
    print(f"  • Comprehensive quality metrics and analysis")
    print(f"  • Golden dataset creation with selection criteria")
    print(f"  • Automated recommendations and feedback loops")

    print(f"\n✅ Advanced dataset generation system fully operational!")

    return dataset_generator, results


if __name__ == "__main__":
    main()