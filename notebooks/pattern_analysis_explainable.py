#!/usr/bin/env python3
"""
Pattern Analysis & Explainable AI for Email Classification
Advanced analysis of email patterns and classification decision explanations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import requests
from datetime import datetime
import re
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ExplainableEmailClassification:
    """
    Advanced pattern analysis and explainable AI for email classification system
    """

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.data_dir = "../data"

        # Load data
        self.emails_df = self._load_emails_as_dataframe()
        self.topics_data = self._load_topics()
        self.classification_patterns = {}
        self.feature_importance = {}

    def _load_emails_as_dataframe(self):
        """Load emails as pandas DataFrame for analysis"""
        with open(f"{self.data_dir}/emails.json", 'r') as f:
            emails = json.load(f)
        return pd.DataFrame(emails)

    def _load_topics(self):
        """Load topics data"""
        with open(f"{self.data_dir}/topic_keywords.json", 'r') as f:
            return json.load(f)

    def analyze_email_patterns_by_topic(self):
        """Analyze email content patterns by topic"""
        print("📊 EMAIL CONTENT PATTERNS BY TOPIC")
        print("="*80)

        # Filter emails with ground truth
        labeled_emails = self.emails_df[self.emails_df['ground_truth'].notna()].copy()

        patterns_by_topic = {}

        for topic in labeled_emails['ground_truth'].unique():
            topic_emails = labeled_emails[labeled_emails['ground_truth'] == topic]

            patterns = {
                'count': len(topic_emails),
                'avg_subject_length': topic_emails['subject'].str.len().mean(),
                'avg_body_length': topic_emails['body'].str.len().mean(),
                'common_subject_words': self._extract_common_words(topic_emails['subject'].tolist()),
                'common_body_words': self._extract_common_words(topic_emails['body'].tolist()),
                'special_char_patterns': self._analyze_special_chars(topic_emails),
                'sentiment_indicators': self._analyze_sentiment_indicators(topic_emails)
            }

            patterns_by_topic[topic] = patterns

            print(f"\n🏷️  Topic: {topic} ({patterns['count']} emails)")
            print(f"   📏 Average lengths: Subject {patterns['avg_subject_length']:.1f}, Body {patterns['avg_body_length']:.1f}")
            print(f"   🔤 Common subject words: {patterns['common_subject_words'][:5]}")
            print(f"   📝 Common body words: {patterns['common_body_words'][:5]}")
            print(f"   ⚡ Special char avg: {patterns['special_char_patterns']['avg']:.1f}")

        self.classification_patterns = patterns_by_topic
        return patterns_by_topic

    def _extract_common_words(self, texts: List[str], min_length=3) -> List[str]:
        """Extract most common words from text list"""
        words = []
        for text in texts:
            words.extend([
                word.lower().strip('.,!?;:"()[]{}')
                for word in str(text).split()
                if len(word) >= min_length and word.isalpha()
            ])

        return [word for word, count in Counter(words).most_common(10)]

    def _analyze_special_chars(self, emails_subset: pd.DataFrame) -> Dict:
        """Analyze special character patterns"""
        def count_special_chars(text):
            return sum(1 for char in str(text) if not char.isalnum() and not char.isspace())

        special_counts = (emails_subset['subject'] + ' ' + emails_subset['body']).apply(count_special_chars)

        return {
            'avg': special_counts.mean(),
            'std': special_counts.std(),
            'min': special_counts.min(),
            'max': special_counts.max()
        }

    def _analyze_sentiment_indicators(self, emails_subset: pd.DataFrame) -> Dict:
        """Analyze sentiment and urgency indicators"""
        combined_text = (emails_subset['subject'] + ' ' + emails_subset['body']).str.lower()

        urgent_words = ['urgent', 'asap', 'immediately', 'deadline', 'critical']
        positive_words = ['great', 'excellent', 'congratulations', 'success', 'thank']
        negative_words = ['problem', 'issue', 'error', 'fail', 'trouble']

        return {
            'urgency_score': combined_text.apply(lambda x: sum(word in str(x) for word in urgent_words)).mean(),
            'positive_score': combined_text.apply(lambda x: sum(word in str(x) for word in positive_words)).mean(),
            'negative_score': combined_text.apply(lambda x: sum(word in str(x) for word in negative_words)).mean()
        }

    def explain_classification_decision(self, email_subject: str, email_body: str) -> Dict:
        """Provide detailed explanation of classification decision"""
        print(f"\n🔍 CLASSIFICATION DECISION EXPLANATION")
        print("="*80)
        print(f"📧 Email: '{email_subject}'")
        print(f"📝 Body: '{email_body[:100]}{'...' if len(email_body) > 100 else ''}'")

        # Get classification result
        test_email = {"subject": email_subject, "body": email_body}

        try:
            response = requests.post(f"{self.base_url}/emails/classify", json=test_email)
            if response.status_code != 200:
                return {"error": f"Classification failed: {response.status_code}"}

            result = response.json()
            predicted_topic = result.get('predicted_topic')
            topic_scores = result.get('topic_scores', {})
            features = result.get('features', {})

            print(f"\n🎯 Prediction: {predicted_topic}")
            print(f"🔢 Confidence: {topic_scores.get(predicted_topic, 0):.3f}")

            # Feature-based explanation
            explanation = self._explain_features(features, email_subject, email_body)

            # Pattern-based explanation
            pattern_explanation = self._explain_patterns(email_subject, email_body, predicted_topic)

            # Score analysis
            score_explanation = self._explain_scores(topic_scores)

            return {
                'predicted_topic': predicted_topic,
                'confidence': topic_scores.get(predicted_topic, 0),
                'feature_explanation': explanation,
                'pattern_explanation': pattern_explanation,
                'score_explanation': score_explanation,
                'all_scores': topic_scores,
                'features': features
            }

        except Exception as e:
            return {"error": f"Error during classification: {e}"}

    def _explain_features(self, features: Dict, subject: str, body: str) -> Dict:
        """Explain how each feature contributes to classification"""
        print(f"\n📊 FEATURE ANALYSIS:")

        explanations = {}

        # Spam features
        spam_score = features.get('spam_has_spam_words', 0)
        print(f"   🚫 Spam indicator: {spam_score} ({'Yes' if spam_score > 0 else 'No'} spam words detected)")
        explanations['spam'] = f"{'Contains' if spam_score > 0 else 'No'} spam keywords"

        # Word length features
        avg_word_length = features.get('word_length_average_word_length', 0)
        print(f"   📏 Average word length: {avg_word_length:.1f} chars")
        if avg_word_length > 6:
            explanations['word_length'] = "Longer words suggest formal/technical content"
        elif avg_word_length < 4:
            explanations['word_length'] = "Shorter words suggest casual content"
        else:
            explanations['word_length'] = "Average word length is typical"

        # Non-text character features
        special_chars = features.get('non_text_non_text_char_count', 0)
        print(f"   🔤 Special characters: {special_chars}")
        if special_chars > 10:
            explanations['special_chars'] = "High special character count (formatted content/technical)"
        elif special_chars > 5:
            explanations['special_chars'] = "Moderate special characters (normal punctuation)"
        else:
            explanations['special_chars'] = "Low special characters (plain text)"

        # Embedding features
        embedding_val = features.get('email_embeddings_average_embedding', 0)
        print(f"   🧠 Content embedding: {embedding_val:.1f}")
        explanations['embedding'] = f"Content complexity score: {embedding_val:.1f}"

        return explanations

    def _explain_patterns(self, subject: str, body: str, predicted_topic: str) -> Dict:
        """Explain pattern-based reasoning"""
        print(f"\n🎨 PATTERN ANALYSIS:")

        combined_text = f"{subject} {body}".lower()

        # Check for topic-specific patterns
        pattern_matches = {}

        # Work patterns
        work_indicators = ['meeting', 'project', 'deadline', 'budget', 'team', 'report', 'review']
        work_matches = [word for word in work_indicators if word in combined_text]
        if work_matches:
            pattern_matches['work'] = f"Work indicators: {work_matches}"

        # Personal patterns
        personal_indicators = ['birthday', 'family', 'friend', 'personal', 'vacation', 'weekend']
        personal_matches = [word for word in personal_indicators if word in combined_text]
        if personal_matches:
            pattern_matches['personal'] = f"Personal indicators: {personal_matches}"

        # Finance patterns
        finance_indicators = ['budget', 'financial', 'money', 'investment', 'revenue', 'cost']
        finance_matches = [word for word in finance_indicators if word in combined_text]
        if finance_matches:
            pattern_matches['finance'] = f"Finance indicators: {finance_matches}"

        # Education patterns
        education_indicators = ['course', 'learning', 'training', 'education', 'certificate', 'grade']
        education_matches = [word for word in education_indicators if word in combined_text]
        if education_matches:
            pattern_matches['education'] = f"Education indicators: {education_matches}"

        print(f"   🔍 Pattern matches found: {len(pattern_matches)}")
        for topic, matches in pattern_matches.items():
            print(f"      {topic}: {matches}")

        return pattern_matches

    def _explain_scores(self, topic_scores: Dict) -> Dict:
        """Explain the scoring mechanism"""
        print(f"\n📈 SCORE ANALYSIS:")

        sorted_scores = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"   🥇 Top 5 scores:")
        for i, (topic, score) in enumerate(sorted_scores[:5], 1):
            print(f"      {i}. {topic}: {score:.3f}")

        # Calculate score gaps
        if len(sorted_scores) >= 2:
            top_score = sorted_scores[0][1]
            second_score = sorted_scores[1][1]
            confidence_gap = top_score - second_score

            print(f"\n   📊 Confidence analysis:")
            print(f"      Gap to second choice: {confidence_gap:.3f}")

            if confidence_gap > 0.3:
                confidence_level = "High"
            elif confidence_gap > 0.1:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

            print(f"      Confidence level: {confidence_level}")

        return {
            'top_scores': dict(sorted_scores[:5]),
            'confidence_gap': confidence_gap if len(sorted_scores) >= 2 else 0,
            'total_topics': len(topic_scores)
        }

    def analyze_misclassifications(self) -> Dict:
        """Analyze potential misclassifications in the dataset"""
        print(f"\n🔍 MISCLASSIFICATION ANALYSIS")
        print("="*80)

        labeled_emails = self.emails_df[self.emails_df['ground_truth'].notna()].copy()
        misclassification_analysis = {}

        print(f"📊 Analyzing {len(labeled_emails)} labeled emails...")

        correct_predictions = 0
        total_predictions = 0

        for idx, email in labeled_emails.iterrows():
            try:
                test_email = {
                    "subject": email['subject'],
                    "body": email['body']
                }

                response = requests.post(f"{self.base_url}/emails/classify", json=test_email)
                if response.status_code == 200:
                    result = response.json()
                    predicted = result.get('predicted_topic')
                    actual = email['ground_truth']

                    total_predictions += 1
                    if predicted == actual:
                        correct_predictions += 1
                    else:
                        # Log misclassification
                        if actual not in misclassification_analysis:
                            misclassification_analysis[actual] = []

                        misclassification_analysis[actual].append({
                            'email_id': email['id'],
                            'subject': email['subject'][:50] + '...',
                            'predicted': predicted,
                            'confidence': result.get('topic_scores', {}).get(predicted, 0)
                        })

            except Exception as e:
                print(f"   Error analyzing email {email['id']}: {e}")

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n📈 Overall Classification Accuracy: {accuracy:.1%}")
        print(f"   Correct: {correct_predictions}/{total_predictions}")

        # Report misclassifications by topic
        for topic, errors in misclassification_analysis.items():
            print(f"\n❌ {topic} misclassifications ({len(errors)}):")
            for error in errors[:3]:  # Show first 3
                print(f"   ID {error['email_id']}: '{error['subject']}' → {error['predicted']} ({error['confidence']:.3f})")

        return {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'misclassifications_by_topic': misclassification_analysis
        }

    def generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for improving classification accuracy"""
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("="*80)

        recommendations = []

        # Based on pattern analysis
        if hasattr(self, 'classification_patterns'):
            recommendations.extend([
                "1. Replace simplified string-length embeddings with semantic embeddings (e.g., sentence-transformers)",
                "2. Implement topic-specific keyword dictionaries based on pattern analysis",
                "3. Add feature weights based on topic-specific importance",
                "4. Increase training data for underrepresented topics",
                "5. Implement confidence thresholding to flag uncertain predictions"
            ])

        # Based on feature analysis
        recommendations.extend([
            "6. Add temporal features (time of day, day of week) for context",
            "7. Implement n-gram features for better text understanding",
            "8. Add sender domain analysis for additional context",
            "9. Implement ensemble methods combining multiple classifiers",
            "10. Add active learning to improve model with user feedback"
        ])

        for rec in recommendations:
            print(f"   {rec}")

        return recommendations

    def run_complete_explainable_analysis(self):
        """Run complete explainable AI analysis"""
        print("🔍 EXPLAINABLE AI ANALYSIS STARTING")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Pattern analysis
        patterns = self.analyze_email_patterns_by_topic()

        # Test explanations with sample emails
        sample_emails = [
            ("Quarterly Financial Review", "We need to discuss Q4 budget allocations and revenue projections"),
            ("Happy Birthday!", "Hope you have a wonderful day celebrating with family and friends"),
            ("Course Registration Open", "New machine learning courses are now available for enrollment")
        ]

        explanations = []
        for subject, body in sample_emails:
            explanation = self.explain_classification_decision(subject, body)
            explanations.append(explanation)

        # Misclassification analysis
        misclass_analysis = self.analyze_misclassifications()

        # Generate recommendations
        recommendations = self.generate_improvement_recommendations()

        print(f"\n📊 EXPLAINABLE AI SUMMARY")
        print("="*80)
        print("✅ Pattern analysis completed")
        print("✅ Classification explanations generated")
        print("✅ Misclassification analysis performed")
        print("✅ Improvement recommendations provided")
        print(f"\n📈 Key Insights:")
        print(f"   - Classification accuracy: {misclass_analysis['overall_accuracy']:.1%}")
        print(f"   - Pattern-based explanations available for all predictions")
        print(f"   - Feature importance documented for transparency")
        print(f"   - {len(recommendations)} improvement recommendations generated")

        return {
            'patterns': patterns,
            'explanations': explanations,
            'misclassification_analysis': misclass_analysis,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    analyzer = ExplainableEmailClassification()
    results = analyzer.run_complete_explainable_analysis()