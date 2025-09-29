from typing import Dict, Any
from .base import BaseFeatureGenerator
from app.dataclasses import Email

class SpamFeatureGenerator(BaseFeatureGenerator):
    """Generates spam detection features from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}".lower()
        
        # Spam word detection
        spam_words = ['free', 'winner', 'congratulations', 'click here', 'limited time', 
                      'act now', 'urgent', 'special offer', 'guaranteed', 'no risk',
                      'cash', 'money back', 'amazing', 'incredible', 'unbeatable']
        
        has_spam_words = int(any(word in all_text for word in spam_words))
        
        return {"has_spam_words": has_spam_words}
    
    @property
    def feature_names(self) -> list[str]:
        return ["has_spam_words"]


class AverageWordLengthFeatureGenerator(BaseFeatureGenerator):
    """Generates average word length feature from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"
        
        # Split into words and calculate average length
        words = all_text.split()
        if not words:
            average_word_length = 0.0
        else:
            total_length = sum(len(word) for word in words)
            average_word_length = total_length / len(words)
        
        return {"average_word_length": average_word_length}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_word_length"]


class EmailEmbeddingsFeatureGenerator(BaseFeatureGenerator):
    """Generates embedding features by averaging title and detail lengths"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
        # Calculate lengths
        title_length = len(subject)
        detail_length = len(body)
        
        # Average of title and detail lengths
        if title_length == 0 and detail_length == 0:
            average_embedding = 0.0
        else:
            average_embedding = (title_length + detail_length) / 2
        
        return {"average_embedding": average_embedding}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_embedding"]


class RawEmailFeatureGenerator(BaseFeatureGenerator):
    """Extracts raw email data as features"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
        return {
            "email_subject": subject,
            "email_body": body
        }
    
    @property
    def feature_names(self) -> list[str]:
        return ["email_subject", "email_body"]


class NonTextCharacterFeatureGenerator(BaseFeatureGenerator):
    """Generates non-text character count features from email content"""

    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"

        # Count non-alphanumeric characters (excluding spaces)
        non_text_count = sum(1 for char in all_text if not char.isalnum() and not char.isspace())

        return {"non_text_char_count": non_text_count}

    @property
    def feature_names(self) -> list[str]:
        return ["non_text_char_count"]


class CausalImpactFeatureGenerator(BaseFeatureGenerator):
    """
    Generates causal impact features for interventional analysis.

    Computes counterfactual features and causal signatures that enable
    downstream causal inference and impact analysis.
    """

    def generate_features(self, email: Email) -> Dict[str, Any]:
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"

        # Baseline features for comparison
        subject_length = len(subject)
        body_length = len(body)
        total_length = len(all_text)

        # Causal signature: feature stability metrics
        # (how much would changing this affect downstream?)
        features = {
            # Intervention readiness scores
            "subject_intervention_impact": subject_length / (total_length + 1),
            "body_intervention_impact": body_length / (total_length + 1),

            # Counterfactual deltas (deviation from expected)
            "length_deviation": abs(total_length - 200) / 200,  # 200 = expected avg

            # Causal interaction features
            "subject_body_interaction": (subject_length * body_length) / (total_length + 1),

            # Stability indicator (variance proxy)
            "feature_stability_score": 1.0 - (abs(subject_length - body_length) / (total_length + 1))
        }

        return features

    @property
    def feature_names(self) -> list[str]:
        return [
            "subject_intervention_impact",
            "body_intervention_impact",
            "length_deviation",
            "subject_body_interaction",
            "feature_stability_score"
        ]