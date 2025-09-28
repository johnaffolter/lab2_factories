import os
import json
import math
from typing import Dict, Any, List, Optional

class EmailClassifierModel:
    """Simple rule-based email classifier model"""

    def __init__(self, use_email_similarity: bool = False):
        self.use_email_similarity = use_email_similarity
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())
        self.stored_emails = self._load_stored_emails() if use_email_similarity else []

    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def _load_stored_emails(self) -> List[Dict[str, Any]]:
        """Load stored emails from data/emails.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature or email similarity"""
        if self.use_email_similarity and self.stored_emails:
            return self._predict_by_email_similarity(features)
        else:
            return self._predict_by_topic_similarity(features)

    def _predict_by_topic_similarity(self, features: Dict[str, Any]) -> str:
        """Classify email using topic similarity"""
        scores = {}

        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score

        return max(scores, key=scores.get)

    def _predict_by_email_similarity(self, features: Dict[str, Any]) -> str:
        """Classify email by finding most similar stored email with ground truth"""
        if not self.stored_emails:
            return self._predict_by_topic_similarity(features)

        # Find emails with ground truth
        labeled_emails = [e for e in self.stored_emails if 'ground_truth' in e]
        if not labeled_emails:
            return self._predict_by_topic_similarity(features)

        # Get email text from features
        email_subject = features.get("raw_email_email_subject", "")
        email_body = features.get("raw_email_email_body", "")
        email_text = f"{email_subject} {email_body}"

        # Calculate similarity with each labeled email
        best_match = None
        best_similarity = -1

        for stored_email in labeled_emails:
            stored_text = f"{stored_email.get('subject', '')} {stored_email.get('body', '')}"
            similarity = self._calculate_text_similarity(email_text, stored_text)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = stored_email

        if best_match and best_similarity > 0.3:  # Threshold for similarity
            return best_match['ground_truth']
        else:
            return self._predict_by_topic_similarity(features)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate similarity score based on length difference"""
        # Get email embedding from features
        email_embedding = features.get("email_embeddings_average_embedding", 0.0)
        
        # Get topic description and create embedding (description length as embedding)
        topic_description = self.topic_data[topic]['description']
        topic_embedding = float(len(topic_description))
        
        # Calculate similarity based on inverse distance
        # Smaller distance = higher similarity
        distance = abs(email_embedding - topic_embedding)
        
        # Normalize to 0-1 range using exponential decay
        # e^(-distance/scale) gives values between 0 and 1
        scale = 50.0  # Adjust this to control how quickly similarity drops with distance
        similarity = math.exp(-distance / scale)
        
        return similarity
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts based on word overlap"""
        # Simple word-based similarity (can be improved with actual embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0