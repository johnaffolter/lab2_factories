"""
Composable Feature Generators
Refactored to use composable component architecture
"""

from typing import Dict, Any
from app.core.composable import (
    ComposableComponent,
    ComponentMetadata,
    ComponentType,
    global_registry
)


@global_registry.register
class ComposableSpamGenerator(ComposableComponent):
    """Composable spam detection feature generator"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Spam Detector",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Detects spam keywords and patterns in email content",
            author="John Affolter",
            tags=["spam", "detection", "nlp", "keywords"],
            icon="🚫",
            color="#FF6B6B",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "spam_score", "type": "float"},
                {"name": "spam_keywords", "type": "List[str]"},
                {"name": "has_spam", "type": "bool"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["free", "winner", "urgent", "click", "limited"]
                    },
                    "threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute spam detection"""
        keywords = self.config.get("keywords", ["free", "winner", "urgent"])
        threshold = self.config.get("threshold", 0.3)

        text = f"{input_data.subject} {input_data.body}".lower()

        found_keywords = [kw for kw in keywords if kw in text]
        spam_score = len(found_keywords) / len(keywords) if keywords else 0

        return {
            "spam_score": spam_score,
            "spam_keywords": found_keywords,
            "has_spam": spam_score > threshold
        }


@global_registry.register
class ComposableWordLengthGenerator(ComposableComponent):
    """Composable word length feature generator"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Word Length Analyzer",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Analyzes average word length and vocabulary richness",
            author="John Affolter",
            tags=["nlp", "linguistics", "statistics"],
            icon="📏",
            color="#4ECDC4",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "avg_word_length", "type": "float"},
                {"name": "total_words", "type": "int"},
                {"name": "unique_words", "type": "int"},
                {"name": "vocabulary_richness", "type": "float"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "include_subject": {
                        "type": "boolean",
                        "default": True
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute word length analysis"""
        include_subject = self.config.get("include_subject", True)

        text = input_data.body
        if include_subject:
            text = f"{input_data.subject} {text}"

        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return {
                "avg_word_length": 0.0,
                "total_words": 0,
                "unique_words": 0,
                "vocabulary_richness": 0.0
            }

        avg_length = sum(len(word) for word in words) / total_words
        unique_words = len(set(words))
        vocab_richness = unique_words / total_words

        return {
            "avg_word_length": round(avg_length, 2),
            "total_words": total_words,
            "unique_words": unique_words,
            "vocabulary_richness": round(vocab_richness, 3)
        }


@global_registry.register
class ComposableEmbeddingGenerator(ComposableComponent):
    """Composable email embedding generator"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Email Embedder",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Generates numerical embeddings from email content",
            author="John Affolter",
            tags=["embeddings", "vector", "ml", "representation"],
            icon="🔢",
            color="#95E1D3",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "embedding_vector", "type": "float"},
                {"name": "total_chars", "type": "int"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "normalize": {
                        "type": "boolean",
                        "default": True
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute embedding generation"""
        text = f"{input_data.subject} {input_data.body}"
        total_chars = len(text)

        # Simple embedding: normalized character count
        embedding = total_chars / 1000.0 if self.config.get("normalize", True) else total_chars

        return {
            "embedding_vector": round(embedding, 4),
            "total_chars": total_chars
        }


@global_registry.register
class ComposableSentimentAnalyzer(ComposableComponent):
    """Composable sentiment analysis component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Sentiment Analyzer",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Analyzes sentiment and tone of email content",
            author="John Affolter",
            tags=["sentiment", "emotion", "nlp", "analysis"],
            icon="😊",
            color="#F38181",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "sentiment", "type": "str"},
                {"name": "confidence", "type": "float"},
                {"name": "tone", "type": "str"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["lexicon", "ml"],
                        "default": "lexicon"
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        text = f"{input_data.subject} {input_data.body}".lower()

        # Simple lexicon-based sentiment
        positive_words = ["great", "excellent", "good", "happy", "love", "wonderful"]
        negative_words = ["bad", "terrible", "hate", "angry", "frustrated", "problem"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / (positive_count + negative_count + 1)
            tone = "enthusiastic"
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / (positive_count + negative_count + 1)
            tone = "concerned"
        else:
            sentiment = "neutral"
            confidence = 0.5
            tone = "professional"

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "tone": tone
        }


@global_registry.register
class ComposableUrgencyDetector(ComposableComponent):
    """Composable urgency detection component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Urgency Detector",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Detects urgency and priority indicators in emails",
            author="John Affolter",
            tags=["urgency", "priority", "detection"],
            icon="⚡",
            color="#FFA500",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "urgency_score", "type": "float"},
                {"name": "urgency_level", "type": "str"},
                {"name": "urgency_keywords", "type": "List[str]"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "urgency_keywords": {
                        "type": "array",
                        "default": ["urgent", "asap", "immediately", "critical", "emergency"]
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute urgency detection"""
        keywords = self.config.get("urgency_keywords", ["urgent", "asap", "immediately"])
        text = f"{input_data.subject} {input_data.body}".lower()

        found = [kw for kw in keywords if kw in text]
        urgency_score = len(found) / len(keywords) if keywords else 0

        if urgency_score > 0.5:
            urgency_level = "high"
        elif urgency_score > 0.2:
            urgency_level = "medium"
        else:
            urgency_level = "low"

        return {
            "urgency_score": round(urgency_score, 2),
            "urgency_level": urgency_level,
            "urgency_keywords": found
        }