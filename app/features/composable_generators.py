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


@global_registry.register
class ComposableEntityExtractor(ComposableComponent):
    """Composable entity extraction component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Entity Extractor",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Extracts named entities like names, emails, dates, and URLs from text",
            author="John Affolter",
            tags=["entities", "ner", "extraction", "nlp"],
            icon="🏷️",
            color="#6C5CE7",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "emails", "type": "List[str]"},
                {"name": "urls", "type": "List[str]"},
                {"name": "dates", "type": "List[str]"},
                {"name": "phone_numbers", "type": "List[str]"},
                {"name": "entity_count", "type": "int"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "extract_emails": {"type": "boolean", "default": True},
                    "extract_urls": {"type": "boolean", "default": True},
                    "extract_dates": {"type": "boolean", "default": True},
                    "extract_phones": {"type": "boolean", "default": True}
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute entity extraction"""
        import re

        text = f"{input_data.subject} {input_data.body}"

        entities = {}

        # Extract emails
        if self.config.get("extract_emails", True):
            entities["emails"] = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

        # Extract URLs
        if self.config.get("extract_urls", True):
            entities["urls"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

        # Extract dates (simple pattern)
        if self.config.get("extract_dates", True):
            entities["dates"] = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)

        # Extract phone numbers
        if self.config.get("extract_phones", True):
            entities["phone_numbers"] = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)

        entity_count = sum(len(v) for v in entities.values())

        return {
            "emails": entities.get("emails", []),
            "urls": entities.get("urls", []),
            "dates": entities.get("dates", []),
            "phone_numbers": entities.get("phone_numbers", []),
            "entity_count": entity_count
        }


@global_registry.register
class ComposableTopicClassifier(ComposableComponent):
    """Composable topic classification component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Topic Classifier",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Classifies email content into predefined topics using keyword matching",
            author="John Affolter",
            tags=["classification", "topics", "categorization"],
            icon="📂",
            color="#00B894",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "primary_topic", "type": "str"},
                {"name": "confidence", "type": "float"},
                {"name": "all_topics", "type": "Dict[str, float]"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "object",
                        "default": {
                            "work": ["meeting", "project", "deadline", "report", "presentation"],
                            "personal": ["family", "friend", "birthday", "party", "vacation"],
                            "finance": ["payment", "invoice", "transaction", "account", "balance"],
                            "shopping": ["order", "purchase", "delivery", "shipping", "product"],
                            "technical": ["error", "bug", "issue", "server", "database", "code"]
                        }
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute topic classification"""
        text = f"{input_data.subject} {input_data.body}".lower()
        topics = self.config.get("topics", {
            "work": ["meeting", "project", "deadline"],
            "personal": ["family", "friend", "birthday"],
            "finance": ["payment", "invoice", "account"]
        })

        topic_scores = {}
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score / len(keywords)

        if not topic_scores:
            return {
                "primary_topic": "unknown",
                "confidence": 0.0,
                "all_topics": {}
            }

        primary_topic = max(topic_scores, key=topic_scores.get)
        confidence = topic_scores[primary_topic]

        return {
            "primary_topic": primary_topic,
            "confidence": round(confidence, 2),
            "all_topics": {k: round(v, 2) for k, v in topic_scores.items()}
        }


@global_registry.register
class ComposableReadabilityAnalyzer(ComposableComponent):
    """Composable readability analysis component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Readability Analyzer",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Analyzes text readability using metrics like sentence length and complexity",
            author="John Affolter",
            tags=["readability", "complexity", "linguistics"],
            icon="📖",
            color="#FD79A8",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "avg_sentence_length", "type": "float"},
                {"name": "sentence_count", "type": "int"},
                {"name": "complexity_score", "type": "float"},
                {"name": "readability_level", "type": "str"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "include_subject": {"type": "boolean", "default": False}
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute readability analysis"""
        import re

        text = input_data.body
        if self.config.get("include_subject", False):
            text = f"{input_data.subject} {text}"

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        if sentence_count == 0:
            return {
                "avg_sentence_length": 0.0,
                "sentence_count": 0,
                "complexity_score": 0.0,
                "readability_level": "unknown"
            }

        words = text.split()
        avg_sentence_length = len(words) / sentence_count

        # Simple complexity score based on word and sentence length
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        complexity_score = (avg_sentence_length / 20 + avg_word_length / 10) / 2

        # Determine readability level
        if complexity_score < 0.3:
            readability_level = "very_easy"
        elif complexity_score < 0.5:
            readability_level = "easy"
        elif complexity_score < 0.7:
            readability_level = "moderate"
        else:
            readability_level = "complex"

        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentence_count": sentence_count,
            "complexity_score": round(complexity_score, 2),
            "readability_level": readability_level
        }


@global_registry.register
class ComposableLanguageDetector(ComposableComponent):
    """Composable language detection component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Language Detector",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Detects the language of email content using character patterns",
            author="John Affolter",
            tags=["language", "detection", "i18n", "multilingual"],
            icon="🌍",
            color="#00CEC9",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "language", "type": "str"},
                {"name": "confidence", "type": "float"},
                {"name": "is_mixed", "type": "bool"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "supported_languages": {
                        "type": "array",
                        "default": ["en", "es", "fr", "de"]
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute language detection"""
        text = f"{input_data.subject} {input_data.body}".lower()

        # Simple language detection using common words
        language_patterns = {
            "en": ["the", "is", "and", "to", "a", "of", "that", "in"],
            "es": ["el", "la", "de", "que", "y", "es", "en", "un"],
            "fr": ["le", "de", "un", "être", "et", "à", "il", "avoir"],
            "de": ["der", "die", "und", "ist", "das", "in", "von", "zu"]
        }

        scores = {}
        for lang, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text.split())
            scores[lang] = score

        if sum(scores.values()) == 0:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_mixed": False
            }

        total_matches = sum(scores.values())
        detected_lang = max(scores, key=scores.get)
        confidence = scores[detected_lang] / total_matches

        # Check if mixed (multiple languages detected)
        high_scores = [s for s in scores.values() if s > 0]
        is_mixed = len(high_scores) > 1 and confidence < 0.7

        return {
            "language": detected_lang,
            "confidence": round(confidence, 2),
            "is_mixed": is_mixed
        }


@global_registry.register
class ComposableActionItemExtractor(ComposableComponent):
    """Composable action item extraction component"""

    def get_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="Action Item Extractor",
            version="1.0.0",
            type=ComponentType.FEATURE_GENERATOR,
            description="Extracts actionable items and tasks from email content",
            author="John Affolter",
            tags=["actions", "tasks", "todo", "extraction"],
            icon="✅",
            color="#A29BFE",
            inputs=[
                {"name": "email", "type": "Email", "required": True}
            ],
            outputs=[
                {"name": "action_items", "type": "List[str]"},
                {"name": "action_count", "type": "int"},
                {"name": "has_deadline", "type": "bool"},
                {"name": "priority_level", "type": "str"}
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "action_verbs": {
                        "type": "array",
                        "default": ["please", "need", "must", "should", "review", "complete", "send", "prepare"]
                    }
                }
            }
        )

    def execute(self, input_data) -> Dict[str, Any]:
        """Execute action item extraction"""
        import re

        text = f"{input_data.subject} {input_data.body}".lower()
        action_verbs = self.config.get("action_verbs", ["please", "need", "must", "should"])

        # Extract sentences with action verbs
        sentences = re.split(r'[.!?]+', text)
        action_items = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(verb in sentence for verb in action_verbs):
                action_items.append(sentence[:100])  # Limit length

        # Check for deadlines
        deadline_patterns = [r'\bby \w+\b', r'\bdeadline\b', r'\bdue\b', r'\basap\b']
        has_deadline = any(re.search(pattern, text) for pattern in deadline_patterns)

        # Determine priority
        if any(word in text for word in ["urgent", "immediately", "asap", "critical"]):
            priority_level = "high"
        elif has_deadline or len(action_items) > 2:
            priority_level = "medium"
        else:
            priority_level = "low"

        return {
            "action_items": action_items[:10],  # Limit to 10 items
            "action_count": len(action_items),
            "has_deadline": has_deadline,
            "priority_level": priority_level
        }