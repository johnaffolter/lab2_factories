"""
Advanced Email Analysis Service
Extracts action items, analyzes attachments, and provides hybrid retrieval
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.dataclasses import Email
from app.services.document_graph_service import DocumentGraphService, DocumentType

@dataclass
class ActionItem:
    """Represents an action item extracted from email"""
    task: str
    deadline: Optional[datetime]
    priority: str  # high, medium, low
    assigned_to: Optional[str]
    context: str
    confidence: float

@dataclass
class AttachmentMetadata:
    """Metadata for email attachments"""
    filename: str
    file_type: str
    size_bytes: int
    mime_type: str
    extracted_text: Optional[str]
    summary: Optional[str]

@dataclass
class EmailAnalysis:
    """Complete email analysis results"""
    email_id: str
    importance_score: float
    urgency_level: str
    action_items: List[ActionItem]
    key_entities: Dict[str, List[str]]
    attachments: List[AttachmentMetadata]
    summary: str
    sentiment: str
    topics: List[str]
    embeddings: List[float]

class AdvancedEmailAnalyzer:
    """Advanced email analysis with NLP and ML capabilities"""

    def __init__(self):
        """Initialize analyzer with NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if spacy model not installed
            self.nlp = None

        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph_service = DocumentGraphService()

        # Action item patterns
        self.action_patterns = [
            r"please\s+(\w+)",
            r"could you\s+(\w+)",
            r"can you\s+(\w+)",
            r"need to\s+(\w+)",
            r"make sure\s+(?:to\s+)?(\w+)",
            r"don't forget\s+(?:to\s+)?(\w+)",
            r"remember to\s+(\w+)",
            r"ensure\s+(?:that\s+)?(\w+)",
            r"action required:\s*(.*?)(?:\.|$)",
            r"todo:\s*(.*?)(?:\.|$)",
            r"task:\s*(.*?)(?:\.|$)",
        ]

        # Deadline patterns
        self.deadline_patterns = [
            r"by\s+(\w+day)",
            r"before\s+(\w+day)",
            r"due\s+(\w+day)",
            r"deadline:\s*(.*?)(?:\.|$)",
            r"by\s+(\d{1,2}[:/]\d{1,2})",
            r"before\s+(\d{1,2}[:/]\d{1,2})",
            r"eod\s+(\w+day)?",
            r"end of day\s+(\w+day)?",
        ]

        # Priority indicators
        self.high_priority_words = ["urgent", "asap", "immediately", "critical", "important"]
        self.medium_priority_words = ["soon", "when possible", "this week"]
        self.low_priority_words = ["eventually", "when you can", "no rush"]

    def analyze_email(self, email: Email, extract_attachments: bool = True) -> EmailAnalysis:
        """
        Comprehensive email analysis

        Args:
            email: Email to analyze
            extract_attachments: Whether to process attachments

        Returns:
            Complete email analysis
        """
        # Generate embeddings
        embeddings = self._generate_embeddings(email)

        # Extract action items
        action_items = self._extract_action_items(email)

        # Analyze importance and urgency
        importance_score = self._calculate_importance(email, action_items)
        urgency_level = self._determine_urgency(email, action_items)

        # Extract entities
        key_entities = self._extract_entities(email)

        # Process attachments
        attachments = self._process_attachments(email) if extract_attachments else []

        # Generate summary
        summary = self._generate_summary(email)

        # Analyze sentiment
        sentiment = self._analyze_sentiment(email)

        # Extract topics
        topics = self._extract_topics(email)

        analysis = EmailAnalysis(
            email_id=f"email_{hash(email.subject + email.body)}",
            importance_score=importance_score,
            urgency_level=urgency_level,
            action_items=action_items,
            key_entities=key_entities,
            attachments=attachments,
            summary=summary,
            sentiment=sentiment,
            topics=topics,
            embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        )

        # Store in graph
        self._store_in_graph(email, analysis)

        return analysis

    def _extract_action_items(self, email: Email) -> List[ActionItem]:
        """Extract action items from email"""
        action_items = []
        full_text = f"{email.subject} {email.body}".lower()
        sentences = full_text.split('.')

        for sentence in sentences:
            # Check for action patterns
            for pattern in self.action_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    task = match.group(0).strip()

                    # Extract deadline
                    deadline = self._extract_deadline(sentence)

                    # Determine priority
                    priority = self._determine_priority(sentence)

                    # Extract assignee
                    assigned_to = self._extract_assignee(sentence)

                    action_items.append(ActionItem(
                        task=task,
                        deadline=deadline,
                        priority=priority,
                        assigned_to=assigned_to,
                        context=sentence.strip(),
                        confidence=0.85
                    ))

        return action_items

    def _extract_deadline(self, text: str) -> Optional[datetime]:
        """Extract deadline from text"""
        for pattern in self.deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                deadline_str = match.group(1) if match.lastindex else match.group(0)

                # Parse common deadline formats
                today = datetime.now()
                deadline_str_lower = deadline_str.lower() if deadline_str else ""

                if "monday" in deadline_str_lower:
                    days_ahead = (0 - today.weekday()) % 7 or 7
                    return today + timedelta(days=days_ahead)
                elif "friday" in deadline_str_lower:
                    days_ahead = (4 - today.weekday()) % 7 or 7
                    return today + timedelta(days=days_ahead)
                elif "tomorrow" in deadline_str_lower:
                    return today + timedelta(days=1)
                elif "eod" in text.lower() or "end of day" in text.lower():
                    return today.replace(hour=17, minute=0, second=0)

        return None

    def _determine_priority(self, text: str) -> str:
        """Determine priority level from text"""
        text_lower = text.lower()

        for word in self.high_priority_words:
            if word in text_lower:
                return "high"

        for word in self.medium_priority_words:
            if word in text_lower:
                return "medium"

        for word in self.low_priority_words:
            if word in text_lower:
                return "low"

        return "medium"  # Default priority

    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract who the action is assigned to"""
        # Look for names after common assignment phrases
        patterns = [
            r"@(\w+)",  # Mentions
            r"assigned to\s+(\w+)",
            r"(\w+),?\s+please",
            r"(\w+),?\s+can you",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _calculate_importance(self, email: Email, action_items: List[ActionItem]) -> float:
        """Calculate email importance score (0-1)"""
        score = 0.5  # Base score

        # Increase for action items
        score += min(0.2, len(action_items) * 0.05)

        # Check for important keywords
        important_keywords = ["urgent", "important", "critical", "asap", "deadline"]
        text = f"{email.subject} {email.body}".lower()

        for keyword in important_keywords:
            if keyword in text:
                score += 0.1

        # Check for high priority action items
        high_priority_count = sum(1 for item in action_items if item.priority == "high")
        score += min(0.2, high_priority_count * 0.1)

        return min(1.0, score)

    def _determine_urgency(self, email: Email, action_items: List[ActionItem]) -> str:
        """Determine urgency level"""
        # Check for urgent keywords
        urgent_keywords = ["urgent", "asap", "immediately", "today", "now"]
        text = f"{email.subject} {email.body}".lower()

        for keyword in urgent_keywords:
            if keyword in text:
                return "urgent"

        # Check action item deadlines
        today = datetime.now()
        for item in action_items:
            if item.deadline:
                days_until = (item.deadline - today).days
                if days_until <= 1:
                    return "urgent"
                elif days_until <= 3:
                    return "high"

        # Default based on importance
        importance = self._calculate_importance(email, action_items)
        if importance > 0.8:
            return "high"
        elif importance > 0.5:
            return "medium"
        else:
            return "low"

    def _extract_entities(self, email: Email) -> Dict[str, List[str]]:
        """Extract named entities from email"""
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "money": [],
            "products": []
        }

        text = f"{email.subject} {email.body}"

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["people"].append(ent.text)
                elif ent.label_ in ["ORG", "COMPANY"]:
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["LOC", "GPE"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
                elif ent.label_ == "PRODUCT":
                    entities["products"].append(ent.text)
        else:
            # Fallback: Simple pattern matching
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities["people"] = re.findall(email_pattern, text)

            # Extract money amounts
            money_pattern = r'\$[\d,]+\.?\d*'
            entities["money"] = re.findall(money_pattern, text)

            # Extract dates
            date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}'
            entities["dates"] = re.findall(date_pattern, text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _process_attachments(self, email: Email) -> List[AttachmentMetadata]:
        """Process email attachments (simulated)"""
        attachments = []

        # Simulate attachment detection from email body
        attachment_indicators = [
            r"attached (?:is|are) (.*?)(?:\.|,|$)",
            r"see attached (.*?)(?:\.|,|$)",
            r"attachment: (.*?)(?:\.|,|$)",
            r"please find attached (.*?)(?:\.|,|$)",
        ]

        for pattern in attachment_indicators:
            matches = re.finditer(pattern, email.body, re.IGNORECASE)
            for match in matches:
                attachment_name = match.group(1).strip()

                # Infer file type
                file_type = "document"
                mime_type = "application/octet-stream"

                if "spreadsheet" in attachment_name.lower() or ".xlsx" in attachment_name:
                    file_type = "spreadsheet"
                    mime_type = "application/vnd.ms-excel"
                elif "pdf" in attachment_name.lower():
                    file_type = "pdf"
                    mime_type = "application/pdf"
                elif "image" in attachment_name.lower() or any(ext in attachment_name for ext in [".jpg", ".png"]):
                    file_type = "image"
                    mime_type = "image/jpeg"

                attachments.append(AttachmentMetadata(
                    filename=attachment_name,
                    file_type=file_type,
                    size_bytes=1024 * 50,  # Simulated size
                    mime_type=mime_type,
                    extracted_text=None,
                    summary=f"Attachment: {attachment_name}"
                ))

        return attachments

    def _generate_summary(self, email: Email) -> str:
        """Generate email summary"""
        # Simple extractive summary
        sentences = email.body.split('.')[:3]  # First 3 sentences
        summary = '. '.join(s.strip() for s in sentences if s.strip())

        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary

    def _analyze_sentiment(self, email: Email) -> str:
        """Analyze email sentiment"""
        text = f"{email.subject} {email.body}".lower()

        positive_words = ["great", "excellent", "good", "happy", "pleased", "thank"]
        negative_words = ["problem", "issue", "error", "fail", "wrong", "bad"]
        neutral_words = ["update", "inform", "report", "status"]

        positive_score = sum(1 for word in positive_words if word in text)
        negative_score = sum(1 for word in negative_words if word in text)
        neutral_score = sum(1 for word in neutral_words if word in text)

        if positive_score > negative_score and positive_score > neutral_score:
            return "positive"
        elif negative_score > positive_score and negative_score > neutral_score:
            return "negative"
        else:
            return "neutral"

    def _extract_topics(self, email: Email) -> List[str]:
        """Extract topics from email"""
        topics = []

        # Topic patterns
        topic_patterns = {
            "meeting": ["meeting", "schedule", "calendar", "appointment"],
            "budget": ["budget", "expense", "cost", "financial"],
            "project": ["project", "milestone", "deliverable", "timeline"],
            "support": ["help", "issue", "problem", "ticket"],
            "review": ["review", "feedback", "evaluation", "assessment"],
            "report": ["report", "analysis", "metrics", "data"],
        }

        text = f"{email.subject} {email.body}".lower()

        for topic, keywords in topic_patterns.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)

        return topics[:5]  # Limit to 5 topics

    def _generate_embeddings(self, email: Email) -> np.ndarray:
        """Generate semantic embeddings for email"""
        text = f"{email.subject} {email.body}"
        embeddings = self.sentence_model.encode([text])[0]
        return embeddings

    def _store_in_graph(self, email: Email, analysis: EmailAnalysis):
        """Store analysis results in graph database"""
        # Create document data
        doc_data = {
            "type": DocumentType.EMAIL.value,
            "title": email.subject,
            "content": email.body,
            "topics": analysis.topics,
            "metadata": {
                "importance_score": analysis.importance_score,
                "urgency_level": analysis.urgency_level,
                "sentiment": analysis.sentiment,
                "action_item_count": len(analysis.action_items),
                "entity_count": sum(len(v) for v in analysis.key_entities.values()),
                "attachment_count": len(analysis.attachments)
            }
        }

        # Store document
        doc_id = self.graph_service.store_document(doc_data)

        # Store action items as separate nodes with relationships
        for action_item in analysis.action_items:
            action_data = {
                "type": "action_item",
                "task": action_item.task,
                "priority": action_item.priority,
                "deadline": action_item.deadline.isoformat() if action_item.deadline else None,
                "assigned_to": action_item.assigned_to,
                "parent_document": doc_id
            }
            # In a real implementation, this would create ACTION_ITEM nodes

    def hybrid_search(
        self,
        query: str,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.5,
        graph_weight: float = 0.2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining keyword, semantic, and graph search

        Args:
            query: Search query
            keyword_weight: Weight for keyword matching (0-1)
            semantic_weight: Weight for semantic similarity (0-1)
            graph_weight: Weight for graph relationships (0-1)
            limit: Maximum results to return

        Returns:
            Ranked list of relevant documents
        """
        results = []

        # 1. Keyword search
        keyword_results = self._keyword_search(query)

        # 2. Semantic search
        semantic_results = self._semantic_search(query)

        # 3. Graph-based search
        graph_results = self._graph_search(query)

        # Combine and rank results
        all_docs = {}

        # Add keyword results
        for doc in keyword_results:
            doc_id = doc.get("id")
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "document": doc,
                    "keyword_score": 0,
                    "semantic_score": 0,
                    "graph_score": 0
                }
            all_docs[doc_id]["keyword_score"] = doc.get("score", 0)

        # Add semantic results
        for doc in semantic_results:
            doc_id = doc.get("id")
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "document": doc,
                    "keyword_score": 0,
                    "semantic_score": 0,
                    "graph_score": 0
                }
            all_docs[doc_id]["semantic_score"] = doc.get("score", 0)

        # Add graph results
        for doc in graph_results:
            doc_id = doc.get("id")
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "document": doc,
                    "keyword_score": 0,
                    "semantic_score": 0,
                    "graph_score": 0
                }
            all_docs[doc_id]["graph_score"] = doc.get("score", 0)

        # Calculate hybrid scores
        for doc_id, doc_data in all_docs.items():
            hybrid_score = (
                keyword_weight * doc_data["keyword_score"] +
                semantic_weight * doc_data["semantic_score"] +
                graph_weight * doc_data["graph_score"]
            )
            doc_data["hybrid_score"] = hybrid_score
            results.append(doc_data)

        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results[:limit]

    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Keyword-based search"""
        # Simple TF-IDF style matching (simplified)
        results = []
        query_words = set(query.lower().split())

        # In production, this would query an index
        # For now, return simulated results
        return results

    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Semantic similarity search using embeddings"""
        query_embedding = self.sentence_model.encode([query])[0]

        # In production, this would search a vector database
        # For now, return simulated results
        return []

    def _graph_search(self, query: str) -> List[Dict[str, Any]]:
        """Graph-based search using relationships"""
        # Extract topics from query
        query_topics = self._extract_topics(Email(subject=query, body=""))

        # Search for documents with similar topics
        similar_docs = self.graph_service.search_similar_documents(
            {"topics": query_topics},
            limit=10
        )

        return similar_docs