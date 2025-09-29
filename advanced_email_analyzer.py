#!/usr/bin/env python3
"""
Advanced Email Analysis System with Grammarly-like Features
Provides comprehensive email analysis including grammar, tone, clarity, and engagement metrics
"""

import json
import re
import spacy
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import numpy as np
from datetime import datetime
import statistics
from collections import Counter, defaultdict
import base64
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import os

# Load spaCy model for advanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class AnalysisCategory(Enum):
    """Categories of email analysis"""
    GRAMMAR = "grammar"
    CLARITY = "clarity"
    TONE = "tone"
    ENGAGEMENT = "engagement"
    STRUCTURE = "structure"
    SECURITY = "security"
    PROFESSIONALISM = "professionalism"
    ACCESSIBILITY = "accessibility"

class SeverityLevel(Enum):
    """Severity levels for issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AnalysisIssue:
    """Individual analysis issue with suggestions"""
    category: AnalysisCategory
    severity: SeverityLevel
    message: str
    suggestion: str
    position: Tuple[int, int]  # start, end character positions
    confidence: float
    rule_id: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailMetrics:
    """Comprehensive email metrics"""
    readability_score: float
    tone_score: float
    professionalism_score: float
    engagement_score: float
    clarity_score: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    average_sentence_length: float
    lexical_diversity: float
    passive_voice_percentage: float
    sentiment_polarity: float
    formality_index: float
    urgency_indicators: int
    call_to_action_strength: float

@dataclass
class AttachmentAnalysis:
    """Analysis of email attachments"""
    filename: str
    file_type: str
    size_bytes: int
    security_risk: str
    content_summary: Optional[str] = None
    text_extracted: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailAnalysisResult:
    """Complete email analysis result"""
    overall_score: float
    metrics: EmailMetrics
    issues: List[AnalysisIssue]
    suggestions: List[str]
    attachments: List[AttachmentAnalysis]
    processing_time: float
    analysis_timestamp: datetime
    improved_version: Optional[str] = None

class GrammarAnalyzer:
    """Advanced grammar analysis with contextual understanding"""

    def __init__(self):
        self.common_errors = self._load_common_errors()
        self.style_patterns = self._load_style_patterns()

    def _load_common_errors(self) -> Dict[str, Dict[str, str]]:
        """Load common grammar error patterns"""
        return {
            "apostrophe_errors": {
                r"\bits\s": "it's (it is/has) vs its (possessive)",
                r"\byour\s+welcome\b": "you're welcome (you are welcome)",
                r"\btheir\s+going\b": "they're going (they are going)",
                r"\bwhos\s+": "who's (who is/has) vs whose (possessive)"
            },
            "subject_verb_agreement": {
                r"\bdata\s+is\b": "data are (data is plural)",
                r"\beveryone\s+are\b": "everyone is (everyone is singular)",
                r"\bnone\s+are\b": "none is (none can be singular)"
            },
            "redundancy": {
                r"\bfree\s+gift\b": "gift (gifts are inherently free)",
                r"\badvance\s+planning\b": "planning (planning is advance preparation)",
                r"\bunexpected\s+surprise\b": "surprise (surprises are unexpected)"
            },
            "formality": {
                r"\bu\s+": "you (use full words in professional emails)",
                r"\br\s+": "are (avoid text speak)",
                r"\bthanks\b": "thank you (more formal)"
            }
        }

    def _load_style_patterns(self) -> Dict[str, List[str]]:
        """Load style improvement patterns"""
        return {
            "weak_verbs": ["is", "are", "was", "were", "have", "has", "had", "do", "does", "did"],
            "filler_words": ["really", "very", "quite", "rather", "somewhat", "actually", "basically"],
            "hedging_words": ["maybe", "perhaps", "possibly", "might", "could", "should"],
            "power_words": ["achieve", "implement", "deliver", "execute", "optimize", "enhance"],
            "transition_words": ["however", "therefore", "furthermore", "consequently", "meanwhile"]
        }

    def analyze_grammar(self, text: str) -> List[AnalysisIssue]:
        """Analyze grammar and return issues"""
        issues = []

        # Check common error patterns
        for category, patterns in self.common_errors.items():
            for pattern, suggestion in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    issues.append(AnalysisIssue(
                        category=AnalysisCategory.GRAMMAR,
                        severity=SeverityLevel.WARNING,
                        message=f"Potential {category.replace('_', ' ')} issue",
                        suggestion=suggestion,
                        position=(match.start(), match.end()),
                        confidence=0.8,
                        rule_id=f"grammar_{category}",
                        context={"pattern": pattern, "match": match.group()}
                    ))

        # Advanced spaCy analysis if available
        if nlp:
            issues.extend(self._spacy_grammar_analysis(text))

        return issues

    def _spacy_grammar_analysis(self, text: str) -> List[AnalysisIssue]:
        """Advanced grammar analysis using spaCy"""
        issues = []
        doc = nlp(text)

        # Check for sentence fragments
        for sent in doc.sents:
            if not self._has_main_verb(sent):
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.GRAMMAR,
                    severity=SeverityLevel.WARNING,
                    message="Possible sentence fragment",
                    suggestion="Ensure sentence has a main verb",
                    position=(sent.start_char, sent.end_char),
                    confidence=0.7,
                    rule_id="grammar_fragment"
                ))

        # Check for run-on sentences
        for sent in doc.sents:
            if len(sent) > 30:  # Arbitrary threshold
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.CLARITY,
                    severity=SeverityLevel.INFO,
                    message="Long sentence may be hard to read",
                    suggestion="Consider breaking into shorter sentences",
                    position=(sent.start_char, sent.end_char),
                    confidence=0.6,
                    rule_id="clarity_long_sentence"
                ))

        return issues

    def _has_main_verb(self, sentence) -> bool:
        """Check if sentence has a main verb"""
        for token in sentence:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return True
        return False

class ToneAnalyzer:
    """Analyze email tone and emotional characteristics"""

    def __init__(self):
        self.tone_indicators = self._load_tone_indicators()
        self.professional_vocabulary = self._load_professional_vocabulary()

    def _load_tone_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """Load tone indicator words and phrases"""
        return {
            "aggressive": {
                "words": ["demand", "insist", "must", "immediately", "unacceptable"],
                "phrases": ["you need to", "you should have", "this is ridiculous"]
            },
            "passive_aggressive": {
                "words": ["obviously", "clearly", "surely", "as I mentioned"],
                "phrases": ["per my last email", "as previously discussed", "just to clarify"]
            },
            "friendly": {
                "words": ["please", "thank you", "appreciate", "grateful", "wonderful"],
                "phrases": ["hope you're well", "thanks in advance", "looking forward"]
            },
            "formal": {
                "words": ["regarding", "pursuant", "henceforth", "aforementioned"],
                "phrases": ["I am writing to", "please find attached", "I would like to request"]
            },
            "urgent": {
                "words": ["urgent", "asap", "immediately", "critical", "deadline"],
                "phrases": ["time sensitive", "urgent attention", "by end of day"]
            }
        }

    def _load_professional_vocabulary(self) -> Dict[str, List[str]]:
        """Load professional vocabulary sets"""
        return {
            "business_terms": ["leverage", "synergy", "optimize", "strategic", "initiative"],
            "academic_terms": ["analyze", "evaluate", "methodology", "framework", "hypothesis"],
            "technical_terms": ["implement", "configure", "deploy", "integrate", "architecture"],
            "casual_terms": ["awesome", "cool", "stuff", "things", "guys"]
        }

    def analyze_tone(self, text: str) -> Tuple[Dict[str, float], List[AnalysisIssue]]:
        """Analyze tone and return scores and issues"""
        tone_scores = {}
        issues = []

        # Calculate tone scores
        for tone, indicators in self.tone_indicators.items():
            score = self._calculate_tone_score(text, indicators)
            tone_scores[tone] = score

            # Generate issues for problematic tones
            if tone in ["aggressive", "passive_aggressive"] and score > 0.3:
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.TONE,
                    severity=SeverityLevel.WARNING,
                    message=f"Email may sound {tone.replace('_', ' ')}",
                    suggestion=f"Consider using more neutral language",
                    position=(0, len(text)),
                    confidence=score,
                    rule_id=f"tone_{tone}"
                ))

        # Check formality level
        formality_score = self._calculate_formality_score(text)
        tone_scores["formality"] = formality_score

        if formality_score < 0.3:
            issues.append(AnalysisIssue(
                category=AnalysisCategory.PROFESSIONALISM,
                severity=SeverityLevel.INFO,
                message="Email tone may be too casual for professional context",
                suggestion="Consider using more formal language",
                position=(0, len(text)),
                confidence=1.0 - formality_score,
                rule_id="tone_formality"
            ))

        return tone_scores, issues

    def _calculate_tone_score(self, text: str, indicators: Dict[str, List[str]]) -> float:
        """Calculate tone score based on indicators"""
        text_lower = text.lower()
        total_indicators = 0
        found_indicators = 0

        for category, items in indicators.items():
            total_indicators += len(items)
            for item in items:
                if item.lower() in text_lower:
                    found_indicators += 1

        return found_indicators / total_indicators if total_indicators > 0 else 0.0

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score based on vocabulary"""
        text_lower = text.lower()
        formal_count = 0
        casual_count = 0

        # Count formal indicators
        for term in self.professional_vocabulary["business_terms"] + self.professional_vocabulary["academic_terms"]:
            if term.lower() in text_lower:
                formal_count += 1

        # Count casual indicators
        for term in self.professional_vocabulary["casual_terms"]:
            if term.lower() in text_lower:
                casual_count += 1

        # Include contractions as casual
        contractions = re.findall(r"\w+'\w+", text)
        casual_count += len(contractions)

        total = formal_count + casual_count
        return formal_count / total if total > 0 else 0.5

class ClarityAnalyzer:
    """Analyze email clarity and readability"""

    def __init__(self):
        self.jargon_terms = self._load_jargon_terms()
        self.complex_words = self._load_complex_words()

    def _load_jargon_terms(self) -> Dict[str, str]:
        """Load jargon terms with plain language alternatives"""
        return {
            "utilize": "use",
            "facilitate": "help",
            "implement": "do",
            "optimize": "improve",
            "leverage": "use",
            "synergize": "work together",
            "paradigm": "model",
            "ideate": "brainstorm",
            "actionable": "useful",
            "deliverable": "result"
        }

    def _load_complex_words(self) -> List[str]:
        """Load complex words that could be simplified"""
        return [
            "accommodate", "acknowledge", "acquisition", "additional", "alternative",
            "anticipate", "appreciate", "appropriate", "approximately", "assistance",
            "beneficial", "capability", "circumstances", "collaborate", "commence",
            "communicate", "compensation", "comprehensive", "consequently", "consideration"
        ]

    def analyze_clarity(self, text: str) -> Tuple[float, List[AnalysisIssue]]:
        """Analyze clarity and return score and issues"""
        issues = []

        # Check for jargon
        for jargon, alternative in self.jargon_terms.items():
            if jargon.lower() in text.lower():
                # Find position
                match = re.search(re.escape(jargon), text, re.IGNORECASE)
                if match:
                    issues.append(AnalysisIssue(
                        category=AnalysisCategory.CLARITY,
                        severity=SeverityLevel.INFO,
                        message=f"Consider simpler alternative to '{jargon}'",
                        suggestion=f"Use '{alternative}' instead of '{jargon}'",
                        position=(match.start(), match.end()),
                        confidence=0.8,
                        rule_id="clarity_jargon"
                    ))

        # Check sentence length
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            if len(words) > 25:
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.CLARITY,
                    severity=SeverityLevel.WARNING,
                    message=f"Sentence {i+1} is very long ({len(words)} words)",
                    suggestion="Consider breaking into shorter sentences",
                    position=(0, len(sentence)),
                    confidence=0.7,
                    rule_id="clarity_sentence_length"
                ))

        # Calculate readability score (simplified Flesch Reading Ease)
        readability_score = self._calculate_readability(text)

        # Check for passive voice
        passive_voice_issues = self._detect_passive_voice(text)
        issues.extend(passive_voice_issues)

        return readability_score, issues

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (0-100, higher is more readable)"""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0

        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))

    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count"""
        words = text.lower().split()
        syllable_count = 0

        for word in words:
            # Remove punctuation
            word = re.sub(r'[^a-z]', '', word)
            if not word:
                continue

            # Count vowel groups
            vowels = 'aeiouy'
            syllables = 0
            prev_was_vowel = False

            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False

            # Handle silent e
            if word.endswith('e') and syllables > 1:
                syllables -= 1

            # Every word has at least one syllable
            syllables = max(1, syllables)
            syllable_count += syllables

        return syllable_count

    def _detect_passive_voice(self, text: str) -> List[AnalysisIssue]:
        """Detect passive voice constructions"""
        issues = []

        # Simple passive voice detection
        passive_patterns = [
            r'\b(was|were|is|are|been|being)\s+\w+ed\b',
            r'\b(was|were|is|are|been|being)\s+\w+en\b'
        ]

        for pattern in passive_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.CLARITY,
                    severity=SeverityLevel.INFO,
                    message="Passive voice detected",
                    suggestion="Consider using active voice for clarity",
                    position=(match.start(), match.end()),
                    confidence=0.6,
                    rule_id="clarity_passive_voice"
                ))

        return issues

class SecurityAnalyzer:
    """Analyze email security concerns"""

    def __init__(self):
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.sensitive_data_patterns = self._load_sensitive_data_patterns()

    def _load_suspicious_patterns(self) -> Dict[str, str]:
        """Load patterns that might indicate security issues"""
        return {
            r'click\s+here': "Generic 'click here' links can be suspicious",
            r'urgent\s+action\s+required': "Urgency tactics are common in phishing",
            r'verify\s+your\s+account': "Account verification requests should be verified",
            r'suspended.*account': "Account suspension claims should be verified",
            r'prize.*won': "Unexpected prize notifications are often scams"
        }

    def _load_sensitive_data_patterns(self) -> Dict[str, str]:
        """Load patterns for sensitive data detection"""
        return {
            r'\b\d{3}-\d{2}-\d{4}\b': "Social Security Number detected",
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b': "Credit card number pattern detected",
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': "Email address detected",
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': "Phone number detected"
        }

    def analyze_security(self, text: str, sender: str = "") -> List[AnalysisIssue]:
        """Analyze security concerns"""
        issues = []

        # Check for suspicious patterns
        for pattern, description in self.suspicious_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.SECURITY,
                    severity=SeverityLevel.WARNING,
                    message="Potentially suspicious content detected",
                    suggestion=description,
                    position=(match.start(), match.end()),
                    confidence=0.7,
                    rule_id="security_suspicious"
                ))

        # Check for sensitive data
        for pattern, description in self.sensitive_data_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append(AnalysisIssue(
                    category=AnalysisCategory.SECURITY,
                    severity=SeverityLevel.ERROR,
                    message="Sensitive data detected",
                    suggestion=f"{description} - Consider if this should be shared via email",
                    position=(match.start(), match.end()),
                    confidence=0.9,
                    rule_id="security_sensitive_data"
                ))

        return issues

class AttachmentAnalyzer:
    """Analyze email attachments"""

    def __init__(self):
        self.risky_extensions = {
            '.exe', '.bat', '.cmd', '.scr', '.pif', '.vbs', '.js',
            '.jar', '.app', '.deb', '.pkg', '.dmg'
        }
        self.safe_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.txt', '.csv', '.jpg', '.jpeg', '.png', '.gif'
        }

    def analyze_attachment(self, filename: str, content: bytes = None) -> AttachmentAnalysis:
        """Analyze a single attachment"""
        file_ext = os.path.splitext(filename)[1].lower()
        file_type = mimetypes.guess_type(filename)[0] or "unknown"

        # Determine security risk
        if file_ext in self.risky_extensions:
            security_risk = "high"
        elif file_ext in self.safe_extensions:
            security_risk = "low"
        else:
            security_risk = "medium"

        analysis = AttachmentAnalysis(
            filename=filename,
            file_type=file_type,
            size_bytes=len(content) if content else 0,
            security_risk=security_risk,
            metadata={
                "extension": file_ext,
                "analysis_timestamp": datetime.now().isoformat()
            }
        )

        # Extract text if possible
        if content and file_ext == '.txt':
            try:
                analysis.text_extracted = content.decode('utf-8')
                analysis.content_summary = f"Text file with {len(analysis.text_extracted.split())} words"
            except UnicodeDecodeError:
                analysis.content_summary = "Binary content detected"

        return analysis

class AdvancedEmailAnalyzer:
    """Main advanced email analyzer orchestrating all analysis components"""

    def __init__(self):
        self.grammar_analyzer = GrammarAnalyzer()
        self.tone_analyzer = ToneAnalyzer()
        self.clarity_analyzer = ClarityAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.attachment_analyzer = AttachmentAnalyzer()

    def analyze_email(self, subject: str, body: str, sender: str = "", attachments: List[Dict] = None) -> EmailAnalysisResult:
        """Perform comprehensive email analysis"""
        start_time = datetime.now()
        all_issues = []
        attachment_analyses = []

        # Combine subject and body for analysis
        full_text = f"{subject}\n\n{body}"

        # Grammar analysis
        grammar_issues = self.grammar_analyzer.analyze_grammar(full_text)
        all_issues.extend(grammar_issues)

        # Tone analysis
        tone_scores, tone_issues = self.tone_analyzer.analyze_tone(full_text)
        all_issues.extend(tone_issues)

        # Clarity analysis
        readability_score, clarity_issues = self.clarity_analyzer.analyze_clarity(full_text)
        all_issues.extend(clarity_issues)

        # Security analysis
        security_issues = self.security_analyzer.analyze_security(full_text, sender)
        all_issues.extend(security_issues)

        # Attachment analysis
        if attachments:
            for attachment in attachments:
                analysis = self.attachment_analyzer.analyze_attachment(
                    attachment.get('filename', 'unknown'),
                    attachment.get('content')
                )
                attachment_analyses.append(analysis)

        # Calculate metrics
        metrics = self._calculate_metrics(full_text, tone_scores, readability_score)

        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, all_issues)

        # Generate suggestions
        suggestions = self._generate_suggestions(all_issues, metrics)

        # Generate improved version
        improved_version = self._generate_improved_version(subject, body, all_issues)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return EmailAnalysisResult(
            overall_score=overall_score,
            metrics=metrics,
            issues=all_issues,
            suggestions=suggestions,
            attachments=attachment_analyses,
            processing_time=processing_time,
            analysis_timestamp=end_time,
            improved_version=improved_version
        )

    def _calculate_metrics(self, text: str, tone_scores: Dict[str, float], readability_score: float) -> EmailMetrics:
        """Calculate comprehensive email metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Calculate various metrics
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraph_count = len(paragraphs)

        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Lexical diversity (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0

        # Sentiment analysis (simplified)
        positive_words = ['good', 'great', 'excellent', 'pleased', 'happy', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'disappointed', 'angry', 'frustrated']

        positive_count = sum(1 for word in words if word.lower() in positive_words)
        negative_count = sum(1 for word in words if word.lower() in negative_words)
        sentiment_polarity = (positive_count - negative_count) / word_count if word_count > 0 else 0

        # Urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'critical', 'deadline', 'rush']
        urgency_indicators = sum(1 for word in words if word.lower() in urgency_words)

        # Call to action strength
        action_words = ['please', 'click', 'download', 'register', 'subscribe', 'buy', 'contact']
        cta_strength = sum(1 for word in words if word.lower() in action_words) / word_count if word_count > 0 else 0

        return EmailMetrics(
            readability_score=readability_score,
            tone_score=tone_scores.get('friendly', 0.5),
            professionalism_score=tone_scores.get('formality', 0.5),
            engagement_score=min(1.0, cta_strength * 2),
            clarity_score=readability_score / 100,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            average_sentence_length=avg_sentence_length,
            lexical_diversity=lexical_diversity,
            passive_voice_percentage=0.0,  # Would need more sophisticated analysis
            sentiment_polarity=sentiment_polarity,
            formality_index=tone_scores.get('formality', 0.5),
            urgency_indicators=urgency_indicators,
            call_to_action_strength=cta_strength
        )

    def _calculate_overall_score(self, metrics: EmailMetrics, issues: List[AnalysisIssue]) -> float:
        """Calculate overall email quality score"""
        base_score = 0.8  # Start with good score

        # Deduct points for issues
        error_count = sum(1 for issue in issues if issue.severity == SeverityLevel.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == SeverityLevel.WARNING)

        base_score -= (error_count * 0.15)  # Major deduction for errors
        base_score -= (warning_count * 0.05)  # Minor deduction for warnings

        # Adjust based on metrics
        if metrics.readability_score < 30:  # Very hard to read
            base_score -= 0.2
        elif metrics.readability_score > 70:  # Easy to read
            base_score += 0.1

        if metrics.average_sentence_length > 30:  # Very long sentences
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    def _generate_suggestions(self, issues: List[AnalysisIssue], metrics: EmailMetrics) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []

        # Group issues by category
        issue_categories = defaultdict(list)
        for issue in issues:
            issue_categories[issue.category].append(issue)

        # Generate category-specific suggestions
        if AnalysisCategory.GRAMMAR in issue_categories:
            suggestions.append("Review grammar and consider using a grammar checker for complex sentences")

        if AnalysisCategory.CLARITY in issue_categories:
            suggestions.append("Simplify language and break long sentences into shorter ones")

        if AnalysisCategory.TONE in issue_categories:
            suggestions.append("Adjust tone to be more professional and less aggressive")

        if AnalysisCategory.SECURITY in issue_categories:
            suggestions.append("Review content for sensitive information and suspicious patterns")

        # Metric-based suggestions
        if metrics.readability_score < 50:
            suggestions.append("Improve readability by using simpler words and shorter sentences")

        if metrics.average_sentence_length > 25:
            suggestions.append("Break long sentences (>25 words) into shorter, clearer statements")

        if metrics.engagement_score < 0.3:
            suggestions.append("Add clearer calls-to-action to improve engagement")

        if not suggestions:
            suggestions.append("Email quality is good - no major improvements needed")

        return suggestions

    def _generate_improved_version(self, subject: str, body: str, issues: List[AnalysisIssue]) -> str:
        """Generate an improved version of the email"""
        improved_subject = subject
        improved_body = body

        # Apply simple improvements based on issues
        for issue in issues:
            if issue.rule_id == "clarity_jargon" and "alternative" in issue.suggestion:
                # Extract alternative from suggestion
                parts = issue.suggestion.split("'")
                if len(parts) >= 4:
                    old_word = parts[3]
                    new_word = parts[1]
                    improved_body = improved_body.replace(old_word, new_word)

        # Add improved formatting
        if improved_body:
            # Ensure proper paragraph breaks
            improved_body = re.sub(r'\n{3,}', '\n\n', improved_body)

            # Add greeting if missing
            if not re.match(r'^(hi|hello|dear|greetings)', improved_body.lower().strip()):
                improved_body = "Hello,\n\n" + improved_body

            # Add closing if missing
            if not re.search(r'(regards|sincerely|thanks|best)', improved_body.lower()):
                improved_body += "\n\nBest regards"

        return f"Subject: {improved_subject}\n\n{improved_body}"

def main():
    """Demonstrate the advanced email analyzer"""
    print("🔍 ADVANCED EMAIL ANALYSIS SYSTEM")
    print("Grammarly-like features for comprehensive email analysis")
    print("=" * 70)

    # Initialize analyzer
    analyzer = AdvancedEmailAnalyzer()

    # Test emails with various issues
    test_emails = [
        {
            "subject": "urgent meeting tommorow",
            "body": "Hi team,\n\nWe have a urgent meeting tommorrow at 10 AM. Its very important that everyone attends. The data is clear and shows that we need to optimize our leverage points and utilize our synergies. Please confirm your attendance ASAP.\n\nThanks",
            "sender": "manager@company.com"
        },
        {
            "subject": "Project Update - Q4 Initiative",
            "body": "Dear Team,\n\nI hope this email finds you well. I wanted to provide an update on our Q4 initiative. The project is progressing smoothly, and we are on track to meet our deliverables.\n\nPlease review the attached document and provide your feedback by Friday.\n\nBest regards,\nProject Manager",
            "sender": "pm@company.com"
        }
    ]

    for i, email_data in enumerate(test_emails, 1):
        print(f"\n📧 ANALYZING EMAIL {i}")
        print("-" * 50)

        result = analyzer.analyze_email(
            subject=email_data["subject"],
            body=email_data["body"],
            sender=email_data["sender"]
        )

        print(f"Overall Score: {result.overall_score:.2f}/1.00")
        print(f"Processing Time: {result.processing_time:.3f}s")

        print(f"\n📊 METRICS:")
        print(f"  Readability: {result.metrics.readability_score:.1f}/100")
        print(f"  Word Count: {result.metrics.word_count}")
        print(f"  Avg Sentence Length: {result.metrics.average_sentence_length:.1f}")
        print(f"  Formality Index: {result.metrics.formality_index:.2f}")

        if result.issues:
            print(f"\n⚠️  ISSUES FOUND ({len(result.issues)}):")
            for issue in result.issues[:5]:  # Show first 5 issues
                print(f"  • {issue.category.value.title()}: {issue.message}")
                print(f"    Suggestion: {issue.suggestion}")

        if result.suggestions:
            print(f"\n💡 SUGGESTIONS:")
            for suggestion in result.suggestions[:3]:  # Show first 3 suggestions
                print(f"  • {suggestion}")

        if result.improved_version:
            print(f"\n✨ IMPROVED VERSION:")
            print(result.improved_version[:200] + "..." if len(result.improved_version) > 200 else result.improved_version)

    print(f"\n" + "=" * 70)
    print("🎯 ADVANCED EMAIL ANALYSIS COMPLETE")
    print("✓ Grammar, tone, clarity, and security analysis")
    print("✓ Actionable suggestions for improvement")
    print("✓ Professional-grade email insights")

if __name__ == "__main__":
    main()