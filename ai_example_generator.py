#!/usr/bin/env python3
"""
AI-Powered Example Generator with LLM Analysis
Uses factory pattern to generate diverse email examples and analyze them with LLMs
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import anthropic

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_API_URL = "http://localhost:8000"

@dataclass
class EmailExample:
    """Email example with metadata"""
    subject: str
    body: str
    generated_by: str
    topic: str
    confidence: float
    features: Dict[str, Any]
    analysis: Dict[str, Any]
    timestamp: str

class EmailGenerator(ABC):
    """Abstract base class for email generators (Factory Pattern)"""

    @abstractmethod
    def generate(self, topic: str, count: int = 1) -> List[Dict[str, str]]:
        """Generate email examples for a topic"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get generator name"""
        pass

class TemplateEmailGenerator(EmailGenerator):
    """Template-based email generator"""

    def __init__(self):
        self.templates = {
            "work": [
                ("Project Update: {project}", "Status report for {project}. Current progress: {progress}%. Next milestone: {milestone}."),
                ("Meeting Request: {topic}", "Please join us for a meeting about {topic} on {date} at {time}."),
                ("Deadline Reminder: {task}", "This is a reminder that {task} is due on {date}. Please prioritize completion."),
            ],
            "finance": [
                ("Q{quarter} Financial Report", "Revenue: ${revenue}M. Profit: ${profit}M. Growth: {growth}% YoY."),
                ("Invoice {invoice_num}", "Invoice for ${amount}. Payment terms: Net {days}. Due date: {date}."),
                ("Budget Approval Required", "The {department} budget of ${amount} requires your approval by {date}."),
            ],
            "personal": [
                ("Hey! {occasion}", "{greeting}! Hope you're doing well. {message}"),
                ("Catching up", "It's been too long since we last talked. {message}"),
                ("Quick question", "Hope you're well. I wanted to ask about {topic}."),
            ],
            "support": [
                ("Support Ticket #{ticket}", "Your support request regarding {issue} has been received. Response time: {time}."),
                ("Issue Resolution", "The {issue} has been resolved. {resolution_details}."),
                ("Support Follow-up", "Following up on ticket #{ticket}. {status_update}."),
            ],
        }

    def generate(self, topic: str, count: int = 1) -> List[Dict[str, str]]:
        """Generate emails from templates"""
        if topic not in self.templates:
            topic = "work"  # Default

        emails = []
        templates = self.templates[topic]

        for i in range(count):
            template_idx = i % len(templates)
            subject_template, body_template = templates[template_idx]

            # Generate sample data
            email = {
                "subject": subject_template.format(
                    project=f"Project Alpha {i+1}",
                    topic="Strategic Planning",
                    task="Deliverable Review",
                    quarter=f"{(i%4)+1}",
                    invoice_num=f"INV-{1000+i}",
                    department="Engineering",
                    occasion="Long Time",
                    greeting="Hi there",
                    ticket=f"{5000+i}",
                    issue="Login Problem",
                    date="Next Friday"
                ),
                "body": body_template.format(
                    progress=f"{65+i*5}",
                    milestone="Beta Release",
                    date="Next Friday",
                    time="2:00 PM",
                    revenue=f"{2.5+i*0.5:.1f}",
                    profit=f"{0.5+i*0.1:.1f}",
                    growth=f"{15+i}",
                    amount=f"{5000+i*500}",
                    days="30",
                    message="Would love to catch up soon!",
                    issue="login timeout issue",
                    resolution_details="System configuration updated",
                    status_update="Issue has been marked as resolved"
                )
            }
            emails.append(email)

        return emails

    def get_name(self) -> str:
        return "Template Generator"

class AnthropicEmailGenerator(EmailGenerator):
    """Anthropic Claude-powered email generator"""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, topic: str, count: int = 1) -> List[Dict[str, str]]:
        """Generate emails using Claude"""

        prompt = f"""Generate {count} realistic email example(s) for the topic: {topic}

For each email, provide:
1. A realistic subject line
2. An appropriate email body (2-4 sentences)

Return ONLY valid JSON in this exact format:
[
  {{
    "subject": "email subject here",
    "body": "email body here"
  }}
]

Make the emails diverse and realistic for the {topic} category."""

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                emails = json.loads(json_str)
                return emails[:count]
            else:
                return []

        except Exception as e:
            print(f"Error generating with Claude: {e}")
            return []

    def get_name(self) -> str:
        return "Anthropic Claude"

class EmailGeneratorFactory:
    """Factory for creating email generators"""

    @staticmethod
    def create_generator(generator_type: str) -> EmailGenerator:
        """Create appropriate email generator"""

        generators = {
            "template": TemplateEmailGenerator,
            "anthropic": AnthropicEmailGenerator,
        }

        if generator_type not in generators:
            raise ValueError(f"Unknown generator type: {generator_type}")

        return generators[generator_type]()

    @staticmethod
    def get_available_generators() -> List[str]:
        """Get list of available generators"""
        available = ["template"]

        if ANTHROPIC_API_KEY:
            available.append("anthropic")

        return available

class FeatureExtractor(ABC):
    """Abstract feature extractor (Factory Pattern)"""

    @abstractmethod
    def extract(self, email: Dict[str, str]) -> Dict[str, Any]:
        """Extract features from email"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get extractor name"""
        pass

class BasicFeatureExtractor(FeatureExtractor):
    """Extract basic text features"""

    def extract(self, email: Dict[str, str]) -> Dict[str, Any]:
        """Extract basic features"""
        subject = email.get("subject", "")
        body = email.get("body", "")
        full_text = f"{subject} {body}"

        words = full_text.lower().split()

        return {
            "word_count": len(words),
            "char_count": len(full_text),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "has_numbers": any(char.isdigit() for char in full_text),
            "has_currency": "$" in full_text,
            "subject_length": len(subject),
            "body_length": len(body),
            "exclamation_count": full_text.count("!"),
            "question_count": full_text.count("?")
        }

    def get_name(self) -> str:
        return "Basic Features"

class SentimentFeatureExtractor(FeatureExtractor):
    """Extract sentiment features"""

    def __init__(self):
        self.positive_words = {"good", "great", "excellent", "happy", "successful", "progress", "achieved"}
        self.negative_words = {"bad", "poor", "failed", "problem", "issue", "error", "urgent", "critical"}
        self.urgency_words = {"urgent", "asap", "immediate", "critical", "emergency", "now"}

    def extract(self, email: Dict[str, str]) -> Dict[str, Any]:
        """Extract sentiment features"""
        full_text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
        words = set(full_text.split())

        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        urgency_count = len(words & self.urgency_words)

        return {
            "positive_word_count": positive_count,
            "negative_word_count": negative_count,
            "urgency_word_count": urgency_count,
            "sentiment_score": positive_count - negative_count,
            "is_urgent": urgency_count > 0
        }

    def get_name(self) -> str:
        return "Sentiment Features"

class FeatureExtractorFactory:
    """Factory for creating feature extractors"""

    @staticmethod
    def create_extractor(extractor_type: str) -> FeatureExtractor:
        """Create appropriate feature extractor"""

        extractors = {
            "basic": BasicFeatureExtractor,
            "sentiment": SentimentFeatureExtractor,
        }

        if extractor_type not in extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        return extractors[extractor_type]()

    @staticmethod
    def get_all_extractors() -> List[FeatureExtractor]:
        """Get all available extractors"""
        return [
            BasicFeatureExtractor(),
            SentimentFeatureExtractor(),
        ]

class EmailAnalyzer:
    """Analyze emails using LLMs and classification system"""

    def __init__(self):
        self.anthropic_client = None
        if ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def classify_email(self, email: Dict[str, str], use_similarity: bool = False) -> Dict[str, Any]:
        """Classify email using API"""

        try:
            response = requests.post(
                f"{BASE_API_URL}/emails/classify",
                json={
                    "subject": email["subject"],
                    "body": email["body"],
                    "use_email_similarity": use_similarity
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    def analyze_with_llm(self, email: Dict[str, str], classification: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email with LLM"""

        if not self.anthropic_client:
            return {"error": "Anthropic API key not configured"}

        prompt = f"""Analyze this email classification result:

Email Subject: {email['subject']}
Email Body: {email['body']}

Predicted Topic: {classification.get('predicted_topic', 'unknown')}
Top 3 Scores: {dict(sorted(classification.get('topic_scores', {}).items(), key=lambda x: x[1], reverse=True)[:3])}

Provide a brief analysis (2-3 sentences) covering:
1. Whether the classification seems accurate
2. Key indicators that led to this classification
3. Any ambiguity or alternative classifications

Return ONLY the analysis text, no formatting."""

        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "analysis": message.content[0].text,
                "model": "claude-3-5-sonnet-20241022"
            }

        except Exception as e:
            return {"error": str(e)}

class ExampleGenerationPipeline:
    """Complete pipeline for generating and analyzing email examples"""

    def __init__(self):
        self.generator_factory = EmailGeneratorFactory()
        self.extractor_factory = FeatureExtractorFactory()
        self.analyzer = EmailAnalyzer()
        self.examples = []

    def run_pipeline(self, topic: str, count: int, generator_type: str = "template") -> List[EmailExample]:
        """Run complete generation and analysis pipeline"""

        print(f"\n{'='*80}")
        print(f"EMAIL GENERATION AND ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Topic: {topic}")
        print(f"Count: {count}")
        print(f"Generator: {generator_type}")

        # Step 1: Generate emails
        print(f"\n[1/5] Generating emails...")
        generator = self.generator_factory.create_generator(generator_type)
        emails = generator.generate(topic, count)
        print(f"  Generated: {len(emails)} emails")

        # Step 2: Extract features
        print(f"\n[2/5] Extracting features...")
        extractors = self.extractor_factory.get_all_extractors()

        for email in emails:
            email["features"] = {}
            for extractor in extractors:
                features = extractor.extract(email)
                email["features"][extractor.get_name()] = features

        print(f"  Extracted: {len(extractors)} feature sets per email")

        # Step 3: Classify emails
        print(f"\n[3/5] Classifying emails...")
        for email in emails:
            classification = self.analyzer.classify_email(email, use_similarity=False)
            email["classification"] = classification

        print(f"  Classified: {len(emails)} emails")

        # Step 4: LLM Analysis
        print(f"\n[4/5] Analyzing with LLM...")
        for email in emails:
            if "classification" in email and "error" not in email["classification"]:
                llm_analysis = self.analyzer.analyze_with_llm(email, email["classification"])
                email["llm_analysis"] = llm_analysis
            else:
                email["llm_analysis"] = {"error": "Classification failed"}

        print(f"  Analyzed: {len(emails)} emails")

        # Step 5: Create EmailExample objects
        print(f"\n[5/5] Packaging results...")
        examples = []
        for email in emails:
            classification = email.get("classification", {})

            example = EmailExample(
                subject=email["subject"],
                body=email["body"],
                generated_by=generator.get_name(),
                topic=classification.get("predicted_topic", "unknown"),
                confidence=max(classification.get("topic_scores", {}).values()) if classification.get("topic_scores") else 0.0,
                features=email.get("features", {}),
                analysis=email.get("llm_analysis", {}),
                timestamp=datetime.now().isoformat()
            )
            examples.append(example)
            self.examples.append(example)

        print(f"  Complete: {len(examples)} examples")
        print(f"\n{'='*80}")

        return examples

    def save_results(self, filename: str = "ai_generated_examples.json"):
        """Save all examples to JSON"""

        data = {
            "generation_session": {
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(self.examples),
                "generators_used": list(set(ex.generated_by for ex in self.examples)),
                "topics": list(set(ex.topic for ex in self.examples))
            },
            "examples": [asdict(ex) for ex in self.examples]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def print_summary(self):
        """Print summary of examples"""

        print(f"\n{'='*80}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Examples: {len(self.examples)}")

        # By generator
        by_generator = {}
        for ex in self.examples:
            by_generator[ex.generated_by] = by_generator.get(ex.generated_by, 0) + 1

        print(f"\nBy Generator:")
        for gen, count in by_generator.items():
            print(f"  {gen}: {count}")

        # By topic
        by_topic = {}
        for ex in self.examples:
            by_topic[ex.topic] = by_topic.get(ex.topic, 0) + 1

        print(f"\nBy Predicted Topic:")
        for topic, count in sorted(by_topic.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {count}")

        # Average confidence
        avg_confidence = sum(ex.confidence for ex in self.examples) / len(self.examples)
        print(f"\nAverage Confidence: {avg_confidence:.3f}")

def main():
    """Main demonstration"""

    print("\n" + "="*80)
    print("AI-POWERED EMAIL EXAMPLE GENERATOR")
    print("Factory Pattern with LLM Analysis")
    print("="*80)

    pipeline = ExampleGenerationPipeline()

    # Available generators
    available_generators = EmailGeneratorFactory.get_available_generators()
    print(f"\nAvailable generators: {', '.join(available_generators)}")

    # Topics to generate
    topics = ["work", "finance", "personal", "support"]

    # Generate examples for each topic
    for topic in topics:
        # Use template generator (always available)
        examples = pipeline.run_pipeline(
            topic=topic,
            count=3,
            generator_type="template"
        )

        # Display sample
        if examples:
            sample = examples[0]
            print(f"\n{'='*80}")
            print(f"SAMPLE EXAMPLE - {topic.upper()}")
            print(f"{'='*80}")
            print(f"Subject: {sample.subject}")
            print(f"Body: {sample.body}")
            print(f"Predicted Topic: {sample.topic} (confidence: {sample.confidence:.3f})")
            if "analysis" in sample.analysis:
                print(f"LLM Analysis: {sample.analysis['analysis']}")

    # If Anthropic is available, generate with Claude
    if "anthropic" in available_generators:
        print(f"\n{'='*80}")
        print(f"GENERATING WITH ANTHROPIC CLAUDE")
        print(f"{'='*80}")

        claude_examples = pipeline.run_pipeline(
            topic="work",
            count=2,
            generator_type="anthropic"
        )

    # Summary
    pipeline.print_summary()

    # Save results
    pipeline.save_results()

    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()