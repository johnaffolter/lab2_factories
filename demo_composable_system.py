"""
Composable System Demonstration
Shows how to build custom ML pipelines using composable components
"""

from dataclasses import dataclass
from app.core.composable import global_registry, ComposablePipeline
from app.features.composable_generators import *  # Register all components
import json


@dataclass
class Email:
    subject: str
    body: str


def demo_component_registry():
    """Demo 1: Component Discovery"""
    print("=" * 70)
    print("DEMO 1: Component Registry & Discovery")
    print("=" * 70)
    print()

    # List all registered components
    components = global_registry.list_all()
    print(f"📦 Registered Components: {len(components)}")
    print()

    for comp in components:
        print(f"{comp.icon} {comp.name} v{comp.version}")
        print(f"   Type: {comp.type.value}")
        print(f"   Description: {comp.description}")
        print(f"   Tags: {', '.join(comp.tags)}")
        print(f"   Color: {comp.color}")
        print()

    # Search functionality
    print("-" * 70)
    print("🔍 Searching for 'sentiment' components:")
    results = global_registry.search("sentiment")
    for r in results:
        print(f"   {r.icon} {r.name}")
    print()


def demo_simple_pipeline():
    """Demo 2: Building and Executing a Simple Pipeline"""
    print("=" * 70)
    print("DEMO 2: Simple Pipeline Execution")
    print("=" * 70)
    print()

    # Create test email
    email = Email(
        subject="URGENT: Free Winner! Click Now!",
        body="Congratulations! You've won a million dollars. Click this link immediately to claim your prize!"
    )

    print("📧 Test Email:")
    print(f"   Subject: {email.subject}")
    print(f"   Body: {email.body[:60]}...")
    print()

    # Create individual components
    spam_detector = ComposableSpamGenerator()
    word_analyzer = ComposableWordLengthGenerator()
    sentiment_analyzer = ComposableSentimentAnalyzer()

    # Execute individually
    print("🔬 Individual Component Results:")
    print()

    spam_result = spam_detector.execute(email)
    print(f"{spam_detector.metadata.icon} Spam Detector:")
    print(f"   Spam Score: {spam_result['spam_score']:.2f}")
    print(f"   Keywords Found: {spam_result['spam_keywords']}")
    print(f"   Is Spam: {spam_result['has_spam']}")
    print()

    word_result = word_analyzer.execute(email)
    print(f"{word_analyzer.metadata.icon} Word Length Analyzer:")
    print(f"   Avg Word Length: {word_result['avg_word_length']}")
    print(f"   Total Words: {word_result['total_words']}")
    print(f"   Vocabulary Richness: {word_result['vocabulary_richness']}")
    print()

    sentiment_result = sentiment_analyzer.execute(email)
    print(f"{sentiment_analyzer.metadata.icon} Sentiment Analyzer:")
    print(f"   Sentiment: {sentiment_result['sentiment']}")
    print(f"   Confidence: {sentiment_result['confidence']}")
    print(f"   Tone: {sentiment_result['tone']}")
    print()


def demo_pipeline_composition():
    """Demo 3: Pipeline Composition with | Operator"""
    print("=" * 70)
    print("DEMO 3: Pipeline Composition")
    print("=" * 70)
    print()

    # Create components
    spam = ComposableSpamGenerator()
    words = ComposableWordLengthGenerator()
    sentiment = ComposableSentimentAnalyzer()
    urgency = ComposableUrgencyDetector()

    # Compose pipeline using | operator
    pipeline = spam | words | sentiment | urgency

    print("🔗 Pipeline Composition:")
    print("   Spam Detector | Word Analyzer | Sentiment | Urgency")
    print()

    # Test emails
    test_emails = [
        Email(
            "Team Meeting Reminder",
            "Just a friendly reminder about our team meeting tomorrow at 10am."
        ),
        Email(
            "URGENT: Server Down!",
            "Critical issue! Production server crashed. Need immediate attention ASAP!"
        ),
        Email(
            "Special Discount - Act Now",
            "Limited time offer! Get 50% off everything. Free shipping. Click here now!"
        )
    ]

    print("📊 Pipeline Results:")
    print()

    for i, email in enumerate(test_emails, 1):
        print(f"{i}. {email.subject}")

        # Note: For full pipeline we'd need to aggregate results
        # For now, show individual component outputs
        spam_result = spam.execute(email)
        sentiment_result = sentiment.execute(email)
        urgency_result = urgency.execute(email)

        print(f"   🚫 Spam: {spam_result['spam_score']:.2f}")
        print(f"   😊 Sentiment: {sentiment_result['sentiment']}")
        print(f"   ⚡ Urgency: {urgency_result['urgency_level']}")
        print()


def demo_pipeline_serialization():
    """Demo 4: Pipeline Serialization and Storage"""
    print("=" * 70)
    print("DEMO 4: Pipeline Serialization")
    print("=" * 70)
    print()

    # Create pipeline
    spam = ComposableSpamGenerator(config={"threshold": 0.5})
    sentiment = ComposableSentimentAnalyzer()

    pipeline = ComposablePipeline([spam, sentiment])
    pipeline.name = "Spam and Sentiment Analyzer"

    # Serialize to JSON
    pipeline_json = pipeline.to_json()

    print("💾 Serialized Pipeline:")
    print(pipeline_json)
    print()

    # Deserialize
    pipeline_data = json.loads(pipeline_json)
    restored_pipeline = ComposablePipeline.from_dict(pipeline_data, global_registry)

    print("✅ Pipeline Restored:")
    print(f"   Name: {restored_pipeline.name}")
    print(f"   Components: {len(restored_pipeline.components)}")
    print()


def demo_component_marketplace():
    """Demo 5: Component Marketplace View"""
    print("=" * 70)
    print("DEMO 5: Component Marketplace")
    print("=" * 70)
    print()

    # Export registry as marketplace data
    marketplace_data = global_registry.to_dict()

    print(f"🏪 Component Marketplace")
    print(f"   Total Components: {marketplace_data['total']}")
    print()

    # Group by type
    by_type = {}
    for comp in marketplace_data['components']:
        comp_type = comp['type']
        if comp_type not in by_type:
            by_type[comp_type] = []
        by_type[comp_type].append(comp)

    for comp_type, components in by_type.items():
        print(f"📂 {comp_type.upper()} ({len(components)} components)")
        for comp in components:
            print(f"   {comp['icon']} {comp['name']}")
            print(f"      Tags: {', '.join(comp['tags'])}")
        print()


def demo_custom_config():
    """Demo 6: Custom Component Configuration"""
    print("=" * 70)
    print("DEMO 6: Custom Component Configuration")
    print("=" * 70)
    print()

    # Create spam detector with custom config
    custom_spam = ComposableSpamGenerator(config={
        "keywords": ["bitcoin", "crypto", "investment", "returns"],
        "threshold": 0.25
    })

    print("⚙️ Custom Configuration:")
    print(f"   Component: {custom_spam.metadata.name}")
    print(f"   Custom Keywords: {custom_spam.config['keywords']}")
    print(f"   Threshold: {custom_spam.config['threshold']}")
    print()

    # Test with crypto-related email
    email = Email(
        "Exclusive Investment Opportunity",
        "Invest in Bitcoin today and see massive returns on your crypto investment!"
    )

    result = custom_spam.execute(email)
    print("📊 Result:")
    print(f"   Spam Score: {result['spam_score']:.2f}")
    print(f"   Found Keywords: {result['spam_keywords']}")
    print(f"   Is Spam: {result['has_spam']}")
    print()


def main():
    """Run all demos"""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║       COMPOSABLE ML SYSTEM DEMONSTRATION                          ║")
    print("║       Interactive Component Architecture                          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    demo_component_registry()
    input("Press Enter to continue to Demo 2...")
    print()

    demo_simple_pipeline()
    input("Press Enter to continue to Demo 3...")
    print()

    demo_pipeline_composition()
    input("Press Enter to continue to Demo 4...")
    print()

    demo_pipeline_serialization()
    input("Press Enter to continue to Demo 5...")
    print()

    demo_component_marketplace()
    input("Press Enter to continue to Demo 6...")
    print()

    demo_custom_config()

    print()
    print("=" * 70)
    print("✅ All Demonstrations Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Build a web UI for visual pipeline creation")
    print("  2. Add more composable components")
    print("  3. Implement plugin system for extensibility")
    print("  4. Create component marketplace")
    print("  5. Add real-time monitoring dashboard")
    print()


if __name__ == "__main__":
    main()