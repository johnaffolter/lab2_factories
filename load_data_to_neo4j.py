"""
Load Expanded Training Data into Neo4j
Includes classification, validation, and LLM-as-a-Judge scoring
"""

import json
from datetime import datetime
from app.services.mlops_neo4j_integration import get_knowledge_graph
from app.features.factory import FeatureGeneratorFactory
from app.models.similarity_model import EmailClassifierModel
from llm_judge_validator import LLMJudge
from dataclasses import dataclass

@dataclass
class Email:
    subject: str
    body: str


def load_expanded_data_to_neo4j(limit: int = None):
    """
    Load expanded training data into Neo4j with full pipeline
    """
    print("Loading Expanded Training Data into Neo4j")
    print("=" * 70)
    print()

    # Initialize components
    kg = get_knowledge_graph()
    factory = FeatureGeneratorFactory()
    model = EmailClassifierModel()
    judge = LLMJudge()

    # Load expanded dataset
    with open("data/expanded_training_emails.json", "r") as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]

    print(f"📧 Processing {len(emails)} emails...")
    print()

    # Statistics
    stats = {
        "processed": 0,
        "stored": 0,
        "validated": 0,
        "correct_predictions": 0,
        "topics": {},
        "validation_scores": []
    }

    # Process each email
    for i, email_data in enumerate(emails, 1):
        try:
            # Create email and extract features
            email = Email(
                subject=email_data["subject"],
                body=email_data["body"]
            )

            features = factory.generate_all_features(email)

            # Classify using model
            predicted_topic = model.predict(features)

            # Prepare classification result
            classification_result = {
                "predicted_label": predicted_topic,
                "confidence": 0.75,
                "all_scores": {predicted_topic: 0.75},
                "model_type": "EmailClassifierModel",
                "features": features
            }

            # Store in Neo4j
            email_id = kg.store_email_with_classification(
                {
                    "subject": email_data["subject"],
                    "body": email_data["body"],
                    "sender": email_data["sender"],
                    "timestamp": email_data["timestamp"]
                },
                classification_result
            )

            # Store ground truth
            ground_truth = email_data["label"]
            kg.store_ground_truth(email_id, ground_truth, "human_annotator")

            # Validate with LLM judge (every 10th email to avoid quota)
            if i % 10 == 0:
                validation = judge.validate_classification(
                    email_data["subject"],
                    email_data["body"],
                    predicted_topic,
                    ground_truth
                )
                stats["validation_scores"].append(validation["quality_score"])
                stats["validated"] += 1

            # Update statistics
            stats["processed"] += 1
            stats["stored"] += 1

            if predicted_topic == ground_truth:
                stats["correct_predictions"] += 1

            topic = email_data["label"]
            stats["topics"][topic] = stats["topics"].get(topic, 0) + 1

            # Progress indicator
            if i % 20 == 0:
                print(f"  Processed {i}/{len(emails)} emails...")

        except Exception as e:
            print(f"  ⚠️ Error processing email {i}: {str(e)}")

    print()
    print("=" * 70)
    print("📊 LOADING RESULTS")
    print("=" * 70)
    print(f"Total Processed: {stats['processed']}")
    print(f"Successfully Stored: {stats['stored']}")
    print(f"Validated with LLM Judge: {stats['validated']}")
    print(f"Prediction Accuracy: {stats['correct_predictions']}/{stats['processed']} ({100*stats['correct_predictions']/stats['processed']:.1f}%)")

    if stats["validation_scores"]:
        avg_quality = sum(stats["validation_scores"]) / len(stats["validation_scores"])
        print(f"Average Validation Quality: {avg_quality:.2f}")

    print()
    print("Topics Stored:")
    for topic, count in sorted(stats["topics"].items()):
        print(f"  {topic:15s}: {count:3d} emails")

    print()

    # Get Neo4j system overview
    overview = kg.get_mlops_system_overview()
    print("=" * 70)
    print("🔍 NEO4J SYSTEM STATUS")
    print("=" * 70)
    print(f"Total Emails in Graph: {overview.get('emails', {}).get('total', 0)}")
    print(f"Labeled Emails: {overview.get('emails', {}).get('labeled', 0)}")
    print(f"Models Tracked: {overview.get('models', {}).get('total', 0)}")
    print(f"Unique Topics: {len(overview.get('topics', []))}")

    print()
    print("✅ Data loading complete!")

    return stats


if __name__ == "__main__":
    import sys

    # Allow limiting the number of emails for testing
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    stats = load_expanded_data_to_neo4j(limit=limit)

    # Save statistics
    with open("data/neo4j_loading_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("Statistics saved to: data/neo4j_loading_stats.json")