#!/usr/bin/env python3
"""
Test Document Graph Integration
Demonstrates storing screenshots and various document types in Neo4j
"""

import json
import requests
from datetime import datetime
from pathlib import Path
import os
from app.services.document_graph_service import DocumentGraphService, DocumentType

def test_document_storage():
    """Test storing various document types in the graph"""

    print("="*80)
    print("DOCUMENT GRAPH INTEGRATION TEST")
    print("="*80)

    # Initialize service
    service = DocumentGraphService()

    # Load screenshot metadata
    screenshot_metadata_path = Path("screenshots/capture_metadata.json")
    if screenshot_metadata_path.exists():
        with open(screenshot_metadata_path, 'r') as f:
            metadata = json.load(f)

        print("\n📸 STORING SCREENSHOTS IN GRAPH...")

        # Store each screenshot as a document
        for screenshot in metadata.get("screenshots", []):
            doc_data = {
                "type": DocumentType.SCREENSHOT.value,
                "title": screenshot["name"],
                "content": screenshot["description"],
                "file_path": screenshot["file"],
                "topics": ["ui", "testing", "mlops"],
                "project": "email_classification",
                "author": "test_system",
                "metadata": {
                    "timestamp": screenshot["timestamp"],
                    "format": "png"
                }
            }

            doc_id = service.store_document(doc_data)
            print(f"✓ Stored screenshot: {screenshot['name']} (ID: {doc_id[:8]}...)")

    # Test storing various document types
    print("\n📄 STORING VARIOUS DOCUMENT TYPES...")

    test_documents = [
        {
            "type": DocumentType.EMAIL.value,
            "title": "Q3 Budget Review Request",
            "content": "Please review the attached Q3 budget report before our meeting.",
            "topics": ["finance", "work", "urgent"],
            "project": "email_classification",
            "author": "john_doe"
        },
        {
            "type": DocumentType.REPORT.value,
            "title": "Email Classification Performance Report",
            "content": "System achieved 92% accuracy with 45ms average response time.",
            "topics": ["mlops", "performance", "analytics"],
            "project": "email_classification",
            "author": "system"
        },
        {
            "type": DocumentType.CODE.value,
            "title": "feature_factory.py",
            "content": "class FeatureGeneratorFactory: # Factory pattern implementation",
            "topics": ["python", "design_patterns", "mlops"],
            "project": "email_classification",
            "author": "developer"
        },
        {
            "type": DocumentType.DOCUMENTATION.value,
            "title": "API Documentation",
            "content": "REST API endpoints for email classification system",
            "topics": ["api", "documentation", "rest"],
            "project": "email_classification",
            "author": "tech_writer"
        },
        {
            "type": DocumentType.MEETING_NOTES.value,
            "title": "MLOps Team Standup",
            "content": "Discussed homework progress and graph database integration",
            "topics": ["meetings", "mlops", "progress"],
            "project": "email_classification",
            "author": "scrum_master"
        }
    ]

    for doc in test_documents:
        doc_id = service.store_document(doc)
        print(f"✓ Stored {doc['type']}: {doc['title']} (ID: {doc_id[:8]}...)")

    # Test classification and storage
    print("\n🤖 TESTING CLASSIFICATION STORAGE...")

    # Classify an email
    email_doc = {
        "type": DocumentType.EMAIL.value,
        "title": "Investment Portfolio Update",
        "content": "Your stocks have gained 8% this quarter.",
        "topics": ["finance"],
        "project": "email_classification",
        "author": "broker"
    }

    # Simulate classification result
    classification_result = {
        "predicted_topic": "finance",
        "confidence": 0.92,
        "algorithm": "cosine_similarity",
        "topic_scores": {
            "finance": 0.92,
            "work": 0.75,
            "personal": 0.60
        },
        "features": {
            "spam_has_spam_words": 0,
            "word_length_average_word_length": 5.8,
            "non_text_char_count": 2,
            "email_embeddings_average_embedding": 42.5
        }
    }

    class_id = service.classify_and_store(email_doc, classification_result)
    print(f"✓ Stored classification: {class_id} for topic 'finance' (92% confidence)")

    # Get graph visualization data
    print("\n📊 RETRIEVING GRAPH DATA...")

    graph_data = service.get_document_graph(limit=20)
    print(f"✓ Retrieved {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")

    # Node type breakdown
    node_types = {}
    for node in graph_data['nodes']:
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print("\nNode Distribution:")
    for ntype, count in node_types.items():
        print(f"  • {ntype}: {count}")

    # Search for similar documents
    print("\n🔍 SEARCHING FOR SIMILAR DOCUMENTS...")

    search_doc = {
        "type": DocumentType.EMAIL.value,
        "topics": ["finance", "work"]
    }

    similar_docs = service.search_similar_documents(search_doc, limit=3)
    print(f"✓ Found {len(similar_docs)} similar documents:")
    for doc in similar_docs:
        print(f"  • {doc.get('title', 'Untitled')} ({doc.get('shared_topics', 0)} shared topics)")

    # Get insights
    print("\n📈 DOCUMENT COLLECTION INSIGHTS...")

    insights = service.get_document_insights()
    if insights['total_documents'] > 0:
        print(f"  • Total Documents: {insights['total_documents']}")
        print(f"  • Document Types: {', '.join(insights['document_types'])}")
        print(f"  • Projects: {insights['project_count']}")
        print(f"  • Avg Confidence: {insights['average_confidence']:.2%}")
        print(f"  • Classifications: {insights['total_classifications']}")

        if insights['top_topics']:
            print("\n  Top Topics:")
            for topic in insights['top_topics'][:5]:
                print(f"    • {topic['topic']}: {topic['count']} documents")
    else:
        print("  ℹ️ Using local storage (Neo4j not connected)")

    print("\n" + "="*80)
    print("✅ DOCUMENT GRAPH INTEGRATION TEST COMPLETE")
    print("="*80)

    # Close connection
    service.close()

    return graph_data

if __name__ == "__main__":
    # Test the integration
    graph_data = test_document_storage()

    # Save graph data for visualization
    with open("data/graph_export.json", "w") as f:
        json.dump(graph_data, f, indent=2)

    print("\n📁 Graph data exported to: data/graph_export.json")
    print("🌐 Open frontend/graph_visualization.html to view the interactive graph")
    print("\n✨ System Capabilities Demonstrated:")
    print("  1. Multiple document type support (10 types)")
    print("  2. Screenshot storage and classification")
    print("  3. Graph-based similarity search")
    print("  4. Rich metadata and relationships")
    print("  5. Neo4j integration with fallback")