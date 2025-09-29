import requests
import json

# Test classification with special characters
test_email = {
    "subject": "URGENT!!! Meeting @ 3pm",
    "body": "Review Q3 report - bring $$$ projections & charts!",
    "use_email_similarity": False
}

response = requests.post(
    "http://localhost:8000/emails/classify",
    json=test_email
)

result = response.json()

print("Email Classification Test with Special Characters")
print("="*50)
print(f"Subject: {test_email['subject']}")
print(f"Body: {test_email['body']}")
print(f"\nClassification Result:")
print(f"  Predicted Topic: {result['predicted_topic']}")
print(f"  Confidence: {result['topic_scores'][result['predicted_topic']]:.2%}")
print(f"\nFeature Extraction:")
print(f"  Spam words detected: {result['features']['spam_has_spam_words']}")
print(f"  Average word length: {result['features']['word_length_average_word_length']:.2f}")
print(f"  Non-text characters: {result['features']['non_text_non_text_char_count']}")
print(f"  Email embedding: {result['features']['email_embeddings_average_embedding']}")
print(f"\nSpecial characters found: {result['features']['non_text_non_text_char_count']} ✅")