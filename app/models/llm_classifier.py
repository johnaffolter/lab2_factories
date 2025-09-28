"""
Optional LLM-based email classifier - COMMENTED OUT BY DEFAULT

To enable this classifier:
1. Uncomment the code below
2. Install required dependencies: pip install openai anthropic
3. Set your API key as an environment variable:
   - For OpenAI: export OPENAI_API_KEY="your-key-here"
   - For Anthropic: export ANTHROPIC_API_KEY="your-key-here"
4. Import and use in your classification pipeline
"""

# import os
# import json
# from typing import Dict, Any, List, Optional
# import logging
#
# # Uncomment one of these based on your preference:
# # from openai import OpenAI  # For GPT models
# # from anthropic import Anthropic  # For Claude models
#
# logger = logging.getLogger(__name__)
#
# class LLMEmailClassifier:
#     """LLM-based email classifier using OpenAI GPT or Anthropic Claude"""
#
#     def __init__(self, provider: str = "openai", model: str = None):
#         """
#         Initialize the LLM classifier
#         Args:
#             provider: "openai" or "anthropic"
#             model: Model name (e.g., "gpt-4", "claude-3-opus")
#         """
#         self.provider = provider
#         self.topic_data = self._load_topic_data()
#         self.topics = list(self.topic_data.keys())
#
#         if provider == "openai":
#             api_key = os.getenv("OPENAI_API_KEY")
#             if not api_key:
#                 raise ValueError("OPENAI_API_KEY environment variable not set")
#             self.client = OpenAI(api_key=api_key)
#             self.model = model or "gpt-4o-mini"
#         elif provider == "anthropic":
#             api_key = os.getenv("ANTHROPIC_API_KEY")
#             if not api_key:
#                 raise ValueError("ANTHROPIC_API_KEY environment variable not set")
#             self.client = Anthropic(api_key=api_key)
#             self.model = model or "claude-3-haiku-20240307"
#         else:
#             raise ValueError(f"Unsupported provider: {provider}")
#
#     def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
#         """Load topic data from data/topic_keywords.json"""
#         data_file = os.path.join(
#             os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
#             'data', 'topic_keywords.json'
#         )
#         with open(data_file, 'r') as f:
#             return json.load(f)
#
#     def classify_email(self, subject: str, body: str) -> Dict[str, Any]:
#         """
#         Classify an email using LLM
#         Returns dict with predicted_topic and confidence
#         """
#         # Create topic descriptions
#         topic_descriptions = "\n".join([
#             f"- {topic}: {data['description']}"
#             for topic, data in self.topic_data.items()
#         ])
#
#         # Build the prompt
#         prompt = f"""Classify the following email into one of these categories:
#
# {topic_descriptions}
#
# Email Subject: {subject}
# Email Body: {body}
#
# Respond with ONLY the category name (one word) that best matches this email.
# Category:"""
#
#         try:
#             if self.provider == "openai":
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": "You are an email classification assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1,
#                     max_tokens=10
#                 )
#                 predicted_topic = response.choices[0].message.content.strip().lower()
#
#             elif self.provider == "anthropic":
#                 response = self.client.messages.create(
#                     model=self.model,
#                     max_tokens=10,
#                     temperature=0.1,
#                     system="You are an email classification assistant.",
#                     messages=[
#                         {"role": "user", "content": prompt}
#                     ]
#                 )
#                 predicted_topic = response.content[0].text.strip().lower()
#
#             # Validate the response
#             if predicted_topic not in self.topics:
#                 logger.warning(f"LLM returned invalid topic: {predicted_topic}")
#                 # Fall back to similarity-based matching
#                 predicted_topic = self._find_closest_topic(predicted_topic)
#
#             return {
#                 "predicted_topic": predicted_topic,
#                 "classification_method": f"llm_{self.provider}",
#                 "model": self.model
#             }
#
#         except Exception as e:
#             logger.error(f"LLM classification failed: {e}")
#             raise
#
#     def _find_closest_topic(self, text: str) -> str:
#         """Find the closest matching topic if LLM returns invalid topic"""
#         text_lower = text.lower()
#         for topic in self.topics:
#             if topic in text_lower or text_lower in topic:
#                 return topic
#         # Default to first topic if no match
#         return self.topics[0] if self.topics else "unknown"
#
#
# # Example integration into existing EmailClassifierModel:
# """
# # In app/models/similarity_model.py, add:
#
# def __init__(self, use_email_similarity: bool = False, use_llm: bool = False):
#     self.use_email_similarity = use_email_similarity
#     self.use_llm = use_llm
#     self.topic_data = self._load_topic_data()
#     self.topics = list(self.topic_data.keys())
#     self.stored_emails = self._load_stored_emails() if use_email_similarity else []
#
#     # Initialize LLM classifier if enabled
#     if use_llm:
#         try:
#             from app.models.llm_classifier import LLMEmailClassifier
#             self.llm_classifier = LLMEmailClassifier()
#         except Exception as e:
#             logger.warning(f"Failed to initialize LLM classifier: {e}")
#             self.use_llm = False
#
# def predict(self, features: Dict[str, Any]) -> str:
#     if self.use_llm and hasattr(self, 'llm_classifier'):
#         # Extract email text from features
#         subject = features.get("raw_email_email_subject", "")
#         body = features.get("raw_email_email_body", "")
#         result = self.llm_classifier.classify_email(subject, body)
#         return result["predicted_topic"]
#     elif self.use_email_similarity and self.stored_emails:
#         return self._predict_by_email_similarity(features)
#     else:
#         return self._predict_by_topic_similarity(features)
# """
#
# # Example API endpoint update:
# """
# # In app/api/routes.py, add to EmailRequest model:
#
# class EmailRequest(BaseModel):
#     subject: str
#     body: str
#     use_email_similarity: Optional[bool] = False
#     use_llm: Optional[bool] = False  # Add this field
#
# # Update the classify endpoint to pass use_llm parameter
# """