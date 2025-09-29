from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from app.features.factory import FeatureGeneratorFactory

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str
    use_email_similarity: Optional[bool] = False

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class TopicRequest(BaseModel):
    topic_name: str
    description: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None

@router.post("/emails/classify", response_model=EmailClassificationResponse,
             summary="Classify Email",
             description="Classify an email into predefined topics using machine learning")
async def classify_email(request: EmailRequest):
    """
    Classify an email into one of the available topics.

    - **subject**: Email subject line
    - **body**: Email body content
    - **use_email_similarity**: Use email similarity mode instead of topic similarity

    Returns classification result with confidence scores and extracted features.
    """
    try:
        inference_service = EmailTopicInferenceService(use_email_similarity=request.use_email_similarity)
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email, use_email_similarity=request.use_email_similarity)

        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics", summary="Get Available Topics",
            description="Retrieve all available email classification topics",
            response_description="List of topic names")
async def topics():
    """
    Get all available email classification topics.

    Returns a list of topics that emails can be classified into,
    including any dynamically added topics.
    """
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

@router.get("/features")
async def get_features():
    """Get information about all available feature generators"""
    try:
        generators_info = FeatureGeneratorFactory.get_available_generators()
        return {
            "available_generators": generators_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/topics")
async def add_topic(request: TopicRequest):
    """Dynamically add a new topic to the topics file"""
    try:
        # Load existing topics
        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        topics_file = os.path.join(data_dir, 'data', 'topic_keywords.json')

        with open(topics_file, 'r') as f:
            topics = json.load(f)

        # Check if topic already exists
        if request.topic_name in topics:
            raise HTTPException(status_code=400, detail=f"Topic '{request.topic_name}' already exists")

        # Add new topic
        topics[request.topic_name] = {
            "description": request.description
        }

        # Save updated topics
        with open(topics_file, 'w') as f:
            json.dump(topics, f, indent=2)

        return {"message": f"Topic '{request.topic_name}' added successfully", "topics": list(topics.keys())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails")
async def store_email(request: EmailStoreRequest):
    """Store an email with optional ground truth for training"""
    try:
        # Load existing emails
        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        emails_file = os.path.join(data_dir, 'data', 'emails.json')

        with open(emails_file, 'r') as f:
            emails = json.load(f)

        # Create email record
        email_record = {
            "id": len(emails) + 1,
            "subject": request.subject,
            "body": request.body
        }

        if request.ground_truth:
            email_record["ground_truth"] = request.ground_truth

        # Add to emails list
        emails.append(email_record)

        # Save updated emails
        with open(emails_file, 'w') as f:
            json.dump(emails, f, indent=2)

        return {"message": "Email stored successfully", "email_id": email_record["id"], "total_emails": len(emails)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emails")
async def get_stored_emails():
    """Get all stored emails"""
    try:
        data_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        emails_file = os.path.join(data_dir, 'data', 'emails.json')

        with open(emails_file, 'r') as f:
            emails = json.load(f)

        return {"emails": emails, "count": len(emails)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

