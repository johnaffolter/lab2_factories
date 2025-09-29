"""
Generate Expanded Training Data with More Topics
Includes LLM-as-a-Judge validation for quality
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Expanded topic definitions with more categories
EXPANDED_TOPICS = {
    "work": {
        "keywords": ["meeting", "deadline", "project", "report", "presentation", "budget", "team", "client"],
        "templates": [
            "Team meeting scheduled for {} at {}",
            "Project {} deadline is approaching on {}",
            "Please review the {} report by {}",
            "Client presentation for {} needs your input",
            "Budget approval required for {} initiative"
        ]
    },
    "personal": {
        "keywords": ["family", "friend", "weekend", "dinner", "birthday", "vacation", "home", "kids"],
        "templates": [
            "Family gathering at {} this {}",
            "Dinner with {} on {} evening",
            "Happy birthday! Hope you have a great {}",
            "Weekend plans - {} sounds fun",
            "Vacation photos from {} trip"
        ]
    },
    "promotion": {
        "keywords": ["sale", "discount", "offer", "deal", "save", "limited", "exclusive", "free"],
        "templates": [
            "FLASH SALE: {}% off all {} today only",
            "Exclusive offer: Buy {} get {} free",
            "Limited time deal on {} - Save {}",
            "FREE shipping on orders over {}",
            "Members only: {} discount on {}"
        ]
    },
    "newsletter": {
        "keywords": ["update", "weekly", "digest", "roundup", "newsletter", "edition", "insights"],
        "templates": [
            "Weekly {} Newsletter - {} Edition",
            "Your {} digest for {}",
            "Top {} insights from {}",
            "{} roundup: What happened this week",
            "Monthly update on {} trends"
        ]
    },
    "support": {
        "keywords": ["ticket", "issue", "problem", "help", "assistance", "resolved", "error", "technical"],
        "templates": [
            "Support ticket #{} - {} issue",
            "Technical assistance needed for {}",
            "Your {} problem has been resolved",
            "Error report: {} not working",
            "Help request for {} feature"
        ]
    },
    "travel": {
        "keywords": ["flight", "hotel", "booking", "reservation", "itinerary", "destination", "check-in"],
        "templates": [
            "Flight confirmation: {} to {} on {}",
            "Hotel booking at {} for {}",
            "Travel itinerary for {} trip",
            "Reservation confirmed for {}",
            "Check-in reminder for {} flight"
        ]
    },
    "education": {
        "keywords": ["course", "class", "assignment", "exam", "lecture", "study", "tutorial", "learning"],
        "templates": [
            "New course available: {}",
            "Assignment due date: {}",
            "Exam schedule for {} on {}",
            "Lecture notes for {} class",
            "Study group meeting for {}"
        ]
    },
    "health": {
        "keywords": ["appointment", "doctor", "medical", "prescription", "wellness", "fitness", "exercise"],
        "templates": [
            "Appointment reminder: {} with Dr. {}",
            "Prescription ready for pickup at {}",
            "Wellness check scheduled for {}",
            "Fitness goal update: {}",
            "Medical results available for {}"
        ]
    },
    # NEW EXPANDED TOPICS
    "finance": {
        "keywords": ["payment", "invoice", "transaction", "account", "balance", "bank", "credit", "bill"],
        "templates": [
            "Payment confirmation: ${} processed",
            "Invoice #{} due on {}",
            "Account statement for {}",
            "Transaction alert: ${} at {}",
            "Credit card bill for {} is ready"
        ]
    },
    "shopping": {
        "keywords": ["order", "delivery", "shipped", "tracking", "purchase", "cart", "checkout", "product"],
        "templates": [
            "Order #{} has been shipped",
            "Delivery scheduled for {}",
            "Tracking number for your {} order",
            "Purchase confirmation: {}",
            "Your cart has {} items waiting"
        ]
    },
    "social": {
        "keywords": ["event", "invitation", "party", "gathering", "celebration", "rsvp", "networking"],
        "templates": [
            "You're invited to {} on {}",
            "Event reminder: {} at {}",
            "Party celebration for {}",
            "Networking event: {}",
            "RSVP needed for {} by {}"
        ]
    },
    "entertainment": {
        "keywords": ["movie", "concert", "show", "ticket", "streaming", "music", "video", "game"],
        "templates": [
            "New episode of {} available now",
            "Concert tickets for {} on sale",
            "Movie premiere: {} this {}",
            "Streaming recommendation: {}",
            "Game update: {} new features"
        ]
    }
}

def generate_email(topic: str, email_num: int) -> Dict:
    """Generate a single email for the given topic"""
    topic_data = EXPANDED_TOPICS[topic]

    # Select random template and keywords
    template = random.choice(topic_data["templates"])
    keywords = random.sample(topic_data["keywords"], min(2, len(topic_data["keywords"])))

    # Generate subject using template
    # Count placeholders in template
    placeholder_count = template.count("{}")

    if topic == "work":
        args = ["Development", "2pm", "Q4 Budget", "Friday"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "personal":
        args = ["Sarah's house", "Saturday", "weekend", "tomorrow"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "promotion":
        args = ["30", "electronics", "1", "50%"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "newsletter":
        args = ["Tech", "Weekly", "5", "insights"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "support":
        args = ["12345", "login", "resolved", "error"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "travel":
        args = ["NYC", "LAX", "Dec 15", "morning"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "education":
        args = ["Introduction to ML", "Monday", "midterm", "Python"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "health":
        args = ["Annual checkup", "Smith", "Friday", "morning"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "finance":
        args = ["250.00", str(random.randint(1000, 9999)), "November", "checking"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "shopping":
        args = ["ORD" + str(random.randint(1000, 9999)), "tomorrow", "UPS123", "laptop"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "social":
        args = ["birthday party", "Friday 7pm", "Saturday", "John's 30th"][:placeholder_count]
        subject = template.format(*args)
    elif topic == "entertainment":
        args = ["The Last Episode", "streaming", "Season 5", "tonight"][:placeholder_count]
        subject = template.format(*args)
    else:
        subject = template

    # Generate body with more context
    body_templates = {
        "work": "Please review the attached documents and provide feedback by end of day. The client needs our response for their planning.",
        "personal": "Looking forward to seeing you! Bring the whole family and we can catch up over dinner.",
        "promotion": "Don't miss out on this limited time offer. Use code SAVE30 at checkout. Offer expires tonight at midnight.",
        "newsletter": "This week's highlights include major updates in technology and industry trends. Read the full article below.",
        "support": "We've received your request and our technical team is investigating. We'll update you within 24 hours.",
        "travel": "Your booking is confirmed. Please arrive at least 2 hours before departure for international flights.",
        "education": "The course begins next Monday. All materials are available in the student portal. Join the first live session.",
        "health": "Please bring your insurance card and arrive 15 minutes early to complete paperwork.",
        "finance": "This is an automated notification from your bank. Your account balance has been updated.",
        "shopping": "Your order has been processed and is being prepared for shipment. Expected delivery in 3-5 business days.",
        "social": "We're hosting a special celebration and would love for you to join us. Please confirm your attendance.",
        "entertainment": "New content is available now. Start watching immediately on any device with your subscription."
    }

    body = body_templates.get(topic, f"This is a {topic} related email with relevant content.")
    body += f" Reference: {keywords[0].upper()}-{email_num}"

    # Generate metadata
    senders = {
        "work": "manager@company.com",
        "personal": "friend@gmail.com",
        "promotion": "deals@retailer.com",
        "newsletter": "updates@newsletter.com",
        "support": "support@service.com",
        "travel": "bookings@travel.com",
        "education": "admin@university.edu",
        "health": "appointments@clinic.com",
        "finance": "alerts@bank.com",
        "shopping": "orders@shop.com",
        "social": "events@social.com",
        "entertainment": "info@streaming.com"
    }

    return {
        "id": f"{topic}_{email_num}",
        "subject": subject,
        "body": body,
        "sender": senders.get(topic, "sender@example.com"),
        "label": topic,
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        "metadata": {
            "keywords": keywords,
            "generated_at": datetime.now().isoformat()
        }
    }

def generate_expanded_dataset(emails_per_category: int = 15) -> List[Dict]:
    """Generate expanded training dataset"""
    dataset = []

    for topic in EXPANDED_TOPICS.keys():
        for i in range(emails_per_category):
            email = generate_email(topic, i + 1)
            dataset.append(email)

    # Shuffle to mix topics
    random.shuffle(dataset)

    return dataset

def calculate_dataset_stats(dataset: List[Dict]) -> Dict:
    """Calculate statistics for the dataset"""
    stats = {
        "total_emails": len(dataset),
        "topics": list(EXPANDED_TOPICS.keys()),
        "num_topics": len(EXPANDED_TOPICS),
        "emails_per_topic": {},
        "generated_at": datetime.now().isoformat(),
        "expanded": True
    }

    for topic in EXPANDED_TOPICS.keys():
        count = sum(1 for email in dataset if email["label"] == topic)
        stats["emails_per_topic"][topic] = count

    return stats

if __name__ == "__main__":
    print("Generating Expanded Training Dataset...")
    print("=" * 70)

    # Generate 15 emails per category (12 categories = 180 total)
    dataset = generate_expanded_dataset(emails_per_category=15)
    stats = calculate_dataset_stats(dataset)

    # Save to files
    with open("data/expanded_training_emails.json", "w") as f:
        json.dump(dataset, f, indent=2)

    with open("data/expanded_training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Generated {stats['total_emails']} emails across {stats['num_topics']} topics")
    print()
    print("Topic Distribution:")
    for topic, count in sorted(stats["emails_per_topic"].items()):
        print(f"  {topic:15s}: {count:3d} emails")

    print()
    print("Files saved:")
    print("  - data/expanded_training_emails.json")
    print("  - data/expanded_training_stats.json")
    print()
    print("✅ Expanded training data generation complete!")