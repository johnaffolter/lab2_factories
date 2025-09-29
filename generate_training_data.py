#!/usr/bin/env python3
"""
Generate Comprehensive Training Dataset for Email Classification
Creates diverse, realistic email examples with ground truth labels
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import random

class TrainingDataGenerator:
    """Generate realistic email training data across multiple categories"""

    def __init__(self):
        self.topics = {
            "work": {
                "keywords": ["meeting", "report", "project", "deadline", "review", "presentation", "team", "client"],
                "senders": ["manager@company.com", "colleague@company.com", "hr@company.com", "ceo@company.com"]
            },
            "personal": {
                "keywords": ["dinner", "weekend", "catch up", "family", "party", "movie", "friend", "vacation"],
                "senders": ["friend@gmail.com", "family@email.com", "buddy@yahoo.com", "bestie@outlook.com"]
            },
            "promotion": {
                "keywords": ["FREE", "WIN", "CLICK NOW", "LIMITED TIME", "OFFER", "DISCOUNT", "SAVE", "ACT NOW"],
                "senders": ["promo@deals.com", "offers@shopping.com", "sale@store.com", "marketing@retail.com"]
            },
            "newsletter": {
                "keywords": ["weekly digest", "newsletter", "latest news", "this week", "subscribe", "unsubscribe"],
                "senders": ["newsletter@techcrunch.com", "digest@nytimes.com", "weekly@medium.com", "updates@substack.com"]
            },
            "support": {
                "keywords": ["ticket", "issue", "problem", "help", "resolved", "support", "assistance", "inquiry"],
                "senders": ["support@service.com", "help@company.com", "tickets@support.com", "care@customer.com"]
            },
            "travel": {
                "keywords": ["flight", "booking", "reservation", "hotel", "itinerary", "confirmation", "trip", "check-in"],
                "senders": ["booking@airline.com", "reservations@hotel.com", "trips@expedia.com", "travel@agency.com"]
            },
            "education": {
                "keywords": ["course", "assignment", "grade", "lecture", "homework", "exam", "class", "professor"],
                "senders": ["professor@university.edu", "admin@college.edu", "registrar@school.edu", "instructor@academy.edu"]
            },
            "health": {
                "keywords": ["appointment", "doctor", "prescription", "medical", "health", "clinic", "checkup", "results"],
                "senders": ["office@clinic.com", "appointments@hospital.com", "care@health.com", "doctor@medical.com"]
            }
        }

        self.generated_emails = []

    def generate_email(self, topic: str) -> Dict[str, Any]:
        """Generate a single email for a specific topic"""
        topic_data = self.topics[topic]

        # Select keywords for this email (ensure we have at least 3)
        num_keywords = min(len(topic_data["keywords"]), random.randint(3, 5))
        selected_keywords = random.sample(topic_data["keywords"], k=num_keywords)
        sender = random.choice(topic_data["senders"])

        # Generate subject and body based on topic
        if topic == "work":
            subjects = [
                f"{selected_keywords[0].title()} scheduled for tomorrow",
                f"Q{random.randint(1,4)} {selected_keywords[0]} ready for {selected_keywords[1]}",
                f"Action required: {selected_keywords[0]} {selected_keywords[1]}",
                f"Follow-up on {selected_keywords[0]} discussion"
            ]
            bodies = [
                f"Hi team, the {selected_keywords[0]} is scheduled for tomorrow at {random.randint(9,17)}:00. Please review the {selected_keywords[1]} before we meet.",
                f"The {selected_keywords[0]} has been completed. Next steps include {selected_keywords[1]} and {selected_keywords[2]}. Let me know if you have questions.",
                f"Just a reminder about the upcoming {selected_keywords[0]}. We need to finalize the {selected_keywords[1]} by end of week.",
                f"Please see attached {selected_keywords[0]} for the {selected_keywords[1]}. Looking forward to your feedback."
            ]

        elif topic == "personal":
            subjects = [
                f"Want to {selected_keywords[0]} this {selected_keywords[1]}?",
                f"Catching up over {selected_keywords[0]}",
                f"Plans for {selected_keywords[0]}?",
                f"You free for {selected_keywords[0]}?"
            ]
            bodies = [
                f"Hey! Been a while since we caught up. Want to {selected_keywords[0]} this {selected_keywords[1]}? Let me know what works for you!",
                f"I found this great place for {selected_keywords[0]}. Should we check it out? Maybe this {selected_keywords[1]}?",
                f"Thinking about organizing a {selected_keywords[0]} with the {selected_keywords[1]}. Are you interested?",
                f"Miss you! Let's plan something soon. How about {selected_keywords[0]} next {selected_keywords[1]}?"
            ]

        elif topic == "promotion":
            k0, k1 = selected_keywords[0], selected_keywords[1]
            k2 = selected_keywords[2] if len(selected_keywords) > 2 else "SAVE"
            k3 = selected_keywords[3] if len(selected_keywords) > 3 else "ACT NOW"

            subjects = [
                f"{k0} {k1} - {random.randint(20,80)}% OFF!!!",
                f"Don't Miss: {k0} {k1} Inside!",
                f"EXCLUSIVE: {k0} {k1} Ends Soon!",
                f"You WON! {k0} - {k1} NOW!"
            ]
            bodies = [
                f"{k0} {k1}! This is your chance to {k2} big! {random.randint(20,80)}% off EVERYTHING! {k3}!!!",
                f"Congratulations! You've been selected for our {k0} {k1}! {k2} to claim your prize! Hurry!",
                f"AMAZING {k0}! For a {k1} only! {k2} on all products! Don't wait - {k3}!",
                f"Special {k0} just for you! {k1} and {k2} today! This is a {k3} opportunity!!!"
            ]

        elif topic == "newsletter":
            subjects = [
                f"This Week's {selected_keywords[0].title()}",
                f"{selected_keywords[0].title()}: Top Stories",
                f"Your Weekly {selected_keywords[0].title()}",
                f"{selected_keywords[0].title()} - {datetime.now().strftime('%B %d, %Y')}"
            ]
            bodies = [
                f"Welcome to this week's {selected_keywords[0]}! Here are the top stories: {selected_keywords[1]}, {selected_keywords[2]}, and more.",
                f"In this edition: Featured articles on {selected_keywords[1]}, upcoming events, and {selected_keywords[2]}. Read the full {selected_keywords[0]} here.",
                f"Your {selected_keywords[0]} is here! This week we're covering {selected_keywords[1]} and {selected_keywords[2]}. Enjoy!",
                f"Thanks for {selected_keywords[1]}! This week's highlights include {selected_keywords[2]}. Stay tuned for next week's {selected_keywords[0]}."
            ]

        elif topic == "support":
            subjects = [
                f"Re: {selected_keywords[0].title()} #{random.randint(1000,9999)}",
                f"Your {selected_keywords[0]} has been {selected_keywords[1]}",
                f"Update on {selected_keywords[0]} - Case #{random.randint(1000,9999)}",
                f"We're here to {selected_keywords[0]}"
            ]
            bodies = [
                f"Thank you for contacting support. Your {selected_keywords[0]} #{random.randint(1000,9999)} regarding {selected_keywords[1]} has been received and assigned to our team.",
                f"Good news! Your {selected_keywords[0]} has been {selected_keywords[1]}. We've addressed the {selected_keywords[2]} you reported. Please let us know if you need further {selected_keywords[3]}.",
                f"We're currently investigating your {selected_keywords[0]}. Our team is working on the {selected_keywords[1]} and will provide an update within 24 hours.",
                f"This is a follow-up regarding your recent {selected_keywords[0]}. We want to ensure the {selected_keywords[1]} has been fully {selected_keywords[2]}."
            ]

        elif topic == "travel":
            subjects = [
                f"Your {selected_keywords[0]} {selected_keywords[1]} - Confirmation #{random.randint(100000,999999)}",
                f"{selected_keywords[0]} {selected_keywords[1]} Confirmed",
                f"Upcoming {selected_keywords[0]} - {selected_keywords[1]} Details",
                f"E-{selected_keywords[0]} for your {selected_keywords[1]}"
            ]
            bodies = [
                f"Your {selected_keywords[0]} {selected_keywords[1]} has been confirmed! Confirmation number: {random.randint(100000,999999)}. Your {selected_keywords[2]} is scheduled for {random.choice(['tomorrow', 'next week', 'next month'])}.",
                f"Thank you for booking with us. Your {selected_keywords[0]} {selected_keywords[1]} details: Check-in at {random.randint(8,18)}:00. Please review your {selected_keywords[2]}.",
                f"Your {selected_keywords[0]} is all set! {selected_keywords[1]} confirmation and {selected_keywords[2]} are attached. Don't forget to {selected_keywords[3]} 24 hours before departure.",
                f"Reminder: Your upcoming {selected_keywords[0]}. {selected_keywords[1]} details and {selected_keywords[2]} information attached. Have a great {selected_keywords[3]}!"
            ]

        elif topic == "education":
            subjects = [
                f"{selected_keywords[0].title()} {selected_keywords[1]} - Due {random.choice(['Monday', 'Friday', 'next week'])}",
                f"Update: {selected_keywords[0]} {selected_keywords[1]}",
                f"Reminder: {selected_keywords[0]} this {random.choice(['Monday', 'Wednesday', 'Friday'])}",
                f"Your {selected_keywords[0]} {selected_keywords[1]} Results"
            ]
            bodies = [
                f"This is a reminder about your upcoming {selected_keywords[0]} {selected_keywords[1]} due {random.choice(['Monday', 'Friday', 'next week'])}. Please review the {selected_keywords[2]} materials.",
                f"The {selected_keywords[0]} has been posted. Your {selected_keywords[1]} is available in the portal. Please check the {selected_keywords[2]} for next week's {selected_keywords[3]}.",
                f"Don't forget about the {selected_keywords[0]} scheduled for this {random.choice(['Monday', 'Wednesday', 'Friday'])}. Topics include {selected_keywords[1]} and {selected_keywords[2]}.",
                f"Your {selected_keywords[0]} {selected_keywords[1]} has been graded. Results and feedback are available. Please see me during office hours if you have questions about the {selected_keywords[2]}."
            ]

        elif topic == "health":
            subjects = [
                f"{selected_keywords[0].title()} Reminder - {random.choice(['Tomorrow', 'Next Week', 'This Friday'])}",
                f"Your {selected_keywords[0]} {selected_keywords[1]} is Ready",
                f"Confirmation: {selected_keywords[0]} Scheduled",
                f"{selected_keywords[0].title()} {selected_keywords[1]} - Important Information"
            ]
            bodies = [
                f"This is a reminder of your upcoming {selected_keywords[0]} on {random.choice(['tomorrow', 'Friday', 'next Tuesday'])} at {random.randint(8,17)}:00. Please arrive 15 minutes early for {selected_keywords[1]}.",
                f"Your {selected_keywords[0]} {selected_keywords[1]} are now available. Please contact our {selected_keywords[2]} to discuss. Call {random.randint(100,999)}-{random.randint(1000,9999)}.",
                f"Your {selected_keywords[0]} has been scheduled for {random.choice(['next week', 'next month'])}. Please bring your insurance card and arrive early for {selected_keywords[1]}.",
                f"Following your recent {selected_keywords[0]}, your {selected_keywords[1]} has been sent to the pharmacy. Please contact us if you need to schedule a follow-up {selected_keywords[2]}."
            ]

        else:
            subjects = [f"Email about {selected_keywords[0]}"]
            bodies = [f"This is an email regarding {selected_keywords[0]} and {selected_keywords[1]}."]

        subject = random.choice(subjects)
        body = random.choice(bodies)

        email = {
            "id": f"email_{len(self.generated_emails) + 1:04d}",
            "subject": subject,
            "body": body,
            "sender": sender,
            "label": topic,
            "timestamp": datetime.now().isoformat()
        }

        return email

    def generate_dataset(self, emails_per_topic: int = 20) -> List[Dict[str, Any]]:
        """Generate complete training dataset"""
        print(f"🎯 Generating Training Dataset")
        print(f"   Topics: {len(self.topics)}")
        print(f"   Emails per topic: {emails_per_topic}")
        print(f"   Total emails: {len(self.topics) * emails_per_topic}")
        print()

        for topic in self.topics.keys():
            print(f"📧 Generating {emails_per_topic} emails for topic: {topic}")
            for i in range(emails_per_topic):
                email = self.generate_email(topic)
                self.generated_emails.append(email)
            print(f"   ✅ Generated {len([e for e in self.generated_emails if e['label'] == topic])} {topic} emails")

        print(f"\n✅ Total emails generated: {len(self.generated_emails)}")
        return self.generated_emails

    def save_dataset(self, filename: str = "data/training_emails.json"):
        """Save dataset to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.generated_emails, f, indent=2)

        print(f"💾 Saved dataset to: {filename}")
        print(f"   File size: {os.path.getsize(filename)} bytes")

        # Also save summary statistics
        stats = {
            "total_emails": len(self.generated_emails),
            "topics": list(self.topics.keys()),
            "distribution": {
                topic: len([e for e in self.generated_emails if e['label'] == topic])
                for topic in self.topics.keys()
            },
            "generated_at": datetime.now().isoformat()
        }

        stats_file = filename.replace('.json', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 Saved statistics to: {stats_file}")
        return filename

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.generated_emails:
            return {"error": "No emails generated yet"}

        return {
            "total_emails": len(self.generated_emails),
            "topics": list(self.topics.keys()),
            "distribution": {
                topic: len([e for e in self.generated_emails if e['label'] == topic])
                for topic in self.topics.keys()
            },
            "avg_subject_length": sum(len(e['subject']) for e in self.generated_emails) / len(self.generated_emails),
            "avg_body_length": sum(len(e['body']) for e in self.generated_emails) / len(self.generated_emails)
        }


def main():
    """Generate and save training dataset"""
    print("="*60)
    print("EMAIL TRAINING DATA GENERATOR")
    print("="*60)
    print()

    # Create generator
    generator = TrainingDataGenerator()

    # Generate dataset (20 emails per topic = 160 total)
    emails = generator.generate_dataset(emails_per_topic=20)

    # Save to file
    filename = generator.save_dataset()

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    stats = generator.get_statistics()
    print(f"Total Emails: {stats['total_emails']}")
    print(f"Topics: {', '.join(stats['topics'])}")
    print(f"\nDistribution:")
    for topic, count in stats['distribution'].items():
        print(f"  {topic:12s}: {count:3d} emails ({count/stats['total_emails']*100:.1f}%)")
    print(f"\nAverage Subject Length: {stats['avg_subject_length']:.1f} characters")
    print(f"Average Body Length: {stats['avg_body_length']:.1f} characters")
    print("="*60)

    return filename, emails


if __name__ == "__main__":
    main()