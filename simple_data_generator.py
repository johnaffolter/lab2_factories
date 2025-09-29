#!/usr/bin/env python3
"""
Simple Training Data Generator - Robust email generation for classification
"""

import json
import os
from datetime import datetime
from typing import List, Dict
import random

# Predefined email templates by category
EMAIL_TEMPLATES = {
    "work": [
        {
            "subject": "Q{q} Financial Report Ready for Review",
            "body": "The quarterly financial report is ready. Please review the revenue figures and provide feedback by Friday.",
            "sender": "finance@company.com"
        },
        {
            "subject": "Team Meeting Rescheduled to {time}",
            "body": "Hi team, the weekly standup has been moved to {time} today in conference room B. Please update your calendars.",
            "sender": "manager@company.com"
        },
        {
            "subject": "Project Deadline Update - Action Required",
            "body": "The project deadline has been moved up. We need to complete the deliverables by end of week. Please prioritize this work.",
            "sender": "pm@company.com"
        },
        {
            "subject": "Client Presentation Tomorrow",
            "body": "Reminder: client presentation is tomorrow at 2pm. Please review the slides and be prepared to discuss the proposal.",
            "sender": "sales@company.com"
        },
    ],
    "personal": [
        {
            "subject": "Dinner plans this weekend?",
            "body": "Hey! Want to grab dinner Saturday? I found a great new Italian place downtown. Let me know!",
            "sender": "friend@gmail.com"
        },
        {
            "subject": "Movie night Friday?",
            "body": "Thinking about having a movie night this Friday. Are you free? We could watch that new sci-fi film.",
            "sender": "buddy@yahoo.com"
        },
        {
            "subject": "Catching up soon",
            "body": "Been too long! Want to catch up over coffee this week? I'm free Tuesday or Thursday afternoon.",
            "sender": "friend@outlook.com"
        },
        {
            "subject": "Weekend getaway idea",
            "body": "Found this cool cabin in the mountains for rent. Want to do a weekend trip next month with the group?",
            "sender": "friend@email.com"
        },
    ],
    "promotion": [
        {
            "subject": "WIN FREE iPHONE - Click NOW!!!",
            "body": "You've been selected to win a FREE iPhone 15! Click here now to claim your prize before it expires! ACT NOW!!!",
            "sender": "promo@deals.com"
        },
        {
            "subject": "EXCLUSIVE OFFER: {percent}% OFF Everything!!!",
            "body": "LIMITED TIME ONLY! Get {percent}% OFF on ALL products! This amazing offer won't last! CLICK NOW to SAVE BIG!!!",
            "sender": "sale@shopping.com"
        },
        {
            "subject": "You WON $1000 Cash Prize!!!",
            "body": "CONGRATULATIONS! You've won $1000 CASH! No purchase necessary! Click to claim your winnings NOW! Don't miss this!",
            "sender": "winner@prize.com"
        },
        {
            "subject": "URGENT: Claim Your Reward NOW!!!",
            "body": "Your exclusive reward is waiting! CLICK HERE immediately to claim! Offer expires in 24 hours! ACT FAST!!!",
            "sender": "rewards@offer.com"
        },
    ],
    "newsletter": [
        {
            "subject": "This Week in Tech - Your Weekly Digest",
            "body": "Welcome to this week's tech newsletter! Top stories: AI breakthroughs, startup funding, and new gadget releases. Read more inside.",
            "sender": "newsletter@techcrunch.com"
        },
        {
            "subject": "Monthly Newsletter - {month} Edition",
            "body": "Your monthly update is here! This month's highlights include industry news, upcoming events, and featured articles. Enjoy!",
            "sender": "digest@nytimes.com"
        },
        {
            "subject": "Weekly Update from Medium",
            "body": "Here are this week's top stories from writers you follow. Topics include technology, design, and business insights.",
            "sender": "weekly@medium.com"
        },
        {
            "subject": "Your Substack Digest",
            "body": "New posts from your subscriptions: Economics Weekly, Tech Trends, and Creative Writing. Check out what you missed!",
            "sender": "updates@substack.com"
        },
    ],
    "support": [
        {
            "subject": "Re: Support Ticket #{ticket} - Issue Resolved",
            "body": "Good news! Your support ticket #{ticket} has been resolved. We've fixed the login issue you reported. Please let us know if you need further assistance.",
            "sender": "support@service.com"
        },
        {
            "subject": "Your Support Request - Case #{case}",
            "body": "Thank you for contacting support. Your request (Case #{case}) has been received and assigned to our technical team. We'll update you within 24 hours.",
            "sender": "help@company.com"
        },
        {
            "subject": "Follow-up on Your Recent Inquiry",
            "body": "This is a follow-up regarding your recent support inquiry. We want to ensure your issue has been fully resolved. Please reply if you need additional help.",
            "sender": "care@customer.com"
        },
    ],
    "travel": [
        {
            "subject": "Flight Booking Confirmation - {airline}",
            "body": "Your flight has been confirmed! Confirmation number: {conf}. Flight {airline}{num} departs on {date}. Please check in 24 hours before departure.",
            "sender": "booking@airline.com"
        },
        {
            "subject": "Hotel Reservation Confirmed",
            "body": "Your hotel reservation is confirmed. Check-in: {date}. Confirmation code: {conf}. We look forward to welcoming you!",
            "sender": "reservations@hotel.com"
        },
        {
            "subject": "Your Itinerary for Upcoming Trip",
            "body": "Your complete travel itinerary is attached. Flight details, hotel confirmation, and rental car information included. Have a great trip!",
            "sender": "trips@expedia.com"
        },
    ],
    "education": [
        {
            "subject": "Assignment Due {day} - Reminder",
            "body": "This is a reminder that your assignment is due on {day}. Please submit via the course portal before 11:59 PM. Review the grading rubric before submitting.",
            "sender": "professor@university.edu"
        },
        {
            "subject": "Course Grades Posted",
            "body": "Grades for the midterm exam have been posted in the student portal. Please review your results and see me during office hours if you have questions.",
            "sender": "instructor@college.edu"
        },
        {
            "subject": "Lecture Notes - Week {week}",
            "body": "This week's lecture notes and study materials are now available. Topics covered: {topic1} and {topic2}. Next class is on {day}.",
            "sender": "prof@academy.edu"
        },
    ],
    "health": [
        {
            "subject": "Appointment Reminder - {day} at {time}",
            "body": "This is a reminder of your upcoming appointment on {day} at {time}. Please arrive 15 minutes early to complete paperwork. Bring your insurance card.",
            "sender": "office@clinic.com"
        },
        {
            "subject": "Your Test Results Are Ready",
            "body": "Your recent test results are now available in the patient portal. Please log in to review. Contact us if you have questions or would like to schedule a follow-up.",
            "sender": "results@hospital.com"
        },
        {
            "subject": "Prescription Ready for Pickup",
            "body": "Your prescription is ready for pickup at our pharmacy. Available Mon-Fri 9am-6pm, Sat 9am-2pm. Please bring your ID.",
            "sender": "pharmacy@health.com"
        },
    ]
}

def generate_training_dataset(emails_per_category: int = 20) -> List[Dict]:
    """Generate training dataset with specified emails per category"""
    dataset = []
    email_id = 1

    for category, templates in EMAIL_TEMPLATES.items():
        print(f"📧 Generating {emails_per_category} emails for: {category}")

        for i in range(emails_per_category):
            template = random.choice(templates)

            # Fill in template variables
            subject = template["subject"].format(
                q=random.randint(1, 4),
                time=f"{random.randint(9, 17)}:00",
                percent=random.randint(20, 80),
                month=random.choice(["January", "February", "March", "April", "May", "June"]),
                ticket=random.randint(1000, 9999),
                case=random.randint(10000, 99999),
                airline=random.choice(["UA", "AA", "DL", "SW"]),
                num=random.randint(100, 999),
                conf=f"{random.choice(['ABC', 'XYZ', 'QRS'])}{random.randint(1000, 9999)}",
                date=random.choice(["tomorrow", "Friday", "next Tuesday", "next Monday"]),
                day=random.choice(["Monday", "Wednesday", "Friday", "next week"]),
                week=random.randint(1, 12),
                topic1=random.choice(["algorithms", "databases", "networking", "security"]),
                topic2=random.choice(["architecture", "testing", "deployment", "optimization"])
            )

            body = template["body"].format(
                q=random.randint(1, 4),
                time=f"{random.randint(9, 17)}:00",
                percent=random.randint(20, 80),
                month=random.choice(["January", "February", "March", "April", "May", "June"]),
                ticket=random.randint(1000, 9999),
                case=random.randint(10000, 99999),
                airline=random.choice(["UA", "AA", "DL", "SW"]),
                num=random.randint(100, 999),
                conf=f"{random.choice(['ABC', 'XYZ', 'QRS'])}{random.randint(1000, 9999)}",
                date=random.choice(["tomorrow", "Friday", "next Tuesday", "next Monday"]),
                day=random.choice(["Monday", "Wednesday", "Friday", "next week"]),
                week=random.randint(1, 12),
                topic1=random.choice(["algorithms", "databases", "networking", "security"]),
                topic2=random.choice(["architecture", "testing", "deployment", "optimization"])
            )

            email = {
                "id": f"email_{email_id:04d}",
                "subject": subject,
                "body": body,
                "sender": template["sender"],
                "label": category,
                "timestamp": datetime.now().isoformat()
            }

            dataset.append(email)
            email_id += 1

        print(f"   ✅ Completed {category}")

    return dataset

def save_dataset(dataset: List[Dict], filename: str = "data/training_emails.json"):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n💾 Saved {len(dataset)} emails to: {filename}")

    # Save statistics
    stats = {
        "total_emails": len(dataset),
        "categories": list(EMAIL_TEMPLATES.keys()),
        "distribution": {
            cat: len([e for e in dataset if e['label'] == cat])
            for cat in EMAIL_TEMPLATES.keys()
        },
        "generated_at": datetime.now().isoformat()
    }

    stats_file = filename.replace('.json', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"📊 Saved statistics to: {stats_file}")
    return filename

def main():
    """Main execution"""
    print("="*60)
    print("TRAINING DATA GENERATOR")
    print("="*60)
    print()

    # Generate dataset
    dataset = generate_training_dataset(emails_per_category=20)

    # Save to file
    filename = save_dataset(dataset)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Emails: {len(dataset)}")
    print(f"Categories: {len(EMAIL_TEMPLATES)}")
    print(f"\nDistribution:")
    for cat in EMAIL_TEMPLATES.keys():
        count = len([e for e in dataset if e['label'] == cat])
        print(f"  {cat:12s}: {count} emails")
    print("="*60)

    return dataset

if __name__ == "__main__":
    main()