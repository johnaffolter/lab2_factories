#!/usr/bin/env python3
"""
Real Data Examples for Email Classification System
Testing with actual realistic email samples
"""

import requests
import json
from datetime import datetime
import time

BASE_URL = "http://localhost:8000"

# Real email examples from different categories
REAL_EMAILS = [
    # Work emails
    {
        "category": "WORK",
        "emails": [
            {
                "subject": "Q4 2024 Budget Review - Action Required",
                "body": """Hi Team,

Please review the attached Q4 budget spreadsheet before our meeting on Friday at 2pm EST.

Key items to discuss:
- Marketing spend increased by 23% over projections
- Engineering headcount below target by 3 positions
- Cloud infrastructure costs up 15% month-over-month

Please come prepared with your department's variance explanations and Q1 2025 projections.

Best regards,
Sarah Chen
CFO"""
            },
            {
                "subject": "Re: Production Deployment Schedule",
                "body": """The deployment window for release v2.3.1 is confirmed for Saturday 2am-6am PST.

Pre-deployment checklist:
✓ All unit tests passing (98.2% coverage)
✓ Integration tests completed
✓ Security scan cleared
□ Database migration scripts reviewed
□ Rollback procedure documented

DevOps team will be on-call. Please ensure your cell phones are charged and available.

- Michael Rodriguez
Sr. DevOps Engineer"""
            },
            {
                "subject": "Annual Performance Review - John Doe",
                "body": """Dear John,

Your annual performance review is scheduled for Tuesday, December 10th at 3:30pm in Conference Room B.

Please complete the self-assessment form in Workday by EOD Monday. We'll discuss your achievements, areas for growth, and 2025 goals.

The review will cover:
• Project deliverables and quality
• Team collaboration and leadership
• Technical skill development
• Career advancement opportunities

Looking forward to our discussion.

Jennifer Park
Engineering Manager"""
            }
        ]
    },

    # Personal emails
    {
        "category": "PERSONAL",
        "emails": [
            {
                "subject": "Mom's 70th Birthday Party Planning 🎉",
                "body": """Hey everyone!

Can't believe Mom's turning 70 next month! Let's make it special.

Party details so far:
- Date: January 15th (Saturday)
- Time: 2pm - 6pm
- Location: The Garden Restaurant (already booked!)
- Guest count: ~45 people

Still need help with:
- Photo slideshow (Tom, can you handle this?)
- Cake order (3-tier, chocolate and vanilla)
- Music playlist (her favorites from the 70s & 80s)

Let me know your thoughts! Also, don't forget we're splitting the cost 4 ways.

Love you all!
Emily"""
            },
            {
                "subject": "Congrats on the new house! 🏡",
                "body": """Jamie!

Just saw your Instagram post - the house looks AMAZING! I'm so happy for you and Mark!

That kitchen is to die for, and the backyard is perfect for BBQs. We definitely need to plan a housewarming party once you're settled in.

Let me know if you need help moving or unpacking. I'm free most weekends and have a truck we can use.

Can't wait to see it in person!

xoxo,
Lisa"""
            }
        ]
    },

    # Promotional emails
    {
        "category": "PROMOTION",
        "emails": [
            {
                "subject": "⚡ FLASH SALE: 60% Off Everything - Today Only!",
                "body": """LIMITED TIME OFFER - ENDS MIDNIGHT TONIGHT!

Shop our biggest sale of the year with 60% OFF EVERYTHING!

No exclusions. No gimmicks. Just incredible savings on:
• Designer clothing
• Electronics
• Home decor
• Beauty products
• And much more!

Use code: FLASH60 at checkout

⏰ Hurry! Sale ends in 14 hours

FREE SHIPPING on orders over $50
30-day return policy
Shop now: www.megastore.com/flash-sale

*Cannot be combined with other offers. While supplies last."""
            },
            {
                "subject": "You're Pre-Approved for our Premium Rewards Card! 💳",
                "body": """Congratulations! You've been pre-approved for our Premium Rewards Card.

Exclusive benefits include:
✓ $200 sign-up bonus after spending $500 in first 3 months
✓ 5% cash back on groceries and gas
✓ 3% back on restaurants and travel
✓ 1% back on all other purchases
✓ No annual fee for the first year
✓ 0% APR for 18 months on balance transfers

Your pre-approved credit limit: $15,000

Apply now - it only takes 60 seconds!
This offer expires in 7 days.

Click here to claim your card → [APPLY NOW]"""
            }
        ]
    },

    # Support emails
    {
        "category": "SUPPORT",
        "emails": [
            {
                "subject": "Ticket #78432: Unable to Reset Password",
                "body": """Hello Support Team,

I've been trying to reset my password for the past hour but I'm not receiving the reset email.

Account details:
- Username: johndoe2024
- Email: john.doe@email.com
- Last successful login: December 1, 2024

I've checked my spam folder and tried multiple times. This is urgent as I need to access important documents for a meeting tomorrow morning.

Please help ASAP.

Thank you,
John Doe
Account ID: ACC-445789"""
            },
            {
                "subject": "Re: Case #92156 - Billing Discrepancy",
                "body": """Dear Customer Service,

Following up on my previous email about the incorrect charge on my account.

Issue summary:
- Charged $299.99 on Nov 28
- Should have been $199.99 (Black Friday promotion)
- Confirmation number: BF2024-88923
- Promotional code was applied but discount not reflected

I've attached the confirmation email showing the promotional price.

Please process the $100 refund at your earliest convenience.

Best regards,
Amanda Foster
Customer since 2019"""
            }
        ]
    },

    # Newsletter emails
    {
        "category": "NEWSLETTER",
        "emails": [
            {
                "subject": "TechWeekly Digest: AI Breakthroughs, Startup Funding & More",
                "body": """Your Weekly Tech News Roundup - December 8, 2024

TOP STORIES THIS WEEK:

🤖 OpenAI Announces GPT-5 Preview
The next generation of AI is here with 10x faster processing and enhanced reasoning capabilities.
[Read more →]

💰 FinTech Startup Raises $500M Series C
Payment processor FastPay valued at $5B after latest funding round led by Sequoia.
[Read more →]

📱 Apple's Foldable iPhone: 2025 Release Confirmed?
Leaked patents suggest revolutionary design coming next year.
[Read more →]

🔐 Major Security Breach at SocialCorp
150 million user records exposed in largest breach of 2024.
[Read more →]

UPCOMING EVENTS:
• Dec 15: Virtual AI Summit 2024
• Jan 8-11: CES Las Vegas
• Jan 20: Startup Pitch Night SF

Want to advertise? Contact newsletter@techweekly.com
Unsubscribe | Update preferences | View in browser"""
            }
        ]
    },

    # Finance emails
    {
        "category": "FINANCE",
        "emails": [
            {
                "subject": "Your November 2024 Investment Portfolio Summary",
                "body": """Portfolio Performance Summary - November 2024

Account Value: $247,832.15
Monthly Return: +4.7%
YTD Return: +18.3%

TOP PERFORMERS:
• NVIDIA (NVDA): +12.3%
• Apple (AAPL): +8.1%
• S&P 500 ETF (SPY): +5.2%

UNDERPERFORMERS:
• Tesla (TSLA): -6.8%
• Bitcoin ETF (BITO): -11.2%

RECOMMENDED ACTIONS:
Consider rebalancing - your tech allocation is now 42% (target: 35%)
Tax loss harvesting opportunity in cryptocurrency holdings

Schedule a call with your advisor: calendly.com/advisor-smith

Schwab Wealth Management
This is not financial advice. Past performance doesn't guarantee future results."""
            },
            {
                "subject": "🏦 Your Chase Statement is Ready",
                "body": """Your November statement is now available.

CHECKING ACCOUNT (...8923)
Beginning Balance: $8,234.67
Deposits: $6,500.00
Withdrawals: $5,123.45
Ending Balance: $9,611.22

CREDIT CARD (...4567)
Current Balance: $2,341.89
Minimum Payment Due: $75.00
Payment Due Date: Dec 15, 2024

Recent transactions:
• Amazon.com: $156.32
• Whole Foods: $234.89
• Shell Gas: $67.45

View full statement in the Chase app or at chase.com

Set up autopay to never miss a payment!"""
            }
        ]
    }
]

def test_real_emails():
    """Test classification with real email examples"""

    print("=" * 80)
    print("REAL EMAIL CLASSIFICATION TEST")
    print("Testing with actual email samples from various categories")
    print("=" * 80)

    results = []

    for category_data in REAL_EMAILS:
        category = category_data["category"]
        print(f"\n📧 Testing {category} Emails")
        print("-" * 40)

        for email in category_data["emails"]:
            # Classify the email
            response = requests.post(
                f"{BASE_URL}/emails/classify",
                json={
                    "subject": email["subject"],
                    "body": email["body"],
                    "use_email_similarity": False
                }
            )

            if response.status_code == 200:
                result = response.json()

                # Store result
                results.append({
                    "expected": category.lower(),
                    "predicted": result["predicted_topic"],
                    "confidence": result["topic_scores"][result["predicted_topic"]],
                    "subject": email["subject"][:50],
                    "features": result["features"]
                })

                # Print result
                print(f"\n📨 Subject: {email['subject'][:60]}...")
                print(f"   Expected: {category}")
                print(f"   Predicted: {result['predicted_topic'].upper()}")
                print(f"   Confidence: {result['topic_scores'][result['predicted_topic']]:.2%}")

                # Show feature extraction
                features = result["features"]
                print(f"   Features:")
                print(f"     • Spam words: {features.get('spam_has_spam_words', 0)}")
                print(f"     • Avg word length: {features.get('word_length_average_word_length', 0):.2f}")
                print(f"     • Special chars: {features.get('non_text_non_text_char_count', 0)}")

                # Show top 3 predictions
                scores = result["topic_scores"]
                top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top classifications:")
                for topic, score in top_3:
                    print(f"     • {topic}: {score:.2%}")
            else:
                print(f"   ❌ Error classifying email: {response.status_code}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    # Calculate accuracy
    correct = sum(1 for r in results if r["expected"] == r["predicted"])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n📊 Overall Statistics:")
    print(f"   • Total emails tested: {total}")
    print(f"   • Correctly classified: {correct}")
    print(f"   • Accuracy: {accuracy:.1f}%")
    print(f"   • Average confidence: {sum(r['confidence'] for r in results)/total:.2%}")

    # Category breakdown
    print(f"\n📈 Performance by Category:")
    for category in ["work", "personal", "promotion", "support", "newsletter", "finance"]:
        cat_results = [r for r in results if r["expected"] == category]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r["predicted"] == category)
            cat_accuracy = (cat_correct / len(cat_results) * 100) if cat_results else 0
            print(f"   • {category.upper()}: {cat_accuracy:.0f}% ({cat_correct}/{len(cat_results)})")

    # Feature statistics
    print(f"\n🔍 Feature Extraction Stats:")
    avg_spam = sum(r["features"].get("spam_has_spam_words", 0) for r in results) / total
    avg_word_len = sum(r["features"].get("word_length_average_word_length", 0) for r in results) / total
    avg_special = sum(r["features"].get("non_text_non_text_char_count", 0) for r in results) / total

    print(f"   • Emails with spam words: {avg_spam*100:.1f}%")
    print(f"   • Average word length: {avg_word_len:.2f}")
    print(f"   • Average special characters: {avg_special:.1f}")

    return results

def store_training_data():
    """Store some emails as training data with ground truth"""

    print("\n" + "=" * 80)
    print("STORING TRAINING DATA")
    print("=" * 80)

    training_samples = [
        {"subject": "Q3 Budget Review", "body": "Please review the attached budget", "ground_truth": "work"},
        {"subject": "Happy Birthday!", "body": "Hope you have an amazing day!", "ground_truth": "personal"},
        {"subject": "50% OFF Sale", "body": "Limited time offer - shop now!", "ground_truth": "promotion"},
        {"subject": "Password Reset Help", "body": "I cannot access my account", "ground_truth": "support"},
        {"subject": "Weekly Tech News", "body": "This week in technology...", "ground_truth": "newsletter"},
        {"subject": "Portfolio Update", "body": "Your investments gained 5% this month", "ground_truth": "finance"}
    ]

    for sample in training_samples:
        response = requests.post(f"{BASE_URL}/emails", json=sample)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stored: {sample['subject']} (ID: {result.get('email_id')})")
        else:
            print(f"❌ Failed to store: {sample['subject']}")

    print("\nTraining data stored successfully!")

if __name__ == "__main__":
    # First store some training data
    store_training_data()

    # Then test with real emails
    time.sleep(1)
    results = test_real_emails()

    print("\n" + "=" * 80)
    print("✅ REAL DATA TESTING COMPLETE")
    print("=" * 80)