"""
Massive Training Data Generation
Generates 500+ diverse training emails across 15 topics with varied characteristics
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Expanded topic definitions with more nuanced examples
TOPIC_DEFINITIONS = {
    "work_project": {
        "templates": [
            "Project Status: {} - Q{} Update",
            "{} Milestone Completed - Next Steps",
            "Team Sync: {} Discussion Points",
            "Project Deliverables for {}",
            "{} Sprint Review and Planning"
        ],
        "bodies": [
            "The {} project is progressing well. We've completed {} of the planned features. Next sprint focuses on {}. Team velocity is {} story points per week.",
            "Quick update on {}: Backend API is {}, frontend is {}, database migration is {}. Blockers: {}. ETA: {}.",
            "Project {} status: On track for {} delivery. Key achievements: {}. Risks: {}. Action items: {}.",
            "Sprint retrospective for {}: What went well: {}. What needs improvement: {}. Action items: {}.",
            "{} project requires attention. Current status: {}. Critical path: {}. Resources needed: {}."
        ],
        "keywords": ["deadline", "sprint", "milestone", "deliverable", "status", "progress", "team", "blockers"]
    },

    "technical_support": {
        "templates": [
            "ISSUE: {} - {} Environment",
            "Bug Report: {} in {} Module",
            "Technical Incident: {} Down",
            "Support Ticket #{}: {}",
            "System Alert: {} Degradation"
        ],
        "bodies": [
            "Critical issue detected in {} system. Error: {}. Impact: {} users affected. Priority: {}. Investigation started at {}.",
            "Bug found in {} module. Steps to reproduce: {}. Expected behavior: {}. Actual behavior: {}. Stack trace: {}.",
            "Production incident: {} service is {}. Root cause: {}. Workaround: {}. ETA for fix: {}.",
            "Performance degradation in {}: Response time increased {}%. CPU utilization: {}%. Memory: {}%. Database queries: slow.",
            "User reports {} error when {}. Frequency: {} per hour. Severity: {}. Logs: {}. Next steps: {}."
        ],
        "keywords": ["error", "bug", "crash", "issue", "incident", "down", "critical", "urgent", "fix"]
    },

    "sales_marketing": {
        "templates": [
            "Q{} Sales Targets - {} Region",
            "Marketing Campaign: {} Results",
            "Lead Generation: {} Strategy",
            "Customer Acquisition: {} Quarter",
            "Product Launch: {} Market Analysis"
        ],
        "bodies": [
            "Q{} sales performance: Revenue ${}, {} deals closed, {} pipeline. Top performers: {}. Forecast: {}.",
            "{} campaign metrics: {} leads generated, {}% conversion, {} CAC, {} ROI. Next steps: {}.",
            "Market analysis for {}: TAM ${}, competitors: {}, our USP: {}, growth rate: {}% YoY.",
            "Sales pipeline review: {} qualified leads, {} in negotiation, {} closing this quarter. Win rate: {}%.",
            "Customer feedback on {}: NPS {}, CSAT {}%, key insights: {}. Action items: {}."
        ],
        "keywords": ["sales", "revenue", "leads", "campaign", "conversion", "ROI", "target", "growth"]
    },

    "hr_recruitment": {
        "templates": [
            "Position Opening: {} - {} Team",
            "Candidate Interview: {} for {}",
            "Team Onboarding: {} Starts {}",
            "Performance Review: Q{} Cycle",
            "HR Policy Update: {}"
        ],
        "bodies": [
            "We're hiring for {}: {}+ years experience, skills: {}, location: {}, salary range: ${}. JD attached.",
            "Interview feedback for {} ({}): Technical skills: {}/5, culture fit: {}/5, communication: {}/5. Recommendation: {}.",
            "{} joins the {} team on {}. Onboarding checklist: hardware setup, access provisioning, training schedule.",
            "Q{} performance reviews: {}% completion rate, average rating: {}/5, promotions: {}, raises: {}%.",
            "Updated {} policy effective {}: Key changes: {}. All employees must acknowledge by {}."
        ],
        "keywords": ["hiring", "interview", "candidate", "onboarding", "performance", "benefits", "policy"]
    },

    "finance_accounting": {
        "templates": [
            "Invoice #{}: ${}",
            "Q{} Financial Report",
            "Expense Approval: {}",
            "Budget Review: {} Department",
            "Payment Reminder: Account {}"
        ],
        "bodies": [
            "Invoice #{} for ${} is due on {}. Services: {}. Payment terms: net {}. Account: {}.",
            "Q{} financials: Revenue ${}, expenses ${}, profit margin {}%, burn rate ${}/month, runway {} months.",
            "Expense report for {}: Total ${}, categories: travel ${}, meals ${}, software ${}. Approval needed.",
            "Budget variance analysis: {} department overspent by {}% due to {}. Forecast adjustment needed.",
            "Payment overdue for invoice #{}. Amount: ${}. Original due date: {}. Please remit payment ASAP."
        ],
        "keywords": ["invoice", "payment", "expense", "budget", "financial", "account", "transaction"]
    },

    "customer_support": {
        "templates": [
            "Support Ticket #{}: {}",
            "Customer Inquiry: {} Issue",
            "Product Question: {}",
            "Refund Request: Order #{}",
            "Technical Help Needed"
        ],
        "bodies": [
            "Customer {} reports {} issue. Order #{}. Problem: {}. Urgency: {}. Last contact: {}.",
            "Support ticket #{}: {} questions about {}. Response time: {} hours. Status: {}.",
            "{} requests refund for order #{}. Reason: {}. Amount: ${}. Policy: {}. Resolution: {}.",
            "Customer satisfaction score: {}%. Common issues: {}, {}, {}. Average resolution time: {} hours.",
            "Escalated case #{}: Customer {} frustrated with {}. History: {} previous tickets. Action: {}."
        ],
        "keywords": ["customer", "support", "ticket", "issue", "help", "problem", "refund", "question"]
    },

    "product_development": {
        "templates": [
            "Feature Request: {}",
            "Product Roadmap: Q{} Priorities",
            "User Feedback: {} Feature",
            "A/B Test Results: {}",
            "Product Launch: {}"
        ],
        "bodies": [
            "Feature request from {} customers: {}. Business value: {}. Effort: {} story points. Priority: {}.",
            "Q{} product roadmap: Priority 1: {}, Priority 2: {}, Priority 3: {}. Resources: {} engineers.",
            "User feedback on {}: {}% satisfied, common complaints: {}, feature requests: {}. Insights: {}.",
            "A/B test '{}': Variant A {}% conversion, Variant B {}% conversion. Winner: {}. Confidence: {}%.",
            "Product launch plan for {}: Target date: {}, marketing: {}, docs: {}, training: {}. Risks: {}."
        ],
        "keywords": ["feature", "roadmap", "feedback", "launch", "development", "product", "release"]
    },

    "legal_compliance": {
        "templates": [
            "Contract Review: {}",
            "Compliance Audit: {}",
            "Legal Notice: {}",
            "Privacy Policy Update",
            "Terms of Service Amendment"
        ],
        "bodies": [
            "Contract review for {}: Key terms: {}, liability: {}, termination: {}. Legal review: {}.",
            "Compliance audit for {}: Findings: {}, severity: {}, remediation: {}, deadline: {}.",
            "Legal notice regarding {}: Action required: {}, deadline: {}, consequences: {}. Contact legal if questions.",
            "Privacy policy updated per {} regulations. Changes: {}, effective: {}, user notification: {}.",
            "GDPR/CCPA compliance check: Data inventory {}, consent mechanisms {}, breach procedures {}. Status: {}."
        ],
        "keywords": ["legal", "contract", "compliance", "policy", "terms", "privacy", "audit", "regulation"]
    },

    "training_education": {
        "templates": [
            "Training Session: {}",
            "Course Enrollment: {}",
            "Certification Program: {}",
            "Workshop: {} Skills",
            "Learning Path: {}"
        ],
        "bodies": [
            "Training on {} scheduled for {}. Duration: {} hours. Prerequisites: {}. Registration: {}.",
            "Certification program in {}: {} modules, {} weeks, exam on {}. Pass rate: {}%. Cost: ${}.",
            "Workshop '{}': Learn {}, {}, and {}. Instructor: {}. Materials: included. Capacity: {} attendees.",
            "New learning path available: {} fundamentals. Includes: {} courses, {} projects, {} certificate.",
            "Training completion: {}% finished {}, {}% passed assessment, {} hours logged. Next: {}."
        ],
        "keywords": ["training", "course", "certification", "workshop", "learning", "education", "skills"]
    },

    "security_incident": {
        "templates": [
            "SECURITY ALERT: {}",
            "Incident Report: {} Breach",
            "Vulnerability: {} Detected",
            "Security Patch: {}",
            "Phishing Alert: {}"
        ],
        "bodies": [
            "CRITICAL: Security incident detected. Type: {}. Affected systems: {}. Exposure: {}. Containment: in progress.",
            "Vulnerability {} identified in {}. CVSS score: {}. Severity: {}. Patch available: {}. Apply by: {}.",
            "Phishing campaign targeting employees. Subject line: '{}'. Indicators: {}. Report suspicious emails immediately.",
            "Incident response for {}: Detection time: {}, containment: {}, investigation: {}, remediation: {}.",
            "Security audit findings: {} high-priority items, {} medium, {} low. Compliance: {}%. Action plan: {}."
        ],
        "keywords": ["security", "breach", "vulnerability", "phishing", "incident", "patch", "threat", "urgent"]
    },

    "event_meeting": {
        "templates": [
            "Event: {} on {}",
            "Meeting Invite: {}",
            "Conference: {} Registration",
            "Webinar: {}",
            "Team Offsite: {}"
        ],
        "bodies": [
            "You're invited to {} on {}. Location: {}. Agenda: {}. RSVP by: {}.",
            "Meeting scheduled: {} at {}. Attendees: {}. Prep needed: {}. Duration: {} minutes.",
            "Annual conference '{}': {} speakers, {} sessions, networking. Early bird: ${}, regular: ${}.",
            "Webinar '{}' on {}. Topics: {}, {}, {}. Register at: {}. Q&A included.",
            "Team offsite {}: Location: {}, dates: {}, activities: {}. Goals: team building, strategic planning."
        ],
        "keywords": ["meeting", "event", "conference", "invite", "schedule", "calendar", "appointment"]
    },

    "personal_family": {
        "templates": [
            "{}'s Birthday Party",
            "Family Reunion - {}",
            "Weekend Plans",
            "Vacation Planning: {}",
            "Family Update"
        ],
        "bodies": [
            "Join us for {}'s birthday on {}! Location: {}. Time: {}. Bring: {}. RSVP: {}.",
            "Family reunion at {} from {} to {}. Accommodations: {}. Activities: {}. Looking forward to seeing everyone!",
            "This weekend: {} on Saturday, {} on Sunday. Want to join for {}? Let me know!",
            "Planning {} vacation for {}. Considering: {}, {}, or {}. Budget: ${}. Thoughts?",
            "Family news: {} started {}, {} graduated from {}, {} moved to {}. Everyone doing well!"
        ],
        "keywords": ["family", "birthday", "party", "vacation", "reunion", "weekend", "personal"]
    },

    "health_wellness": {
        "templates": [
            "Appointment: {} Checkup",
            "Wellness Program: {}",
            "Health Insurance: {}",
            "Fitness Challenge: {}",
            "Medical Results"
        ],
        "bodies": [
            "Appointment reminder: {} checkup on {} at {}. Location: {}. Bring: insurance card, ID.",
            "New wellness program: {} challenges, prizes, tracking app. Goal: improve {}, {}, {}.",
            "Health insurance enrollment: Plan options: {}, {}, {}. Deadline: {}. Benefits: {}.",
            "Fitness challenge update: {} completed this week, {} participants, leader: {}. Keep it up!",
            "Lab results available. Key findings: {}. Follow-up: {} needed. Next appointment: {}."
        ],
        "keywords": ["health", "appointment", "wellness", "fitness", "medical", "insurance", "doctor"]
    },

    "shopping_ecommerce": {
        "templates": [
            "Order Confirmation: #{}",
            "Shipping Update: Package #{}",
            "Product Recommendation: {}",
            "Special Offer: {}% Off",
            "Review Request: Your Order"
        ],
        "bodies": [
            "Order #{} confirmed! Items: {}. Total: ${}. Shipping: {}. Expected delivery: {}.",
            "Your package #{} shipped via {}. Tracking: {}. Estimated delivery: {}. Track at: {}.",
            "Based on your purchase of {}, you might like: {}, {}, {}. Special price: ${}.",
            "FLASH SALE: {}% off {} today only! Use code: {}. Expires: {}. Shop now: {}.",
            "How was your recent order #{}? Rate your experience and earn {} rewards points. Review: {}."
        ],
        "keywords": ["order", "shipping", "delivery", "product", "purchase", "discount", "sale"]
    },

    "social_community": {
        "templates": [
            "Community Event: {}",
            "Group Discussion: {}",
            "Volunteer Opportunity: {}",
            "Neighborhood Update",
            "Social Gathering: {}"
        ],
        "bodies": [
            "Community event '{}' this {}! Join {} neighbors for {}. Location: {}. Time: {}.",
            "Active discussion on {}: {} comments, {} participants. Top insights: {}. Join at: {}.",
            "Volunteer opportunity: {} needs help with {}. Time commitment: {} hours. Sign up: {}.",
            "Neighborhood update: {}, {}, and {}. Meeting on {} at {}. All residents welcome.",
            "You're invited to {} at {}'s place on {}. Casual gathering, bring: {}. RSVP: {}."
        ],
        "keywords": ["community", "social", "neighbor", "volunteer", "gathering", "group", "event"]
    }
}

def generate_random_date():
    """Generate random date within last 90 days"""
    days_ago = random.randint(0, 90)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def fill_template(template: str, topic: str) -> str:
    """Fill template with realistic values"""
    replacements = {
        "work_project": ["Alpha", "Beta", "Migration", "Platform", "Infrastructure"],
        "technical_support": ["Login", "Payment", "API", "Database", "Production"],
        "sales_marketing": ["Email", "Social", "Digital", "Content", "Outbound"],
        "hr_recruitment": ["Senior Engineer", "Product Manager", "Designer", "Analyst"],
        "finance_accounting": [f"{random.randint(10000, 99999)}", f"{random.randint(500, 50000)}"],
        "customer_support": ["Billing", "Technical", "General", "Feature"],
        "product_development": ["Dashboard", "API", "Mobile App", "Analytics"],
        "legal_compliance": ["Vendor Agreement", "NDA", "SLA", "MSA"],
        "training_education": ["Python", "Leadership", "Data Science", "Communication"],
        "security_incident": ["Malware", "DDoS", "Data Leak", "SQL Injection"],
        "event_meeting": ["Quarterly Review", "All Hands", "Planning Session"],
        "personal_family": ["Sarah", "Mom", "John", "Kids"],
        "health_wellness": ["Annual", "Dental", "Vision", "Physical"],
        "shopping_ecommerce": [f"{random.randint(100000, 999999)}"],
        "social_community": ["Park Cleanup", "Book Club", "BBQ", "Meetup"]
    }

    values = replacements.get(topic, ["Update", "Review", "Discussion"])
    quarters = [str(random.randint(1, 4))]

    try:
        filled = template.format(*random.choices(values + quarters, k=template.count("{}")))
        return filled
    except:
        return template.replace("{}", random.choice(values))

def generate_email(topic: str, index: int) -> Dict[str, Any]:
    """Generate single email for given topic"""
    topic_data = TOPIC_DEFINITIONS[topic]

    subject = fill_template(random.choice(topic_data["templates"]), topic)
    body = fill_template(random.choice(topic_data["bodies"]), topic)

    # Add entities to some emails
    if random.random() < 0.3:
        body += f" Contact: user{random.randint(1,100)}@example.com"

    if random.random() < 0.2:
        body += f" Meeting: {generate_random_date()} at {random.randint(9,17)}:00"

    if random.random() < 0.15:
        body += f" Call: {random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

    return {
        "subject": subject,
        "body": body,
        "topic": topic,
        "id": f"{topic}_{index}",
        "generated_at": datetime.now().isoformat()
    }

def generate_massive_dataset(emails_per_topic: int = 40) -> List[Dict[str, Any]]:
    """Generate large dataset across all topics"""
    dataset = []

    print(f"\nGenerating {len(TOPIC_DEFINITIONS)} topics × {emails_per_topic} emails = {len(TOPIC_DEFINITIONS) * emails_per_topic} total emails\n")

    for topic in TOPIC_DEFINITIONS.keys():
        print(f"  Generating {emails_per_topic:3d} emails for: {topic:25s}", end="")
        for i in range(emails_per_topic):
            email = generate_email(topic, i + 1)
            dataset.append(email)
        print(" ✓")

    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    print("="*70)
    print("MASSIVE TRAINING DATA GENERATION")
    print("="*70)

    # Generate 600 emails (40 per topic × 15 topics)
    dataset = generate_massive_dataset(emails_per_topic=40)

    # Save as JSON
    output_file = "training_data_massive.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    # Generate statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Total emails: {len(dataset)}")
    print(f"Topics: {len(TOPIC_DEFINITIONS)}")
    print(f"Average emails per topic: {len(dataset) / len(TOPIC_DEFINITIONS):.1f}")

    # Topic distribution
    topic_counts = {}
    for email in dataset:
        topic_counts[email["topic"]] = topic_counts.get(email["topic"], 0) + 1

    print(f"\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:25s}: {count:3d} emails")

    # Sample emails
    print(f"\n{'='*70}")
    print("SAMPLE EMAILS")
    print(f"{'='*70}\n")

    for i in range(3):
        email = dataset[i]
        print(f"Topic: {email['topic']}")
        print(f"Subject: {email['subject']}")
        print(f"Body: {email['body'][:150]}...")
        print()

    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ Total size: {len(json.dumps(dataset)) / 1024:.1f} KB")