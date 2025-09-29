#!/usr/bin/env python3
"""
Comprehensive Email Generation and Testing Suite
Generates realistic emails and tests the classification system
"""

import requests
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import statistics

BASE_URL = "http://localhost:8000"

class EmailGenerator:
    """Generates realistic emails for different categories"""

    def __init__(self):
        self.email_templates = {
            "work": {
                "subjects": [
                    "Meeting Request: {date} at {time}",
                    "Q{quarter} Budget Review - Action Required",
                    "Project {project} Status Update",
                    "Team Sync: {topic} Discussion",
                    "Deadline Reminder: {task} Due {date}",
                    "Performance Review Schedule",
                    "New Policy Implementation - Please Review",
                    "{department} Department Update",
                    "Urgent: {issue} Needs Attention",
                    "Weekly Report - {week} Summary"
                ],
                "bodies": [
                    "Please join us for a discussion about {topic}. We'll review the current status and plan next steps. Your input is valuable for this meeting.",
                    "The quarterly report is ready for review. Please provide your feedback by EOD {day}. Key metrics show {metric}% improvement.",
                    "I wanted to update you on the {project} project. We've completed {progress}% of the planned milestones. Next steps include {next_step}.",
                    "Following up on our previous conversation about {topic}. Can we schedule time to discuss the implementation plan?",
                    "Please review the attached document and provide your approval. This is related to {topic} and requires sign-off by {date}.",
                    "The team has been working on {task} and we're ready to present our findings. Available times are {time1} or {time2}.",
                    "As discussed in the last meeting, we need to finalize {decision}. Please share your thoughts before {deadline}.",
                    "I'm reaching out regarding the {issue} we discussed. The proposed solution involves {solution}. Let me know your thoughts.",
                    "Quick reminder that {task} is due by {date}. Please ensure all deliverables are submitted to the shared folder.",
                    "The {department} team has completed the analysis. Results indicate {finding}. We should discuss next steps."
                ]
            },
            "personal": {
                "subjects": [
                    "Happy Birthday {name}!",
                    "Weekend Plans?",
                    "Photos from {event}",
                    "Catching up - Long time no talk!",
                    "Family Reunion - Save the Date",
                    "Thank you for {occasion}",
                    "How's {topic} going?",
                    "Miss you!",
                    "Free this {day}?",
                    "Great seeing you at {place}!"
                ],
                "bodies": [
                    "Hey {name}! Hope you're doing well. It's been too long since we last caught up. How about we {activity} sometime soon?",
                    "Thanks so much for {occasion}! It really meant a lot to me. We should definitely {activity} again soon.",
                    "Just wanted to check in and see how you're doing. I heard about {news} - that's amazing! So happy for you.",
                    "Remember when we {memory}? I was just thinking about that and it made me smile. Good times!",
                    "Are you free {day}? I was thinking we could {activity}. It's been ages since we hung out!",
                    "I saw {thing} and it reminded me of you! How have you been? What's new in your life?",
                    "Happy Birthday! I hope your day is filled with {wishes}. Can't wait to celebrate with you!",
                    "The photos from {event} are ready! You looked great. I'll send them over. When can we do this again?",
                    "Just wanted to say thank you for being such a great friend. Your support with {situation} meant everything.",
                    "Hey! I'm going to be in {location} next week. Would love to grab {meal} if you're around!"
                ]
            },
            "promotion": {
                "subjects": [
                    "🎉 {percent}% OFF Everything - {days} Days Only!",
                    "FLASH SALE: Ends in {hours} Hours",
                    "Exclusive Offer Just for You, {name}",
                    "Black Friday Early Access 🛍️",
                    "Last Chance: {product} Sale Ends Tonight",
                    "New Arrivals + Special Discount Inside",
                    "Your {amount} Credit Expires Soon",
                    "⚡ Lightning Deal: {product} at {percent}% Off",
                    "VIP Member Exclusive: Save Big Today",
                    "Weekend Special: Buy One Get One {offer}"
                ],
                "bodies": [
                    "Don't miss out on our biggest sale of the year! Use code {code} to save {percent}% on your entire purchase. Shop now before it's gone!",
                    "As a valued customer, you're getting early access to our {event} sale. Save up to {percent}% on {category}. Limited quantities available!",
                    "Hurry! Only {hours} hours left to save on {product}. This exclusive deal won't last long. Use code {code} at checkout.",
                    "We noticed you left items in your cart. Complete your purchase now and save {amount} with our special offer. Don't let this deal slip away!",
                    "Flash Sale Alert! For the next {hours} hours, everything in {category} is {percent}% off. No code needed - discount applied automatically!",
                    "You've earned it! As a loyal customer, here's your exclusive {amount} off coupon. Valid on purchases over {minimum}. Shop now!",
                    "New arrivals are here! Be the first to shop our latest {category} collection. Plus, save {percent}% with code {code}.",
                    "Last day to save! Our {event} sale ends at midnight. Don't miss {percent}% off {category}. Stock is running low!",
                    "VIP Access Granted! You're invited to our exclusive sale. Save up to {percent}% on premium {category}. This link expires in {hours} hours.",
                    "Special offer just for you! Buy any {product} and get {percent}% off your second item. Perfect for {occasion}!"
                ]
            },
            "newsletter": {
                "subjects": [
                    "{month} Newsletter: {topic} Edition",
                    "Weekly Digest: Top Stories from {source}",
                    "Industry Update: {industry} News",
                    "Your {frequency} Summary is Here",
                    "Trending Now: {topic} Insights",
                    "Newsletter #{number}: {highlight}",
                    "This Week in {field}",
                    "Monthly Roundup: Best of {month}",
                    "{company} Updates & Announcements",
                    "Essential Reading: {topic} Guide"
                ],
                "bodies": [
                    "Welcome to this week's newsletter! This edition covers {topic1}, {topic2}, and {topic3}. Read on for insights and updates.",
                    "Here's your {frequency} digest of the most important news in {industry}. Top story: {headline}. Plus, exclusive insights on {topic}.",
                    "In this issue: {number} things you need to know about {topic}. Expert analysis on {trend}. Upcoming events in {field}.",
                    "Thanks for being a subscriber! This month we're featuring {feature}. Don't miss our special report on {report_topic}.",
                    "Industry trends to watch: {trend1}, {trend2}, and {trend3}. Plus, an exclusive interview with {expert} on {topic}.",
                    "Your personalized content digest: Based on your interests in {interest}, we've curated articles on {topic1} and {topic2}.",
                    "Breaking developments in {field}: {news}. What this means for you and how to prepare for {change}.",
                    "This week's highlights: {achievement} reached new milestone. {company} announces {announcement}. Market analysis inside.",
                    "Essential updates for {profession} professionals: New regulations on {topic}. Best practices for {task}. Tools and resources.",
                    "Thank you for reading! This edition includes: {content1}, {content2}, and our popular {series} series continues."
                ]
            },
            "support": {
                "subjects": [
                    "Ticket #{number}: {issue} - Update",
                    "Password Reset Request",
                    "Account Security Alert",
                    "Service Interruption Notice",
                    "Your Support Request #{number}",
                    "Action Required: Verify Your Account",
                    "System Maintenance Scheduled",
                    "Response to Your Inquiry",
                    "Important: Account Update Needed",
                    "Resolution: {issue} Fixed"
                ],
                "bodies": [
                    "We've received your support request regarding {issue}. Ticket #{number} has been created. Our team will respond within {time} hours.",
                    "A password reset was requested for your account. Click here to reset your password. This link expires in {hours} hours.",
                    "We detected unusual activity on your account from {location}. If this wasn't you, please secure your account immediately.",
                    "Scheduled maintenance will occur on {date} from {start_time} to {end_time}. Services may be unavailable during this period.",
                    "Your issue with {problem} has been resolved. If you continue to experience problems, please reply to this ticket.",
                    "Your account requires verification. Please confirm your identity by {action}. This is required for security purposes.",
                    "Thank you for contacting support. Regarding your question about {topic}: {answer}. Is there anything else we can help with?",
                    "We're investigating the {issue} you reported. Current status: {status}. We'll update you as soon as we have more information.",
                    "Important security update: Please update your {software} to version {version} to maintain account security.",
                    "Good news! The issue affecting {service} has been resolved. All systems are now operational. Thank you for your patience."
                ]
            },
            "travel": {
                "subjects": [
                    "Booking Confirmation: {destination}",
                    "Flight {flight} - Check-in Now Available",
                    "Your {destination} Itinerary",
                    "Travel Alert: {location} Update",
                    "Hotel Reservation Confirmed",
                    "{days} Days Until Your Trip!",
                    "Boarding Pass - {airline} {flight}",
                    "Travel Insurance Quote",
                    "Discover {destination} - Special Offers",
                    "Trip Reminder: {destination} Tomorrow"
                ],
                "bodies": [
                    "Your booking to {destination} is confirmed! Confirmation number: {code}. Departure: {date}. Check-in opens {hours} hours before.",
                    "Your flight {flight} from {origin} to {destination} is confirmed. Departure: {time}. Please arrive {hours} hours early.",
                    "Hotel reservation confirmed at {hotel} in {city}. Check-in: {date1}, Check-out: {date2}. Confirmation: {code}.",
                    "Travel advisory for {destination}: {advisory}. Please review before your departure on {date}.",
                    "Your complete itinerary for {destination}: Flight {flight}, Hotel {hotel}, Activities planned for {days} days.",
                    "Don't forget! Your trip to {destination} is in {days} days. Have you completed {task}? Click here for checklist.",
                    "Exclusive deal: Save {percent}% on {destination} packages. Includes flight, hotel, and {feature}. Book by {date}.",
                    "Online check-in is now available for your flight {flight}. Check in now to select seats and get boarding passes.",
                    "Travel insurance for your {destination} trip: Coverage includes {coverage}. Premium: {amount}. Protect your trip today.",
                    "Welcome to {destination}! Local tips: {tip1}, {tip2}. Emergency contact: {number}. Enjoy your stay!"
                ]
            },
            "education": {
                "subjects": [
                    "Course Starting Soon: {course}",
                    "Assignment Due: {assignment}",
                    "New Learning Path Available",
                    "Certification Exam Reminder",
                    "Webinar Invitation: {topic}",
                    "Your Learning Progress Report",
                    "Registration Open: {program}",
                    "Study Group Meeting: {subject}",
                    "Course Material Updated",
                    "Congratulations on Completing {course}!"
                ],
                "bodies": [
                    "Your course {course} begins on {date}. Materials are now available in the learning portal. First assignment due {due_date}.",
                    "Reminder: {assignment} is due {date} at {time}. Submit through the online portal. Late submissions will receive {penalty}% penalty.",
                    "Congratulations on completing Module {number}! You've earned {points} points. Next module: {next_module} starts {date}.",
                    "Join us for a live webinar on {topic} with expert {speaker}. Date: {date}, Time: {time}. Register now - limited seats!",
                    "New course available: {course}. Learn {skill} in just {duration} weeks. Early bird discount of {percent}% ends {date}.",
                    "Your learning streak: {days} days! Keep it up! Today's recommended lesson: {lesson}. Estimated time: {time} minutes.",
                    "Exam reminder: {exam} scheduled for {date}. Review materials available. Practice tests can be found in {location}.",
                    "Feedback on your {assignment}: Grade: {grade}. Instructor comments: {comment}. Office hours available {day} at {time}.",
                    "Certificate earned! You've successfully completed {course} with {grade}% score. Your certificate is attached.",
                    "Study group for {subject} meeting {day} at {time} in {location}. Topics: {topic1}, {topic2}. Bring your questions!"
                ]
            },
            "health": {
                "subjects": [
                    "Appointment Reminder: {date} at {time}",
                    "Test Results Available",
                    "Prescription Ready for Pickup",
                    "Health Insurance Update",
                    "Wellness Check Reminder",
                    "New Message from Dr. {doctor}",
                    "Vaccination Record Updated",
                    "Health Alert: {topic}",
                    "Annual Checkup Due",
                    "Telehealth Appointment Confirmed"
                ],
                "bodies": [
                    "Your appointment with Dr. {doctor} is confirmed for {date} at {time}. Location: {location}. Please arrive 15 minutes early.",
                    "Your recent test results are now available in your patient portal. Please log in to review them. Contact us with questions.",
                    "Your prescription for {medication} is ready for pickup at {pharmacy}. Copay: {amount}. Expires: {date}.",
                    "Important insurance update: Your coverage for {service} has changed. New copay: {amount}. Effective date: {date}.",
                    "It's time for your annual {checkup} checkup. Please schedule an appointment at your earliest convenience.",
                    "Dr. {doctor} has sent you a message regarding {topic}. Please log into your patient portal to view and respond.",
                    "Reminder: Take {medication} as prescribed. Refills remaining: {number}. Order refill at least {days} days before running out.",
                    "Your vaccination record has been updated. {vaccine} administered on {date}. Next dose due: {next_date}.",
                    "Health tip: {tip}. Based on your health profile, we recommend {recommendation}. Learn more in our wellness center.",
                    "Telehealth appointment confirmed with Dr. {doctor} on {date} at {time}. Video link will be sent {minutes} minutes before."
                ]
            }
        }

        # Placeholder data for templates
        self.placeholders = {
            "name": ["John", "Sarah", "Mike", "Emma", "David", "Lisa"],
            "date": ["Monday", "March 15", "Next Week", "Tomorrow", "April 1st"],
            "time": ["2:00 PM", "10:30 AM", "3:15 PM", "9:00 AM", "4:45 PM"],
            "quarter": ["1", "2", "3", "4"],
            "project": ["Alpha", "Phoenix", "Nexus", "Titan", "Mercury"],
            "topic": ["strategy", "budget", "roadmap", "performance", "innovation"],
            "task": ["Report", "Proposal", "Analysis", "Presentation", "Review"],
            "department": ["Marketing", "Sales", "Engineering", "HR", "Finance"],
            "issue": ["Server Outage", "Budget Overrun", "Deadline Change", "Resource Allocation"],
            "week": ["Week 12", "Week 13", "Week 14", "Week 15"],
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "metric": ["15", "23", "8", "42", "31"],
            "progress": ["75", "50", "90", "25", "60"],
            "next_step": ["testing", "deployment", "review", "approval", "implementation"],
            "decision": ["vendor selection", "budget allocation", "timeline", "resources"],
            "deadline": ["EOD Friday", "Next Monday", "End of Month", "Tomorrow 5PM"],
            "solution": ["upgrading servers", "hiring contractors", "process automation", "training"],
            "finding": ["20% improvement", "cost savings identified", "risk factors found"],
            "event": ["Summer BBQ", "Birthday Party", "Wedding", "Reunion", "Concert"],
            "occasion": ["your help", "the gift", "dinner", "the surprise", "everything"],
            "activity": ["grab coffee", "have lunch", "go hiking", "watch a movie", "catch up"],
            "memory": ["went to Vegas", "graduated", "started the company", "traveled to Japan"],
            "news": ["your promotion", "the engagement", "the new job", "the move"],
            "wishes": ["joy and laughter", "love and happiness", "success", "good health"],
            "thing": ["that article", "your favorite band", "that restaurant", "our old photo"],
            "situation": ["the move", "job search", "difficult time", "decision"],
            "location": ["downtown", "your area", "the city", "San Francisco", "New York"],
            "meal": ["coffee", "lunch", "dinner", "breakfast", "drinks"],
            "percent": ["20", "30", "40", "50", "60", "70"],
            "days": ["3", "5", "7", "10", "14"],
            "hours": ["2", "4", "6", "12", "24", "48"],
            "product": ["Laptops", "Shoes", "Electronics", "Clothing", "Accessories"],
            "code": ["SAVE20", "FLASH50", "DEAL30", "VIP40", "SPECIAL25"],
            "amount": ["$10", "$25", "$50", "$100", "$15"],
            "category": ["Electronics", "Fashion", "Home & Garden", "Sports", "Books"],
            "minimum": ["$50", "$75", "$100", "$150", "$200"],
            "month": ["January", "February", "March", "April", "May"],
            "frequency": ["Weekly", "Monthly", "Daily", "Quarterly"],
            "industry": ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing"],
            "source": ["TechNews", "Industry Weekly", "Market Watch", "Business Daily"],
            "number": ["101", "52", "237", "48", "163"],
            "highlight": ["Innovation Special", "Year in Review", "Expert Insights", "Market Analysis"],
            "field": ["Technology", "Business", "Science", "Marketing", "Finance"],
            "company": ["TechCorp", "Global Industries", "Innovation Labs", "Future Systems"],
            "headline": ["Market reaches new high", "Breakthrough announced", "Merger completed"],
            "trend": ["AI adoption", "Remote work", "Sustainability", "Digital transformation"],
            "expert": ["Dr. Smith", "Prof. Johnson", "CEO Williams", "Analyst Brown"],
            "interest": ["technology", "business", "innovation", "leadership"],
            "profession": ["Software", "Marketing", "Finance", "Healthcare", "Education"],
            "destination": ["Paris", "Tokyo", "London", "Bali", "New York"],
            "flight": ["AA123", "UA456", "DL789", "BA321", "LH654"],
            "airline": ["American", "United", "Delta", "British Airways", "Lufthansa"],
            "origin": ["JFK", "LAX", "SFO", "ORD", "ATL"],
            "hotel": ["Hilton", "Marriott", "Hyatt", "Sheraton", "Holiday Inn"],
            "city": ["Paris", "London", "Tokyo", "Rome", "Barcelona"],
            "advisory": ["Weather alert", "Security update", "Transit strike", "Health notice"],
            "coverage": ["Medical", "Trip cancellation", "Lost luggage", "Emergency evacuation"],
            "tip1": ["Best restaurants", "Local transport", "Must-see attractions"],
            "course": ["Python Programming", "Data Science", "Web Development", "Machine Learning"],
            "assignment": ["Project 1", "Final Paper", "Lab Report", "Case Study"],
            "skill": ["Programming", "Analytics", "Design", "Leadership"],
            "duration": ["4", "6", "8", "12"],
            "exam": ["Midterm", "Final", "Certification", "Quiz 3"],
            "grade": ["A", "B+", "A-", "B"],
            "lesson": ["Variables and Data Types", "Functions", "Classes", "APIs"],
            "subject": ["Math", "Computer Science", "Physics", "Economics"],
            "doctor": ["Smith", "Johnson", "Williams", "Brown", "Davis"],
            "medication": ["Medication A", "Vitamin D", "Antibiotics", "Pain Relief"],
            "pharmacy": ["CVS", "Walgreens", "Rite Aid", "Local Pharmacy"],
            "service": ["Specialist visits", "Lab tests", "Physical therapy", "Mental health"],
            "checkup": ["Physical", "Dental", "Vision", "Wellness"],
            "vaccine": ["Flu", "COVID-19", "Tetanus", "Hepatitis"],
            "tip": ["Stay hydrated", "Exercise regularly", "Get enough sleep", "Eat balanced meals"]
        }

    def generate_email(self, category: str) -> Dict[str, str]:
        """Generate a single email for a given category"""
        if category not in self.email_templates:
            category = random.choice(list(self.email_templates.keys()))

        template = self.email_templates[category]
        subject_template = random.choice(template["subjects"])
        body_template = random.choice(template["bodies"])

        # Fill in placeholders
        subject = subject_template
        body = body_template

        for placeholder, values in self.placeholders.items():
            placeholder_tag = f"{{{placeholder}}}"
            if placeholder_tag in subject:
                subject = subject.replace(placeholder_tag, random.choice(values))
            if placeholder_tag in body:
                body = body.replace(placeholder_tag, random.choice(values))

        return {
            "subject": subject,
            "body": body,
            "expected_category": category
        }

    def generate_batch(self, count: int = 10, category: str = None) -> List[Dict[str, str]]:
        """Generate multiple emails"""
        emails = []
        categories = [category] * count if category else None

        if not categories:
            # Mix of categories
            categories = []
            for cat in self.email_templates.keys():
                categories.extend([cat] * max(1, count // len(self.email_templates)))
            categories = categories[:count]
            random.shuffle(categories)

        for cat in categories:
            emails.append(self.generate_email(cat))

        return emails


class EmailTestRunner:
    """Tests the classification system with generated emails"""

    def __init__(self):
        self.generator = EmailGenerator()
        self.results = []

    def test_single_email(self, email: Dict[str, str], use_similarity: bool = False) -> Dict:
        """Test a single email classification"""
        try:
            response = requests.post(
                f"{BASE_URL}/emails/classify",
                json={
                    "subject": email["subject"],
                    "body": email["body"],
                    "use_email_similarity": use_similarity
                }
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "predicted": data["predicted_topic"],
                    "expected": email["expected_category"],
                    "correct": data["predicted_topic"] == email["expected_category"],
                    "confidence": data["topic_scores"][data["predicted_topic"]],
                    "all_scores": data["topic_scores"]
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def add_training_data(self, emails: List[Dict[str, str]]) -> int:
        """Add emails as training data with ground truth"""
        added = 0
        for email in emails:
            try:
                response = requests.post(
                    f"{BASE_URL}/emails",
                    json={
                        "subject": email["subject"],
                        "body": email["body"],
                        "ground_truth": email["expected_category"]
                    }
                )
                if response.status_code == 200:
                    added += 1
            except:
                pass
        return added

    def run_comprehensive_test(self):
        """Run comprehensive testing suite"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EMAIL CLASSIFICATION TEST")
        print("="*70)

        # Test 1: Baseline accuracy without training
        print("\n[TEST 1] Baseline Accuracy (No Training Data)")
        print("-" * 50)

        test_emails = self.generator.generate_batch(50)
        baseline_results = []

        for email in test_emails[:10]:  # Test subset
            result = self.test_single_email(email, use_similarity=False)
            baseline_results.append(result)

            if result["success"] and result["correct"]:
                status = "✓"
            else:
                status = "✗"

            print(f"{status} {email['expected_category']:12} -> {result.get('predicted', 'ERROR'):12} "
                  f"(conf: {result.get('confidence', 0):.2f})")

        baseline_accuracy = sum(1 for r in baseline_results if r.get("correct", False)) / len(baseline_results) * 100
        print(f"\nBaseline Accuracy: {baseline_accuracy:.1f}%")

        # Test 2: Add training data
        print("\n[TEST 2] Adding Training Data")
        print("-" * 50)

        training_emails = self.generator.generate_batch(30)
        added = self.add_training_data(training_emails)
        print(f"Added {added} training emails with ground truth labels")

        # Test 3: Test with email similarity
        print("\n[TEST 3] Testing with Email Similarity Mode")
        print("-" * 50)

        similarity_results = []
        for email in test_emails[10:20]:  # Different subset
            result = self.test_single_email(email, use_similarity=True)
            similarity_results.append(result)

            if result["success"] and result["correct"]:
                status = "✓"
            else:
                status = "✗"

            print(f"{status} {email['expected_category']:12} -> {result.get('predicted', 'ERROR'):12} "
                  f"(conf: {result.get('confidence', 0):.2f})")

        similarity_accuracy = sum(1 for r in similarity_results if r.get("correct", False)) / len(similarity_results) * 100
        print(f"\nSimilarity Mode Accuracy: {similarity_accuracy:.1f}%")
        print(f"Improvement: {similarity_accuracy - baseline_accuracy:+.1f}%")

        # Test 4: Category-specific accuracy
        print("\n[TEST 4] Category-Specific Performance")
        print("-" * 50)

        category_results = {}
        for category in ["work", "personal", "promotion", "support", "travel", "education", "health"]:
            cat_emails = self.generator.generate_batch(5, category=category)
            cat_correct = 0

            for email in cat_emails:
                result = self.test_single_email(email, use_similarity=True)
                if result.get("correct", False):
                    cat_correct += 1

            accuracy = cat_correct / len(cat_emails) * 100
            category_results[category] = accuracy
            print(f"  {category:12}: {accuracy:5.1f}% accurate")

        # Test 5: Edge cases
        print("\n[TEST 5] Edge Cases and Difficult Classifications")
        print("-" * 50)

        edge_cases = [
            {
                "subject": "URGENT: Meeting about vacation policy",
                "body": "Discussion about time off and work-life balance",
                "expected_category": "work",
                "note": "Mixed work/personal signals"
            },
            {
                "subject": "50% off professional development courses",
                "body": "Enhance your career with our certified training programs",
                "expected_category": "education",
                "note": "Promotion-like but educational"
            },
            {
                "subject": "Your prescription discount card",
                "body": "Save on medications with our special program",
                "expected_category": "health",
                "note": "Health with promotional elements"
            }
        ]

        for edge_case in edge_cases:
            result = self.test_single_email(edge_case, use_similarity=True)
            print(f"\nEdge Case: {edge_case['note']}")
            print(f"  Subject: '{edge_case['subject'][:50]}...'")
            print(f"  Expected: {edge_case['expected_category']}")
            print(f"  Predicted: {result.get('predicted', 'ERROR')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")

            if result.get("all_scores"):
                top_3 = sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top 3: {', '.join([f'{t[0]}({t[1]:.2f})' for t in top_3])}")

        # Summary Report
        print("\n" + "="*70)
        print("TEST SUMMARY REPORT")
        print("="*70)

        print(f"\n1. Baseline Accuracy: {baseline_accuracy:.1f}%")
        print(f"2. With Training Data: {similarity_accuracy:.1f}%")
        print(f"3. Improvement: {similarity_accuracy - baseline_accuracy:+.1f}%")
        print(f"4. Best Category: {max(category_results.items(), key=lambda x: x[1])[0]} "
              f"({max(category_results.values()):.1f}%)")
        print(f"5. Worst Category: {min(category_results.items(), key=lambda x: x[1])[0]} "
              f"({min(category_results.values()):.1f}%)")

        return {
            "baseline_accuracy": baseline_accuracy,
            "similarity_accuracy": similarity_accuracy,
            "category_results": category_results
        }


def main():
    """Main test execution"""
    print("Email Generation and Testing Suite")
    print("=" * 70)

    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("ERROR: API server not responding. Please start the server with:")
            print("  uvicorn app.main:app --reload")
            return
    except:
        print("ERROR: Cannot connect to API server at", BASE_URL)
        print("Please start the server with: uvicorn app.main:app --reload")
        return

    print("API Server: ✓ Connected")

    # Run tests
    runner = EmailTestRunner()
    results = runner.run_comprehensive_test()

    # Generate sample emails for manual review
    print("\n" + "="*70)
    print("SAMPLE EMAILS FOR MANUAL REVIEW")
    print("="*70)

    generator = EmailGenerator()
    for category in ["work", "personal", "promotion", "travel", "education", "health"]:
        print(f"\n[{category.upper()}]")
        email = generator.generate_email(category)
        print(f"Subject: {email['subject']}")
        print(f"Body: {email['body'][:100]}...")
        print()

    print("\n✅ Testing Complete!")


if __name__ == "__main__":
    main()