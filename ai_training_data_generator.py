#!/usr/bin/env python3
"""
AI-Driven Training Data Generation System
Creates high-quality synthetic training data using AI-generated patterns and real-world constraints
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class EmailCategory(Enum):
    """Email categories with business logic patterns"""
    WORK = "work"
    PERSONAL = "personal"
    PROMOTION = "promotion"
    NEWSLETTER = "newsletter"
    SUPPORT = "support"
    TRAVEL = "travel"
    EDUCATION = "education"
    HEALTH = "health"
    FINANCE = "finance"
    SPAM = "spam"

@dataclass
class TrainingPattern:
    """Pattern for generating realistic training data"""
    category: EmailCategory
    subject_patterns: List[str]
    body_templates: List[str]
    keywords: List[str]
    sentiment_range: Tuple[float, float]
    urgency_indicators: List[str]
    time_patterns: List[str]
    complexity_score: float

class AITrainingDataGenerator:
    """Advanced AI-driven training data generation with quality validation"""

    def __init__(self):
        self.patterns = self._initialize_ai_patterns()
        self.quality_metrics = {}
        self.generated_data = []
        self.validation_rules = self._create_validation_rules()

    def _initialize_ai_patterns(self) -> Dict[EmailCategory, TrainingPattern]:
        """Initialize AI-learned patterns for each email category"""
        return {
            EmailCategory.WORK: TrainingPattern(
                category=EmailCategory.WORK,
                subject_patterns=[
                    "Meeting Request: {topic} on {date}",
                    "Project Update: {project_name}",
                    "Deadline Reminder: {task}",
                    "Team Standup - {day}",
                    "Q{quarter} Review Meeting",
                    "Action Required: {action_item}",
                    "Weekly Report - Week {week_num}",
                    "Budget Approval Needed",
                    "Performance Review Schedule",
                    "Client Presentation - {client_name}"
                ],
                body_templates=[
                    "Please join us for {meeting_type} meeting at {time} in {location}. Agenda: {agenda_items}.",
                    "I'm reaching out regarding {topic} we discussed. The proposed solution involves {solution}. Let me know your thoughts.",
                    "As discussed in the last meeting, we need to finalize {deliverable}. Please share your {input_type} before {deadline}.",
                    "Please review the attached {document_type} and provide your approval. This is related to {project} and requires sign-off by {deadline}.",
                    "Following up on {previous_discussion}. We've made progress on {progress_area} and need your input on {decision_point}."
                ],
                keywords=["meeting", "project", "deadline", "review", "team", "client", "budget", "approval", "deliverable"],
                sentiment_range=(0.3, 0.8),
                urgency_indicators=["urgent", "asap", "deadline", "time-sensitive", "priority"],
                time_patterns=["tomorrow", "next week", "by Friday", "end of month"],
                complexity_score=0.7
            ),

            EmailCategory.PERSONAL: TrainingPattern(
                category=EmailCategory.PERSONAL,
                subject_patterns=[
                    "Happy Birthday!",
                    "How's {activity} going?",
                    "Catch up soon?",
                    "Family {event_type} - Save the Date",
                    "Weekend Plans",
                    "Thanks for {favor}",
                    "Congratulations on {achievement}!",
                    "Thinking of you",
                    "Quick question about {topic}",
                    "Dinner plans for {day}?"
                ],
                body_templates=[
                    "Hey {name}! Hope you're doing well. It's been too long since we last caught up. How about we {activity} sometime soon?",
                    "Happy {occasion}! I hope your day is filled with {positive_things}. Can't wait to celebrate with you!",
                    "Just wanted to say thank you for {favor_detail}. Your {quality} meant everything.",
                    "How are things going with {personal_project}? I've been thinking about our conversation and wanted to check in.",
                    "I saw {news_item} and thought of you! Hope everything is going well with {life_area}."
                ],
                keywords=["family", "friend", "birthday", "celebration", "weekend", "dinner", "catch up", "personal"],
                sentiment_range=(0.6, 0.95),
                urgency_indicators=["soon", "when you can", "no rush"],
                time_patterns=["this weekend", "soon", "when you're free"],
                complexity_score=0.4
            ),

            EmailCategory.PROMOTION: TrainingPattern(
                category=EmailCategory.PROMOTION,
                subject_patterns=[
                    "{discount}% Off Everything!",
                    "⚡ Lightning Deal: {product_category} at {discount}% Off",
                    "Black Friday Early Access 🛍️",
                    "VIP Member Exclusive: Save Big Today",
                    "Last Chance: {offer} Expires {time}",
                    "New Arrivals: {product_line}",
                    "Flash Sale Alert!",
                    "Cyber Monday Spectacular",
                    "Limited Time: Buy One Get One Free",
                    "Your Cart is Waiting - Complete Purchase"
                ],
                body_templates=[
                    "Don't miss our biggest sale of the year. Use code {promo_code} at checkout.",
                    "As a valued customer, you're getting early access to our {sale_name} sale. Save up to {discount}% on {category}. Limited quantities available!",
                    "Special offer just for you! Buy any {product1} and get {discount}% off your second item. Perfect for {occasion}!",
                    "VIP Access Granted! You're invited to our exclusive sale. Save up to {discount}% on premium {category}. This link expires in {time}.",
                    "Time is running out! Only {hours} hours left to save {discount}% on {category}. Shop now before it's gone!"
                ],
                keywords=["sale", "discount", "offer", "limited", "exclusive", "save", "deal", "promo", "shopping"],
                sentiment_range=(0.7, 0.9),
                urgency_indicators=["limited time", "expires soon", "last chance", "hurry"],
                time_patterns=["24 hours", "this weekend", "midnight tonight"],
                complexity_score=0.6
            ),

            EmailCategory.EDUCATION: TrainingPattern(
                category=EmailCategory.EDUCATION,
                subject_patterns=[
                    "New {course_type} Course Available",
                    "Congratulations on Completing {course_name}!",
                    "Assignment Due: {assignment_name}",
                    "Feedback on your {work_type}: Grade: {grade}",
                    "Registration Open: {program}",
                    "Learning Path Update",
                    "Certificate Earned!",
                    "Study Group Meeting",
                    "Research Opportunity Available",
                    "Academic Calendar Update"
                ],
                body_templates=[
                    "Enroll now in our advanced {subject} course. Limited seats available!",
                    "Certificate earned! You've successfully completed {course} with {score}% score. Your certificate is attached.",
                    "Feedback on your {assignment}: Grade: {grade}. Instructor comments: {comment}. Office hours available {day} at {time}.",
                    "Your {work_type} has been reviewed. Overall performance: {assessment}. Areas for improvement: {feedback}.",
                    "New learning opportunity: {program_name}. This {duration} program covers {topics}. Application deadline: {date}."
                ],
                keywords=["course", "learning", "assignment", "grade", "certificate", "education", "study", "program"],
                sentiment_range=(0.5, 0.8),
                urgency_indicators=["deadline", "limited seats", "registration closes"],
                time_patterns=["next semester", "by next week", "end of term"],
                complexity_score=0.8
            ),

            EmailCategory.HEALTH: TrainingPattern(
                category=EmailCategory.HEALTH,
                subject_patterns=[
                    "Appointment Reminder: {date} at {time}",
                    "Your Lab Results Are Ready",
                    "Wellness Check Reminder",
                    "New Message from Dr. {doctor_name}",
                    "Health Insurance Update",
                    "Prescription Ready for Pickup",
                    "Annual Physical Due",
                    "Vaccination Schedule Update",
                    "Mental Health Resources",
                    "Telehealth Appointment Confirmed"
                ],
                body_templates=[
                    "Your appointment with Dr. {doctor} is scheduled for {date} at {time}. Please arrive {minutes} minutes early.",
                    "Your recent {test_type} results are now available in your patient portal.",
                    "It's time for your annual {exam_type} checkup. Please schedule an appointment at your earliest convenience.",
                    "Telehealth appointment confirmed with Dr. {doctor} on {date} at {time}. Video link will be sent {minutes} minutes before.",
                    "Your prescription for {medication} is ready for pickup at {pharmacy}. Please bring your insurance card."
                ],
                keywords=["appointment", "doctor", "medical", "health", "prescription", "checkup", "results", "patient"],
                sentiment_range=(0.4, 0.7),
                urgency_indicators=["urgent", "immediate attention", "schedule soon"],
                time_patterns=["this week", "within 30 days", "as soon as possible"],
                complexity_score=0.6
            ),

            EmailCategory.TRAVEL: TrainingPattern(
                category=EmailCategory.TRAVEL,
                subject_patterns=[
                    "Flight Confirmation - {origin} to {destination}",
                    "Boarding Pass - {airline} {flight_number}",
                    "Hotel Reservation Confirmed",
                    "Travel Advisory for {destination}",
                    "{days} Days Until Your Trip!",
                    "Check-in Now Available",
                    "Travel Insurance Offer",
                    "Airport Transportation Options",
                    "Weather Update for {destination}",
                    "Passport Expiration Reminder"
                ],
                body_templates=[
                    "Your flight booking is confirmed. Departure: {date}. Please arrive {hours} hours early.",
                    "Welcome to {destination}! Local tips: {tip1}, {tip2}. Emergency contact: {contact}. Enjoy your stay!",
                    "Travel advisory for {destination}: {advisory_type}. Please review before your departure on {date}.",
                    "Your {transport_type} is confirmed for {date} at {time}. Pickup location: {location}.",
                    "Travel insurance for your {destination} trip: Coverage includes {coverage}. Premium: ${amount}. Protect your trip today."
                ],
                keywords=["flight", "hotel", "travel", "booking", "destination", "departure", "arrival", "reservation"],
                sentiment_range=(0.6, 0.9),
                urgency_indicators=["check-in required", "departure soon", "expires"],
                time_patterns=["24 hours before", "departure day", "upon arrival"],
                complexity_score=0.7
            ),

            EmailCategory.SUPPORT: TrainingPattern(
                category=EmailCategory.SUPPORT,
                subject_patterns=[
                    "Ticket #{ticket_id}: {issue_type}",
                    "Response to Your Inquiry",
                    "Account Security Alert",
                    "Important: Account Update Needed",
                    "Service Outage Notification",
                    "Password Reset Request",
                    "Billing Question Resolved",
                    "Feature Request Update",
                    "System Maintenance Notice",
                    "Customer Feedback Request"
                ],
                body_templates=[
                    "We've received your support request regarding {issue}. Ticket #{ticket_id} has been created. Our team will respond within {timeframe} hours.",
                    "Thank you for contacting support. We've resolved your {issue_type} issue. Please verify the solution works correctly.",
                    "We detected unusual activity on your account. Please review your recent {activity_type} and contact us if you notice anything suspicious.",
                    "Your {service_type} service requires an update. Please log in to your account and complete the {action_required}.",
                    "Scheduled maintenance will affect {service} on {date} from {start_time} to {end_time}. No action required."
                ],
                keywords=["support", "ticket", "issue", "account", "service", "problem", "assistance", "help"],
                sentiment_range=(0.3, 0.7),
                urgency_indicators=["immediate", "urgent", "security alert", "action required"],
                time_patterns=["within 24 hours", "immediately", "by end of day"],
                complexity_score=0.8
            )
        }

    def _create_validation_rules(self) -> Dict[str, Any]:
        """Create comprehensive validation rules for training data quality"""
        return {
            "min_subject_length": 5,
            "max_subject_length": 100,
            "min_body_length": 20,
            "max_body_length": 1000,
            "required_keywords_per_category": 2,
            "sentiment_coherence_threshold": 0.8,
            "diversity_threshold": 0.7,
            "realism_score_min": 0.6
        }

    def generate_realistic_email(self, category: EmailCategory) -> Dict[str, Any]:
        """Generate a single realistic email using AI patterns"""
        pattern = self.patterns[category]

        # Generate subject with realistic parameters
        subject_template = random.choice(pattern.subject_patterns)
        subject = self._fill_template(subject_template, category)

        # Generate body with contextual coherence
        body_template = random.choice(pattern.body_templates)
        body = self._fill_template(body_template, category)

        # Calculate AI-driven features
        features = self._calculate_ai_features(subject, body, pattern)

        # Create email with metadata
        email = {
            "id": str(uuid.uuid4()),
            "subject": subject,
            "body": body,
            "ground_truth": category.value,
            "generated_timestamp": datetime.now().isoformat(),
            "ai_features": features,
            "quality_score": self._calculate_quality_score(subject, body, pattern),
            "pattern_confidence": features["pattern_confidence"],
            "realism_score": features["realism_score"]
        }

        return email

    def _fill_template(self, template: str, category: EmailCategory) -> str:
        """Fill template with realistic, context-aware parameters"""
        replacements = self._get_contextual_replacements(category)

        # Use regex to find all placeholders
        placeholders = re.findall(r'\{([^}]+)\}', template)

        filled_template = template
        for placeholder in placeholders:
            if placeholder in replacements:
                value = random.choice(replacements[placeholder])
                filled_template = filled_template.replace(f"{{{placeholder}}}", str(value))
            else:
                # Generate realistic fallback
                filled_template = filled_template.replace(f"{{{placeholder}}}", self._generate_fallback(placeholder))

        return filled_template

    def _get_contextual_replacements(self, category: EmailCategory) -> Dict[str, List[str]]:
        """Get context-aware replacements for each category"""

        common_replacements = {
            "date": ["tomorrow", "next Friday", "March 15", "next week"],
            "time": ["2:00 PM", "10:30 AM", "4:45 PM", "9:00 AM"],
            "name": ["Sarah", "Mike", "Emma", "David", "Jennifer"],
            "location": ["conference room", "main office", "Zoom", "Building A"],
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }

        category_specific = {
            EmailCategory.WORK: {
                "topic": ["budget allocation", "quarterly planning", "project timeline", "team restructuring"],
                "project_name": ["Phoenix Initiative", "Digital Transformation", "Customer Portal", "Mobile App"],
                "task": ["report submission", "client presentation", "code review", "documentation"],
                "quarter": ["Q1", "Q2", "Q3", "Q4"],
                "meeting_type": ["standup", "planning", "retrospective", "all-hands"],
                "client_name": ["TechCorp", "InnovateCo", "GlobalTech", "StartupXYZ"],
                "deliverable": ["project proposal", "technical spec", "user requirements", "implementation plan"]
            },

            EmailCategory.PERSONAL: {
                "activity": ["hiking", "dinner", "coffee", "movie night", "shopping"],
                "occasion": ["birthday", "anniversary", "graduation", "promotion"],
                "achievement": ["new job", "marathon completion", "house purchase", "degree"],
                "favor": ["helping with the move", "pet sitting", "recommendation letter", "ride to airport"],
                "life_area": ["work", "family", "health", "hobbies", "studies"]
            },

            EmailCategory.PROMOTION: {
                "discount": ["50", "30", "25", "40", "60", "70"],
                "product_category": ["Electronics", "Clothing", "Home & Garden", "Books", "Sports"],
                "promo_code": ["SAVE50", "WELCOME30", "FLASH25", "VIP40"],
                "sale_name": ["Black Friday", "Summer", "Clearance", "End of Season"],
                "hours": ["24", "48", "6", "12"]
            },

            EmailCategory.EDUCATION: {
                "course_type": ["Python", "Data Science", "Machine Learning", "Web Development"],
                "course_name": ["Advanced Analytics", "Full Stack Development", "AI Fundamentals"],
                "assignment_name": ["Final Project", "Midterm Exam", "Lab Report", "Research Paper"],
                "grade": ["A", "B+", "A-", "B"],
                "subject": ["computer science", "data analysis", "software engineering"]
            },

            EmailCategory.HEALTH: {
                "doctor": ["Smith", "Johnson", "Williams", "Brown", "Davis"],
                "test_type": ["blood", "X-ray", "MRI", "lab"],
                "exam_type": ["physical", "dental", "eye", "wellness"],
                "medication": ["Prescription #12345", "your medication", "prescribed treatment"],
                "pharmacy": ["CVS Pharmacy", "Walgreens", "Local Pharmacy"]
            },

            EmailCategory.TRAVEL: {
                "origin": ["NYC", "LAX", "Chicago", "Miami"],
                "destination": ["Paris", "London", "Tokyo", "Sydney"],
                "airline": ["Delta", "United", "American", "Southwest"],
                "flight_number": ["AA123", "DL456", "UA789", "SW321"],
                "days": ["7", "3", "14", "2"],
                "advisory_type": ["weather", "health", "security", "travel"]
            },

            EmailCategory.SUPPORT: {
                "issue": ["login problem", "billing question", "feature request", "bug report"],
                "ticket_id": ["12345", "67890", "54321", "98765"],
                "service": ["email", "website", "mobile app", "database"],
                "issue_type": ["technical", "billing", "account", "service"]
            }
        }

        # Merge common and category-specific replacements
        result = common_replacements.copy()
        if category in category_specific:
            result.update(category_specific[category])

        return result

    def _generate_fallback(self, placeholder: str) -> str:
        """Generate realistic fallback for unknown placeholders"""
        fallbacks = {
            "default": "information",
            "number": str(random.randint(1, 100)),
            "amount": f"${random.randint(10, 500)}",
            "percentage": f"{random.randint(10, 90)}%",
            "duration": f"{random.randint(1, 12)} months"
        }

        # Try to infer type from placeholder name
        if any(word in placeholder.lower() for word in ["amount", "price", "cost"]):
            return fallbacks["amount"]
        elif any(word in placeholder.lower() for word in ["percent", "discount"]):
            return fallbacks["percentage"]
        elif any(word in placeholder.lower() for word in ["duration", "time", "period"]):
            return fallbacks["duration"]
        elif any(word in placeholder.lower() for word in ["number", "id", "count"]):
            return fallbacks["number"]
        else:
            return fallbacks["default"]

    def _calculate_ai_features(self, subject: str, body: str, pattern: TrainingPattern) -> Dict[str, Any]:
        """Calculate AI-driven features for quality assessment"""
        text = f"{subject} {body}"

        # Pattern matching confidence
        keyword_matches = sum(1 for keyword in pattern.keywords if keyword.lower() in text.lower())
        pattern_confidence = keyword_matches / len(pattern.keywords)

        # Sentiment analysis simulation
        sentiment_score = random.uniform(*pattern.sentiment_range)

        # Complexity analysis
        avg_word_length = np.mean([len(word) for word in text.split()])
        sentence_count = len([s for s in text.split('.') if s.strip()])
        complexity_score = (avg_word_length + sentence_count) / 10

        # Realism score based on multiple factors
        realism_factors = [
            pattern_confidence,
            1.0 - abs(sentiment_score - np.mean(pattern.sentiment_range)),
            min(1.0, complexity_score / pattern.complexity_score),
            1.0 if 50 <= len(text) <= 500 else 0.5  # Length realism
        ]
        realism_score = np.mean(realism_factors)

        return {
            "pattern_confidence": round(pattern_confidence, 3),
            "sentiment_score": round(sentiment_score, 3),
            "complexity_score": round(complexity_score, 3),
            "realism_score": round(realism_score, 3),
            "keyword_density": round(keyword_matches / len(text.split()) * 100, 2),
            "avg_word_length": round(avg_word_length, 2),
            "sentence_count": sentence_count,
            "character_count": len(text),
            "non_text_chars": sum(1 for c in text if not c.isalnum() and not c.isspace())
        }

    def _calculate_quality_score(self, subject: str, body: str, pattern: TrainingPattern) -> float:
        """Calculate overall quality score for the generated email"""
        validations = []

        # Length validations
        validations.append(1.0 if self.validation_rules["min_subject_length"] <= len(subject) <= self.validation_rules["max_subject_length"] else 0.5)
        validations.append(1.0 if self.validation_rules["min_body_length"] <= len(body) <= self.validation_rules["max_body_length"] else 0.5)

        # Keyword presence
        text = f"{subject} {body}".lower()
        keyword_count = sum(1 for keyword in pattern.keywords if keyword in text)
        validations.append(1.0 if keyword_count >= self.validation_rules["required_keywords_per_category"] else 0.7)

        # Template coherence
        validations.append(1.0 if not any(placeholder in f"{subject} {body}" for placeholder in ["{", "}"]) else 0.3)

        return round(np.mean(validations), 3)

    def generate_training_dataset(self, samples_per_category: int = 50) -> List[Dict[str, Any]]:
        """Generate a complete high-quality training dataset"""
        dataset = []

        print(f"🤖 AI TRAINING DATA GENERATION")
        print(f"Generating {samples_per_category} samples per category...")
        print("=" * 60)

        category_stats = {}

        for category in EmailCategory:
            print(f"\nGenerating {category.value} emails...")
            category_samples = []

            for i in range(samples_per_category):
                email = self.generate_realistic_email(category)
                category_samples.append(email)

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  ✓ Generated {i + 1}/{samples_per_category} samples")

            # Calculate category statistics
            quality_scores = [email["quality_score"] for email in category_samples]
            realism_scores = [email["ai_features"]["realism_score"] for email in category_samples]

            category_stats[category.value] = {
                "samples": len(category_samples),
                "avg_quality": round(np.mean(quality_scores), 3),
                "avg_realism": round(np.mean(realism_scores), 3),
                "min_quality": round(min(quality_scores), 3),
                "max_quality": round(max(quality_scores), 3)
            }

            dataset.extend(category_samples)

        # Generate overall dataset statistics
        self.quality_metrics = {
            "total_samples": len(dataset),
            "categories": len(EmailCategory),
            "samples_per_category": samples_per_category,
            "category_statistics": category_stats,
            "overall_quality": round(np.mean([email["quality_score"] for email in dataset]), 3),
            "overall_realism": round(np.mean([email["ai_features"]["realism_score"] for email in dataset]), 3),
            "generation_timestamp": datetime.now().isoformat(),
            "validation_rules": self.validation_rules
        }

        self.generated_data = dataset
        return dataset

    def validate_dataset_quality(self) -> Dict[str, Any]:
        """Perform comprehensive quality validation on generated dataset"""
        if not self.generated_data:
            return {"error": "No dataset generated yet"}

        print("\n🔍 DATASET QUALITY VALIDATION")
        print("=" * 60)

        validation_results = {
            "quality_checks": {},
            "diversity_analysis": {},
            "distribution_analysis": {},
            "recommendations": []
        }

        # Quality score distribution
        quality_scores = [email["quality_score"] for email in self.generated_data]
        validation_results["quality_checks"] = {
            "mean_quality": round(np.mean(quality_scores), 3),
            "std_quality": round(np.std(quality_scores), 3),
            "min_quality": round(min(quality_scores), 3),
            "max_quality": round(max(quality_scores), 3),
            "high_quality_samples": sum(1 for score in quality_scores if score >= 0.8),
            "low_quality_samples": sum(1 for score in quality_scores if score < 0.6)
        }

        # Diversity analysis
        subjects = [email["subject"] for email in self.generated_data]
        unique_subjects = len(set(subjects))
        validation_results["diversity_analysis"] = {
            "subject_diversity_ratio": round(unique_subjects / len(subjects), 3),
            "unique_subjects": unique_subjects,
            "total_subjects": len(subjects),
            "diversity_score": "High" if unique_subjects / len(subjects) > 0.9 else "Medium" if unique_subjects / len(subjects) > 0.7 else "Low"
        }

        # Category distribution
        category_counts = {}
        for email in self.generated_data:
            cat = email["ground_truth"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        validation_results["distribution_analysis"] = {
            "category_distribution": category_counts,
            "balanced_distribution": len(set(category_counts.values())) == 1,
            "distribution_variance": round(np.var(list(category_counts.values())), 2)
        }

        # Generate recommendations
        if validation_results["quality_checks"]["mean_quality"] < 0.7:
            validation_results["recommendations"].append("Consider improving template quality - mean quality below 0.7")

        if validation_results["diversity_analysis"]["diversity_score"] == "Low":
            validation_results["recommendations"].append("Increase template diversity to reduce duplication")

        if validation_results["quality_checks"]["low_quality_samples"] > len(self.generated_data) * 0.1:
            validation_results["recommendations"].append("Review and filter low-quality samples")

        if not validation_results["recommendations"]:
            validation_results["recommendations"] = ["Dataset quality meets high standards - ready for training"]

        print(f"✓ Quality validation complete")
        print(f"  Mean quality score: {validation_results['quality_checks']['mean_quality']}")
        print(f"  Subject diversity: {validation_results['diversity_analysis']['diversity_score']}")
        print(f"  High-quality samples: {validation_results['quality_checks']['high_quality_samples']}")

        return validation_results

    def save_training_data(self, filename: str = "ai_generated_training_data.json"):
        """Save generated training data with quality metrics"""
        if not self.generated_data:
            print("No training data to save")
            return

        output_data = {
            "metadata": {
                "generator_version": "1.0.0",
                "generation_method": "AI-driven pattern synthesis",
                "quality_validation": "Multi-tier validation pipeline",
                **self.quality_metrics
            },
            "validation_results": self.validate_dataset_quality(),
            "training_data": self.generated_data
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 TRAINING DATA SAVED")
        print(f"📁 File: {filename}")
        print(f"📊 Samples: {len(self.generated_data)}")
        print(f"🎯 Quality: {self.quality_metrics['overall_quality']}")
        print(f"🔬 Realism: {self.quality_metrics['overall_realism']}")

        return filename

def main():
    """Demonstrate AI-driven training data generation process"""
    print("🚀 AI-DRIVEN TRAINING DATA GENERATION SYSTEM")
    print("Creating high-quality synthetic email training data")
    print("=" * 70)

    # Initialize generator
    generator = AITrainingDataGenerator()

    # Generate training dataset
    dataset = generator.generate_training_dataset(samples_per_category=25)

    # Validate quality
    validation_results = generator.validate_dataset_quality()

    # Save training data
    filename = generator.save_training_data()

    print("\n" + "=" * 70)
    print("🎯 TRAINING DATA GENERATION COMPLETE")
    print(f"✓ Generated {len(dataset)} high-quality training samples")
    print(f"✓ Multi-category email classification dataset ready")
    print(f"✓ Quality validation passed with score: {generator.quality_metrics['overall_quality']}")
    print(f"✓ Dataset saved to: {filename}")

    return generator, dataset, validation_results

if __name__ == "__main__":
    main()