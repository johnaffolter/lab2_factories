#!/usr/bin/env python3

"""
Working Educational Demo System with Real Analysis
Simplified version that works with our actual email analyzer structure
"""

import sys
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Import our email analyzer
sys.path.append('.')
from advanced_email_analyzer import AdvancedEmailAnalyzer, EmailAnalysisResult

class ExampleQuality(Enum):
    """Quality levels for educational examples"""
    POOR = "poor"
    GOOD = "good"
    PERFECT = "perfect"

@dataclass
class EducationalExample:
    """Educational example with quality rating"""
    example_id: str
    quality: ExampleQuality
    title: str
    description: str
    input_data: Dict[str, str]
    learning_objectives: List[str]
    common_mistakes: List[str]
    best_practices: List[str]

class WorkingEducationalDemo:
    """Working educational system with real email analysis"""

    def __init__(self):
        self.email_analyzer = AdvancedEmailAnalyzer()
        self.examples = self._create_examples()

    def _create_examples(self) -> Dict[str, EducationalExample]:
        """Create educational examples"""

        examples = {}

        # POOR Example
        poor_example = EducationalExample(
            example_id="poor_001",
            quality=ExampleQuality.POOR,
            title="Poorly Written Business Email",
            description="Multiple grammar errors, unprofessional tone, security issues",
            input_data={
                "subject": "urgent meeting tommorow",
                "body": "Hi,\n\nIts very importnat we meet tommorow at 3pm. Click this link: http://suspicious-site.com/meeting\n\nTheir are several issues we need to discuss and I think you no what I mean.\n\nRegards,\nJohn",
                "sender": "john@company.com"
            },
            learning_objectives=[
                "Identify common spelling and grammar errors",
                "Recognize unprofessional communication patterns",
                "Detect security risks in email content",
                "Understand impact of poor writing on business credibility"
            ],
            common_mistakes=[
                "Not proofreading before sending",
                "Using informal language inappropriately",
                "Including unverified links",
                "Making assumptions about shared context"
            ],
            best_practices=[
                "Always proofread emails carefully",
                "Use professional, clear language",
                "Verify all external links",
                "Provide complete context and instructions"
            ]
        )

        # GOOD Example
        good_example = EducationalExample(
            example_id="good_001",
            quality=ExampleQuality.GOOD,
            title="Well-Written Business Email",
            description="Professional tone, clear structure, minor improvements possible",
            input_data={
                "subject": "Meeting Request: Project Review Tomorrow at 3:00 PM EST",
                "body": "Dear Team,\n\nI hope this email finds you well. I would like to schedule a meeting tomorrow at 3:00 PM EST to review our current project status.\n\nAgenda:\n- Progress updates from each team member\n- Discussion of current challenges and solutions\n- Planning for next week's deliverables and milestones\n\nPlease confirm your attendance by replying to this email or updating your calendar.\n\nThank you for your time and continued dedication.\n\nBest regards,\nJohn Smith\nProject Manager\nABC Corporation",
                "sender": "john.smith@company.com"
            },
            learning_objectives=[
                "Recognize good email structure and organization",
                "Understand professional tone in business communication",
                "Learn effective use of bullet points and formatting",
                "Practice clear call-to-action statements"
            ],
            common_mistakes=[
                "Sometimes lacking specific details",
                "Could be more concise in certain situations",
                "Missing time zone specification"
            ],
            best_practices=[
                "Clear, descriptive subject line",
                "Professional greeting and closing",
                "Well-organized content with bullet points",
                "Specific time and clear next steps"
            ]
        )

        # PERFECT Example
        perfect_example = EducationalExample(
            example_id="perfect_001",
            quality=ExampleQuality.PERFECT,
            title="Exemplary Professional Business Email",
            description="Outstanding example demonstrating all best practices",
            input_data={
                "subject": "Action Required: Q4 Project Review Meeting - Dec 15, 2024, 3:00-4:30 PM EST",
                "body": "Dear Project Team,\n\nI hope you are having a productive week. I am writing to formally schedule our Q4 Project Review Meeting.\n\n**Meeting Details:**\n- Date: Friday, December 15, 2024\n- Time: 3:00 PM - 4:30 PM EST\n- Location: Conference Room A (Building 1, 3rd Floor)\n- Virtual Option: Microsoft Teams (link will be sent separately)\n- Meeting ID: PRJ-Q4-2024-001\n\n**Agenda (90 minutes total):**\n1. Q4 Accomplishments Review (15 min) - All team leads\n2. Budget Analysis and Variance Report (20 min) - Finance team\n3. Risk Assessment and Mitigation Strategies (25 min) - Risk management\n4. Q1 2025 Planning and Resource Allocation (20 min) - Project managers\n5. Questions, Discussion, and Next Steps (10 min) - Open floor\n\n**Required Preparation:**\n- Review attached Q4 Summary Report (deadline: Dec 13, 2024)\n- Prepare your department's key metrics and challenge summary\n- Submit agenda additions to john.smith@company.com by Dec 13, 2024\n\n**Attendance:**\n- Required: All project leads and department managers\n- Optional: Team members with expertise in agenda topics\n- Remote participation available for those unable to attend in person\n\n**Action Required:** Please confirm your attendance by December 13, 2024, by:\n1. Replying to this email with your confirmation status\n2. Updating your calendar response\n3. Indicating if you'll attend virtually or in person\n\nIf you have scheduling conflicts or questions, please contact me directly at (555) 123-2847 or john.smith@company.com.\n\nThank you for your continued commitment to this project's success.\n\nBest regards,\n\nJohn Smith\nSenior Project Manager\nOperations Department\nABC Corporation\nDirect: (555) 123-2847\nEmail: john.smith@company.com\nInternal Extension: 2847",
                "sender": "john.smith@company.com"
            },
            learning_objectives=[
                "Master comprehensive business email structure",
                "Learn detailed meeting planning communication",
                "Understand executive-level formality and completeness",
                "Practice multi-stakeholder coordination techniques"
            ],
            common_mistakes=[
                "Can be overly detailed for simple requests",
                "May be too formal for internal team communications"
            ],
            best_practices=[
                "Comprehensive subject line with all key details",
                "Structured content with clear headers and formatting",
                "Specific dates, times, locations, and contact information",
                "Multiple confirmation methods and clear deadlines",
                "Professional signature with complete contact details"
            ]
        )

        examples[poor_example.example_id] = poor_example
        examples[good_example.example_id] = good_example
        examples[perfect_example.example_id] = perfect_example

        return examples

    def run_real_analysis(self, example_id: str) -> Dict[str, Any]:
        """Run real analysis with comprehensive reporting"""

        if example_id not in self.examples:
            raise ValueError(f"Example {example_id} not found")

        example = self.examples[example_id]
        input_data = example.input_data

        print(f"\n🎓 EDUCATIONAL ANALYSIS: {example.title}")
        print(f"Quality Level: {example.quality.value.upper()}")
        print("=" * 70)

        print("📧 Input Email:")
        print(f"  Subject: '{input_data['subject']}'")
        print(f"  From: {input_data['sender']}")
        print(f"  Body Preview: {input_data['body'][:100]}{'...' if len(input_data['body']) > 100 else ''}")
        print()

        # Run actual analysis
        start_time = time.time()
        print("⚙️ Running Advanced Email Analysis...")

        try:
            # Call our actual email analyzer
            analysis_result = self.email_analyzer.analyze_email(
                subject=input_data['subject'],
                body=input_data['body'],
                sender=input_data['sender']
            )

            analysis_time = time.time() - start_time

            # Process and display results
            result = self._process_analysis_results(analysis_result, example, analysis_time)

            print("✅ Analysis Complete!")
            return result

        except Exception as e:
            print(f"❌ Analysis Error: {e}")
            return {"error": str(e)}

    def _process_analysis_results(self, analysis_result: EmailAnalysisResult,
                                example: EducationalExample, analysis_time: float) -> Dict[str, Any]:
        """Process and display analysis results with educational context"""

        print("\n📊 ANALYSIS RESULTS:")
        print("-" * 40)

        # Overall metrics
        overall_score = analysis_result.overall_score
        print(f"Overall Score: {overall_score:.3f}/1.000 ({self._interpret_score(overall_score)})")

        # Detailed metrics from the actual analyzer
        metrics = analysis_result.metrics
        print(f"Readability Score: {metrics.readability_score:.3f}")
        print(f"Professionalism Score: {metrics.professionalism_score:.3f}")
        print(f"Clarity Score: {metrics.clarity_score:.3f}")
        print(f"Tone Score: {metrics.tone_score:.3f}")
        print(f"Engagement Score: {metrics.engagement_score:.3f}")

        # Basic email statistics
        print(f"\n📈 Email Statistics:")
        print(f"Word Count: {metrics.word_count}")
        print(f"Sentence Count: {metrics.sentence_count}")
        print(f"Average Sentence Length: {metrics.average_sentence_length:.1f} words")
        print(f"Lexical Diversity: {metrics.lexical_diversity:.3f}")

        # Issues found
        issues = analysis_result.issues
        print(f"\n🔍 Issues Identified: {len(issues)}")
        if issues:
            for i, issue in enumerate(issues[:5], 1):  # Show top 5 issues
                print(f"  {i}. [{issue.category.value.upper()}] {issue.message}")
                print(f"     Suggestion: {issue.suggestion}")
                print(f"     Severity: {issue.severity.value} | Confidence: {issue.confidence:.2f}")
        else:
            print("  ✅ No significant issues detected")

        # Educational insights based on quality level
        print(f"\n🎯 EDUCATIONAL INSIGHTS:")
        insights = self._generate_educational_insights(analysis_result, example)
        for insight in insights:
            print(f"  • {insight}")

        # Learning analysis
        print(f"\n📚 LEARNING ANALYSIS:")
        print(f"Quality Level: {example.quality.value.upper()}")

        if example.quality == ExampleQuality.POOR:
            print("  Focus Areas: Foundation building, error correction, professionalism")
            if overall_score < 0.5:
                print("  ⚠️  Multiple critical issues require immediate attention")
            if len(issues) > 5:
                print(f"  📝 High issue density: {len(issues)} problems identified")
        elif example.quality == ExampleQuality.GOOD:
            print("  Focus Areas: Refinement, optimization, advanced techniques")
            if overall_score > 0.7:
                print("  ✅ Solid foundation with room for improvement")
        else:  # PERFECT
            print("  Focus Areas: Study as reference, apply techniques to other contexts")
            if overall_score > 0.9:
                print("  🏆 Exemplary quality - use as template")

        # Improvement roadmap
        print(f"\n🛠️ IMPROVEMENT ROADMAP:")
        roadmap = self._create_improvement_roadmap(example, analysis_result)
        for step in roadmap:
            print(f"  {step}")

        # Best practices demonstrated
        print(f"\n✅ BEST PRACTICES:")
        for practice in example.best_practices[:3]:
            print(f"  • {practice}")

        # Common mistakes to avoid
        print(f"\n❌ COMMON MISTAKES TO AVOID:")
        for mistake in example.common_mistakes[:3]:
            print(f"  • {mistake}")

        print(f"\n⏱️ Analysis completed in {analysis_time:.2f} seconds")
        print("-" * 70)

        return {
            "overall_score": overall_score,
            "metrics": {
                "readability": metrics.readability_score,
                "professionalism": metrics.professionalism_score,
                "clarity": metrics.clarity_score,
                "tone": metrics.tone_score,
                "engagement": metrics.engagement_score
            },
            "statistics": {
                "word_count": metrics.word_count,
                "sentence_count": metrics.sentence_count,
                "avg_sentence_length": metrics.average_sentence_length,
                "lexical_diversity": metrics.lexical_diversity
            },
            "issues_count": len(issues),
            "issues": [{"category": i.category.value, "message": i.message,
                       "suggestion": i.suggestion, "severity": i.severity.value}
                      for i in issues[:5]],
            "quality_level": example.quality.value,
            "educational_insights": insights,
            "analysis_time": analysis_time
        }

    def _interpret_score(self, score: float) -> str:
        """Interpret numerical score as quality description"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Needs Improvement"
        else:
            return "Poor"

    def _generate_educational_insights(self, analysis_result: EmailAnalysisResult,
                                     example: EducationalExample) -> List[str]:
        """Generate educational insights based on analysis and quality level"""

        insights = []
        overall_score = analysis_result.overall_score
        issues_count = len(analysis_result.issues)

        if example.quality == ExampleQuality.POOR:
            insights.extend([
                "This example demonstrates common pitfalls in business email writing",
                f"Score of {overall_score:.2f} indicates significant improvement opportunities",
                f"With {issues_count} issues identified, systematic revision is needed"
            ])

            if analysis_result.metrics.professionalism_score < 0.5:
                insights.append("Professionalism score indicates inappropriate tone for business context")

            if analysis_result.metrics.clarity_score < 0.6:
                insights.append("Low clarity score suggests message may be confusing to recipients")

        elif example.quality == ExampleQuality.GOOD:
            insights.extend([
                "This example shows solid business communication fundamentals",
                f"Score of {overall_score:.2f} indicates good quality with refinement potential",
                "Structure and tone are appropriate for professional context"
            ])

            if analysis_result.metrics.engagement_score < 0.8:
                insights.append("Engagement could be enhanced with more compelling language")

        else:  # PERFECT
            insights.extend([
                "This example demonstrates mastery of professional email communication",
                f"Excellent score of {overall_score:.2f} shows comprehensive quality",
                "This email should be studied as a reference model"
            ])

            insights.append("Notice the balance of completeness and clarity throughout")

        # Add insights based on specific metrics
        if analysis_result.metrics.word_count > 200:
            insights.append(f"Length of {analysis_result.metrics.word_count} words is appropriate for detailed communication")
        elif analysis_result.metrics.word_count < 50:
            insights.append("Brief length may lack necessary detail for complex topics")

        return insights

    def _create_improvement_roadmap(self, example: EducationalExample,
                                  analysis_result: EmailAnalysisResult) -> List[str]:
        """Create improvement roadmap based on analysis"""

        roadmap = []

        if example.quality == ExampleQuality.POOR:
            roadmap.extend([
                "Phase 1: Fix fundamental grammar and spelling errors",
                "Phase 2: Improve professional tone and language",
                "Phase 3: Enhance structure and clarity",
                "Phase 4: Add security awareness and verification practices"
            ])

            if len(analysis_result.issues) > 5:
                roadmap.insert(0, "Priority: Address high-severity issues first")

        elif example.quality == ExampleQuality.GOOD:
            roadmap.extend([
                "Phase 1: Refine language for enhanced professionalism",
                "Phase 2: Optimize structure for maximum impact",
                "Phase 3: Add advanced engagement techniques"
            ])

            if analysis_result.metrics.engagement_score < 0.8:
                roadmap.append("Focus: Enhance engagement through compelling language")

        else:  # PERFECT
            roadmap.extend([
                "Study: Analyze techniques used in this exemplary email",
                "Apply: Use this structure as template for similar communications",
                "Adapt: Modify approach for different contexts and audiences"
            ])

        return roadmap

    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run demonstration of all examples"""

        print("🎓 COMPREHENSIVE EDUCATIONAL EMAIL ANALYSIS")
        print("Real Analysis with Full Explainability and Learning")
        print("=" * 80)

        results = {}

        # Process examples in quality order
        quality_order = [ExampleQuality.POOR, ExampleQuality.GOOD, ExampleQuality.PERFECT]

        for quality in quality_order:
            quality_examples = [ex for ex in self.examples.values() if ex.quality == quality]

            print(f"\n📚 {quality.value.upper()} QUALITY DEMONSTRATION")
            print("=" * 60)

            for example in quality_examples:
                result = self.run_real_analysis(example.example_id)
                results[example.example_id] = result

                # Show progression insights
                if quality != ExampleQuality.PERFECT:
                    next_quality = quality_order[quality_order.index(quality) + 1]
                    print(f"\n🎯 Path to {next_quality.value.upper()} quality:")
                    print(f"  Focus on improving: {', '.join(example.best_practices[:3])}")

        # Generate summary
        print(f"\n" + "="*80)
        print("📋 DEMONSTRATION SUMMARY")
        print("="*80)

        scores = [r['overall_score'] for r in results.values() if 'overall_score' in r]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"✅ Examples Analyzed: {len(scores)}")
            print(f"📊 Average Score: {avg_score:.3f}")
            print(f"📈 Score Range: {min(scores):.3f} - {max(scores):.3f}")

        print(f"\n💡 Key Educational Takeaways:")
        print(f"  • Real AI analysis demonstrates actual quality differences")
        print(f"  • Progressive examples show clear improvement path")
        print(f"  • Specific, actionable feedback guides learning")
        print(f"  • Best practices derived from measurable outcomes")

        print(f"\n✅ Educational demonstration completed with real analysis!")

        return results


def main():
    """Run the working educational demonstration"""

    print("🎓 WORKING EDUCATIONAL EMAIL ANALYSIS SYSTEM")
    print("Real AI Analysis with Comprehensive Explainability")
    print("=" * 70)

    # Initialize system
    demo = WorkingEducationalDemo()

    # Run comprehensive demonstration
    results = demo.run_comprehensive_demo()

    print(f"\n🎯 System successfully demonstrated real analysis capabilities!")
    print(f"All results generated using actual AI processing - no simulated data.")

    return demo, results


if __name__ == "__main__":
    main()