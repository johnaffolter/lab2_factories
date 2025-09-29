"""
LLM-as-a-Judge: Classification Validation System
Uses OpenAI GPT to validate email classifications and provide quality scores
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Check if OpenAI is available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


class LLMJudge:
    """LLM-based judge for validating email classifications"""

    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)

    def validate_classification(
        self,
        email_subject: str,
        email_body: str,
        predicted_label: str,
        ground_truth_label: Optional[str] = None,
        available_topics: List[str] = None
    ) -> Dict:
        """
        Use LLM to validate email classification

        Returns:
            {
                "is_correct": bool,
                "confidence": float,
                "reasoning": str,
                "suggested_label": str,
                "quality_score": float
            }
        """
        if not self.client:
            return self._mock_validation(predicted_label, ground_truth_label)

        # Build prompt for LLM judge
        topics_list = ", ".join(available_topics) if available_topics else "work, personal, promotion, newsletter, support, travel, education, health, finance, shopping, social, entertainment"

        prompt = f"""You are an expert email classifier. Evaluate the following classification.

Email Subject: {email_subject}
Email Body: {email_body}

Predicted Classification: {predicted_label}
{f'Ground Truth: {ground_truth_label}' if ground_truth_label else ''}

Available Topics: {topics_list}

Your task:
1. Determine if the predicted classification is correct
2. Provide a confidence score (0.0 to 1.0)
3. Explain your reasoning
4. Suggest the best classification if prediction is wrong
5. Rate the overall quality of this classification (0.0 to 1.0)

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "suggested_label": "best topic",
    "quality_score": 0.0-1.0
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert email classification validator. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse LLM response
            result_text = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            # Add metadata
            result["judge_model"] = "gpt-4"
            result["timestamp"] = datetime.now().isoformat()
            result["ground_truth_match"] = result["is_correct"] if ground_truth_label else None

            return result

        except Exception as e:
            print(f"LLM Judge error: {str(e)}")
            return self._mock_validation(predicted_label, ground_truth_label)

    def _mock_validation(self, predicted_label: str, ground_truth_label: Optional[str]) -> Dict:
        """Fallback validation when LLM is not available"""
        if ground_truth_label:
            is_correct = predicted_label == ground_truth_label
            return {
                "is_correct": is_correct,
                "confidence": 0.85 if is_correct else 0.65,
                "reasoning": f"Mock validation: Predicted '{predicted_label}' vs Ground Truth '{ground_truth_label}'",
                "suggested_label": ground_truth_label if not is_correct else predicted_label,
                "quality_score": 0.8 if is_correct else 0.5,
                "judge_model": "mock",
                "timestamp": datetime.now().isoformat(),
                "ground_truth_match": is_correct
            }
        else:
            return {
                "is_correct": True,
                "confidence": 0.75,
                "reasoning": "Mock validation: No ground truth available",
                "suggested_label": predicted_label,
                "quality_score": 0.7,
                "judge_model": "mock",
                "timestamp": datetime.now().isoformat(),
                "ground_truth_match": None
            }

    def batch_validate(self, emails: List[Dict]) -> Dict:
        """
        Validate multiple email classifications

        Args:
            emails: List of {subject, body, predicted_label, ground_truth_label}

        Returns:
            Summary statistics and detailed results
        """
        results = []
        correct_count = 0
        total_quality = 0.0

        for email in emails:
            validation = self.validate_classification(
                email["subject"],
                email["body"],
                email["predicted_label"],
                email.get("ground_truth_label")
            )

            results.append({
                "email_id": email.get("id", "unknown"),
                "subject": email["subject"][:50],
                "validation": validation
            })

            if validation["is_correct"]:
                correct_count += 1
            total_quality += validation["quality_score"]

        summary = {
            "total_evaluated": len(emails),
            "correct": correct_count,
            "incorrect": len(emails) - correct_count,
            "accuracy": correct_count / len(emails) if emails else 0,
            "average_quality": total_quality / len(emails) if emails else 0,
            "average_confidence": sum(r["validation"]["confidence"] for r in results) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "summary": summary,
            "results": results
        }


def test_llm_judge():
    """Test the LLM judge system"""
    print("LLM-as-a-Judge Validation System Test")
    print("=" * 70)
    print()

    judge = LLMJudge()

    if not judge.client:
        print("⚠️ OpenAI not configured - using mock validation")
    else:
        print("✅ OpenAI configured - using GPT-4 validation")

    print()

    # Test cases
    test_emails = [
        {
            "id": "test_1",
            "subject": "Team Meeting Tomorrow at 2pm",
            "body": "Please join the weekly standup to discuss project progress and blockers.",
            "predicted_label": "work",
            "ground_truth_label": "work"
        },
        {
            "id": "test_2",
            "subject": "50% OFF Everything Today Only!",
            "body": "Flash sale! Use code SAVE50 at checkout. Limited time offer ends tonight.",
            "predicted_label": "promotion",
            "ground_truth_label": "promotion"
        },
        {
            "id": "test_3",
            "subject": "Doctor Appointment Reminder",
            "body": "Your appointment with Dr. Smith is scheduled for Friday at 10am.",
            "predicted_label": "personal",
            "ground_truth_label": "health"
        }
    ]

    # Validate batch
    results = judge.batch_validate(test_emails)

    print("Validation Results:")
    print("-" * 70)
    print(f"Total Evaluated: {results['summary']['total_evaluated']}")
    print(f"Correct: {results['summary']['correct']}")
    print(f"Incorrect: {results['summary']['incorrect']}")
    print(f"Accuracy: {results['summary']['accuracy']*100:.1f}%")
    print(f"Average Quality: {results['summary']['average_quality']:.2f}")
    print(f"Average Confidence: {results['summary']['average_confidence']:.2f}")
    print()

    print("Individual Results:")
    print("-" * 70)
    for i, result in enumerate(results["results"], 1):
        val = result["validation"]
        print(f"{i}. {result['subject']}")
        print(f"   Correct: {val['is_correct']} | Quality: {val['quality_score']:.2f} | Confidence: {val['confidence']:.2f}")
        print(f"   Reasoning: {val['reasoning'][:80]}...")
        print()

    print("✅ LLM Judge test complete!")


if __name__ == "__main__":
    test_llm_judge()