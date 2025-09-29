"""
AI Output Report Generator
Generates comprehensive reports on classification trials with analysis
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from pathlib import Path

class ClassificationReportGenerator:
    """Generate detailed reports on email classification trials"""

    def __init__(self):
        self.trials_file = Path("data/classification_trials.json")
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.trials = self._load_trials()

    def _load_trials(self) -> List[Dict]:
        """Load existing trials from file"""
        if self.trials_file.exists():
            with open(self.trials_file, 'r') as f:
                return json.load(f)
        return []

    def record_trial(self, email_data: Dict, prediction_result: Dict) -> Dict:
        """Record a classification trial with full AI output"""
        trial = {
            "trial_id": len(self.trials) + 1,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "subject": email_data.get("subject"),
                "body": email_data.get("body"),
                "use_learning": email_data.get("use_learning", False)
            },
            "ai_output": {
                "predicted_topic": prediction_result.get("predicted_topic"),
                "confidence_scores": prediction_result.get("topic_scores"),
                "features_extracted": prediction_result.get("features"),
                "processing_time_ms": prediction_result.get("processing_time", 0)
            },
            "model_insights": self._generate_insights(prediction_result)
        }

        self.trials.append(trial)
        self._save_trials()
        return trial

    def _generate_insights(self, result: Dict) -> Dict:
        """Generate AI insights from the prediction"""
        scores = result.get("topic_scores", {})
        features = result.get("features", {})

        # Calculate confidence metrics
        confidence_values = list(scores.values())
        max_confidence = max(confidence_values) if confidence_values else 0
        min_confidence = min(confidence_values) if confidence_values else 0
        avg_confidence = np.mean(confidence_values) if confidence_values else 0

        # Identify top competing topics
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_topics[:3] if len(sorted_topics) >= 3 else sorted_topics

        # Analyze feature contributions
        feature_analysis = self._analyze_features(features)

        return {
            "confidence_analysis": {
                "highest_confidence": max_confidence,
                "lowest_confidence": min_confidence,
                "average_confidence": avg_confidence,
                "confidence_spread": max_confidence - min_confidence,
                "decision_certainty": self._calculate_certainty(scores)
            },
            "top_competing_topics": [
                {"topic": t[0], "score": t[1]} for t in top_3
            ],
            "feature_importance": feature_analysis,
            "recommendation": self._generate_recommendation(scores, features)
        }

    def _analyze_features(self, features: Dict) -> Dict:
        """Analyze feature contributions"""
        analysis = {}

        # Check spam detection
        if features.get("spam_has_spam_words"):
            analysis["spam_detected"] = True
            analysis["spam_impact"] = "High - promotional content detected"
        else:
            analysis["spam_detected"] = False
            analysis["spam_impact"] = "None"

        # Analyze text complexity
        avg_word_length = features.get("word_length_average_word_length", 0)
        if avg_word_length > 6:
            analysis["text_complexity"] = "High - formal/technical language"
        elif avg_word_length > 4.5:
            analysis["text_complexity"] = "Medium - standard business language"
        else:
            analysis["text_complexity"] = "Low - casual language"

        # Non-text character analysis
        non_text_count = features.get("non_text_non_text_char_count", 0)
        analysis["formatting_level"] = "High" if non_text_count > 5 else "Low"

        return analysis

    def _calculate_certainty(self, scores: Dict) -> str:
        """Calculate decision certainty based on score distribution"""
        if not scores:
            return "Unknown"

        values = list(scores.values())
        max_score = max(values)
        second_max = sorted(values, reverse=True)[1] if len(values) > 1 else 0

        margin = max_score - second_max

        if margin > 0.3:
            return "Very High"
        elif margin > 0.2:
            return "High"
        elif margin > 0.1:
            return "Medium"
        else:
            return "Low - Multiple viable classifications"

    def _generate_recommendation(self, scores: Dict, features: Dict) -> str:
        """Generate AI recommendation based on analysis"""
        predicted = max(scores.items(), key=lambda x: x[1])[0] if scores else "unknown"
        confidence = scores.get(predicted, 0) if scores else 0

        if confidence > 0.9:
            return f"High confidence classification as '{predicted}'. No further review needed."
        elif confidence > 0.7:
            return f"Moderate confidence in '{predicted}'. Consider manual review for critical emails."
        else:
            return f"Low confidence classification. Manual review recommended. Consider adding to training data."

    def _save_trials(self):
        """Save trials to file"""
        with open(self.trials_file, 'w') as f:
            json.dump(self.trials, f, indent=2)

    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report of all trials"""
        if not self.trials:
            return "<h1>No trials recorded yet</h1>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Classification Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body class="bg-gray-50 p-8">
            <div class="max-w-7xl mx-auto">
                <h1 class="text-3xl font-bold mb-6">Email Classification Analysis Report</h1>
                <p class="text-gray-600 mb-8">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="text-2xl font-bold text-blue-600">{len(self.trials)}</div>
                        <div class="text-gray-600">Total Trials</div>
                    </div>
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="text-2xl font-bold text-green-600">{self._calculate_accuracy():.1%}</div>
                        <div class="text-gray-600">Average Confidence</div>
                    </div>
                    <div class="bg-white p-6 rounded-lg shadow">
                        <div class="text-2xl font-bold text-purple-600">{self._get_avg_processing_time():.0f}ms</div>
                        <div class="text-gray-600">Avg Processing Time</div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6 mb-8">
                    <h2 class="text-xl font-bold mb-4">Classification Distribution</h2>
                    <canvas id="distributionChart"></canvas>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-bold mb-4">Recent Trials</h2>
                    <div class="overflow-x-auto">
                        <table class="w-full">
                            <thead>
                                <tr class="border-b">
                                    <th class="text-left p-2">Time</th>
                                    <th class="text-left p-2">Subject</th>
                                    <th class="text-left p-2">Predicted</th>
                                    <th class="text-left p-2">Confidence</th>
                                    <th class="text-left p-2">Certainty</th>
                                    <th class="text-left p-2">Time (ms)</th>
                                </tr>
                            </thead>
                            <tbody>
        """

        # Add trial rows
        for trial in self.trials[-10:]:  # Last 10 trials
            timestamp = datetime.fromisoformat(trial["timestamp"]).strftime("%H:%M:%S")
            subject = trial["input"]["subject"][:50] + "..." if len(trial["input"]["subject"]) > 50 else trial["input"]["subject"]
            predicted = trial["ai_output"]["predicted_topic"]
            confidence = trial["ai_output"]["confidence_scores"].get(predicted, 0)
            certainty = trial["model_insights"]["confidence_analysis"]["decision_certainty"]
            processing_time = trial["ai_output"].get("processing_time_ms", 0)

            html += f"""
                                <tr class="border-b hover:bg-gray-50">
                                    <td class="p-2">{timestamp}</td>
                                    <td class="p-2">{subject}</td>
                                    <td class="p-2">
                                        <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
                                            {predicted}
                                        </span>
                                    </td>
                                    <td class="p-2">{confidence:.1%}</td>
                                    <td class="p-2">{certainty}</td>
                                    <td class="p-2">{processing_time}</td>
                                </tr>
            """

        html += """
                            </tbody>
                        </table>
                    </div>
                </div>

                <script>
                    // Create distribution chart
                    const ctx = document.getElementById('distributionChart').getContext('2d');
                    const distribution = """ + json.dumps(self._get_topic_distribution()) + """;

                    new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: Object.keys(distribution),
                            datasets: [{
                                data: Object.values(distribution),
                                backgroundColor: [
                                    'rgba(59, 130, 246, 0.8)',
                                    'rgba(16, 185, 129, 0.8)',
                                    'rgba(147, 51, 234, 0.8)',
                                    'rgba(251, 146, 60, 0.8)',
                                    'rgba(239, 68, 68, 0.8)',
                                    'rgba(99, 102, 241, 0.8)',
                                    'rgba(236, 72, 153, 0.8)',
                                    'rgba(107, 114, 128, 0.8)'
                                ]
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                </script>
            </div>
        </body>
        </html>
        """

        # Save report
        report_file = self.reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html)

        return str(report_file)

    def _calculate_accuracy(self) -> float:
        """Calculate average confidence across all trials"""
        if not self.trials:
            return 0

        confidences = []
        for trial in self.trials:
            predicted = trial["ai_output"]["predicted_topic"]
            score = trial["ai_output"]["confidence_scores"].get(predicted, 0)
            confidences.append(score)

        return np.mean(confidences) if confidences else 0

    def _get_avg_processing_time(self) -> float:
        """Get average processing time"""
        if not self.trials:
            return 0

        times = [t["ai_output"].get("processing_time_ms", 0) for t in self.trials]
        return np.mean(times) if times else 0

    def _get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of predicted topics"""
        distribution = {}
        for trial in self.trials:
            topic = trial["ai_output"]["predicted_topic"]
            distribution[topic] = distribution.get(topic, 0) + 1
        return distribution

    def export_to_csv(self) -> str:
        """Export trials to CSV for analysis"""
        if not self.trials:
            return None

        rows = []
        for trial in self.trials:
            rows.append({
                "trial_id": trial["trial_id"],
                "timestamp": trial["timestamp"],
                "subject": trial["input"]["subject"],
                "body_preview": trial["input"]["body"][:100],
                "use_learning": trial["input"]["use_learning"],
                "predicted_topic": trial["ai_output"]["predicted_topic"],
                "confidence": trial["ai_output"]["confidence_scores"].get(
                    trial["ai_output"]["predicted_topic"], 0
                ),
                "decision_certainty": trial["model_insights"]["confidence_analysis"]["decision_certainty"],
                "processing_time_ms": trial["ai_output"].get("processing_time_ms", 0)
            })

        df = pd.DataFrame(rows)
        csv_file = self.reports_dir / f"trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        return str(csv_file)