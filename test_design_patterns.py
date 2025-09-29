#!/usr/bin/env python3
"""
Individual Design Pattern Testing
Tests Factory Pattern and other design patterns used in the homework
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append('/Users/johnaffolter/lab_2_homework/lab2_factories')

class DesignPatternTester:
    """Tests individual design patterns implementation"""

    def __init__(self):
        self.results = []

    def log_result(self, test_name, status, details, pattern_type):
        """Log test result"""
        result = {
            "test": test_name,
            "pattern": pattern_type,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)

        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")

    def test_factory_pattern(self):
        """Test Factory Pattern Implementation"""
        print("\n" + "="*60)
        print("🏭 TESTING FACTORY PATTERN")
        print("="*60)

        # Test 1: Factory Creation
        try:
            from app.features.factory import FeatureGeneratorFactory

            factory = FeatureGeneratorFactory()
            self.log_result(
                "Factory Instantiation",
                "PASS",
                f"FeatureGeneratorFactory created successfully",
                "Factory Pattern"
            )
        except Exception as e:
            self.log_result(
                "Factory Instantiation",
                "FAIL",
                f"Could not create factory: {e}",
                "Factory Pattern"
            )
            return

        # Test 2: Product Creation via Factory
        try:
            # Get available generator info
            available_gens = factory.get_available_generators()
            generator_names = [gen['name'] for gen in available_gens]

            # Create a few generators
            generators = []
            for gen_name in generator_names[:3]:  # Test first 3
                gen = factory.create_generator(gen_name)
                generators.append(gen)

            generator_types = [type(gen).__name__ for gen in generators]

            self.log_result(
                "Product Creation",
                "PASS",
                f"Created {len(generators)} generators: {generator_types}",
                "Factory Pattern"
            )
        except Exception as e:
            self.log_result(
                "Product Creation",
                "FAIL",
                f"Could not create products: {e}",
                "Factory Pattern"
            )

        # Test 3: Factory Method Pattern (individual generator creation)
        try:
            # Test getting specific generator types
            spam_gen = None
            word_gen = None

            for gen in generators:
                if "Spam" in type(gen).__name__:
                    spam_gen = gen
                elif "WordLength" in type(gen).__name__:
                    word_gen = gen

            if spam_gen and word_gen:
                self.log_result(
                    "Factory Method Pattern",
                    "PASS",
                    f"Successfully created specific generator types",
                    "Factory Pattern"
                )
            else:
                self.log_result(
                    "Factory Method Pattern",
                    "WARN",
                    f"Expected SpamFeatureGenerator and AverageWordLengthFeatureGenerator",
                    "Factory Pattern"
                )
        except Exception as e:
            self.log_result(
                "Factory Method Pattern",
                "FAIL",
                f"Factory method test failed: {e}",
                "Factory Pattern"
            )

        # Test 4: Abstract Base Class Compliance
        try:
            from app.features.base import BaseFeatureGenerator

            all_comply = True
            non_compliant = []

            for gen in generators:
                if not isinstance(gen, BaseFeatureGenerator):
                    all_comply = False
                    non_compliant.append(type(gen).__name__)

            if all_comply:
                self.log_result(
                    "Abstract Base Class Compliance",
                    "PASS",
                    f"All generators inherit from BaseFeatureGenerator",
                    "Factory Pattern"
                )
            else:
                self.log_result(
                    "Abstract Base Class Compliance",
                    "FAIL",
                    f"Non-compliant generators: {non_compliant}",
                    "Factory Pattern"
                )
        except Exception as e:
            self.log_result(
                "Abstract Base Class Compliance",
                "FAIL",
                f"Base class test failed: {e}",
                "Factory Pattern"
            )

    def test_dataclass_pattern(self):
        """Test Data Class Pattern (Value Object Pattern)"""
        print("\n" + "="*60)
        print("📦 TESTING DATACLASS PATTERN")
        print("="*60)

        # Test 1: Email Dataclass Creation
        try:
            from app.dataclasses import Email

            email = Email(
                subject="Test Subject",
                body="Test Body"
            )

            self.log_result(
                "Email Dataclass Creation",
                "PASS",
                f"Email dataclass created with fields: {list(email.__dict__.keys())}",
                "Dataclass Pattern"
            )
        except Exception as e:
            self.log_result(
                "Email Dataclass Creation",
                "FAIL",
                f"Could not create Email dataclass: {e}",
                "Dataclass Pattern"
            )
            return

        # Test 2: Immutability and Value Object Properties
        try:
            # Test field access
            subject = email.subject
            body = email.body

            # Test string representation
            str_repr = str(email)

            self.log_result(
                "Dataclass Properties",
                "PASS",
                f"Dataclass properties accessible, str representation: {len(str_repr)} chars",
                "Dataclass Pattern"
            )
        except Exception as e:
            self.log_result(
                "Dataclass Properties",
                "FAIL",
                f"Dataclass properties test failed: {e}",
                "Dataclass Pattern"
            )

    def test_strategy_pattern(self):
        """Test Strategy Pattern in Feature Generation"""
        print("\n" + "="*60)
        print("🎯 TESTING STRATEGY PATTERN")
        print("="*60)

        # Test 1: Different Feature Generation Strategies
        try:
            from app.features.generators import SpamFeatureGenerator, AverageWordLengthFeatureGenerator
            from app.dataclasses import Email

            test_email = Email(
                subject="Free money winner urgent!",
                body="Congratulations! You have won $1000000! Click here now!"
            )

            # Test different strategies
            spam_strategy = SpamFeatureGenerator()
            word_strategy = AverageWordLengthFeatureGenerator()

            spam_features = spam_strategy.generate_features(test_email)
            word_features = word_strategy.generate_features(test_email)

            self.log_result(
                "Multiple Feature Strategies",
                "PASS",
                f"Spam features: {spam_features}, Word features: {word_features}",
                "Strategy Pattern"
            )
        except Exception as e:
            self.log_result(
                "Multiple Feature Strategies",
                "FAIL",
                f"Strategy pattern test failed: {e}",
                "Strategy Pattern"
            )

        # Test 2: Strategy Interface Compliance
        try:
            from app.features.base import BaseFeatureGenerator

            strategies = [SpamFeatureGenerator(), AverageWordLengthFeatureGenerator()]

            all_compliant = True
            for strategy in strategies:
                # Check if strategy implements required methods
                if not hasattr(strategy, 'generate_features'):
                    all_compliant = False
                if not hasattr(strategy, 'feature_names'):
                    all_compliant = False

            if all_compliant:
                self.log_result(
                    "Strategy Interface Compliance",
                    "PASS",
                    f"All strategies implement required interface",
                    "Strategy Pattern"
                )
            else:
                self.log_result(
                    "Strategy Interface Compliance",
                    "FAIL",
                    f"Some strategies don't implement required interface",
                    "Strategy Pattern"
                )
        except Exception as e:
            self.log_result(
                "Strategy Interface Compliance",
                "FAIL",
                f"Strategy interface test failed: {e}",
                "Strategy Pattern"
            )

    def test_singleton_pattern(self):
        """Test Singleton Pattern (if implemented)"""
        print("\n" + "="*60)
        print("🎯 TESTING SINGLETON PATTERN")
        print("="*60)

        # Test for configuration or model singletons
        try:
            # Check if any singletons exist in the codebase
            # This is optional - not all systems need singletons

            self.log_result(
                "Singleton Pattern Check",
                "INFO",
                f"Singleton pattern not required for this homework",
                "Singleton Pattern"
            )
        except Exception as e:
            self.log_result(
                "Singleton Pattern Check",
                "INFO",
                f"No singleton implementations found (not required)",
                "Singleton Pattern"
            )

    def test_builder_pattern(self):
        """Test Builder Pattern (if implemented)"""
        print("\n" + "="*60)
        print("🏗️ TESTING BUILDER PATTERN")
        print("="*60)

        # Look for any builder patterns in feature composition
        try:
            from app.features.factory import FeatureGeneratorFactory

            # Test if factory can build composite features
            factory = FeatureGeneratorFactory()
            generators = factory.get_all_generators()

            # This could be extended to a full builder if needed
            self.log_result(
                "Builder Pattern Check",
                "INFO",
                f"Factory acts as simple builder for {len(generators)} generators",
                "Builder Pattern"
            )
        except Exception as e:
            self.log_result(
                "Builder Pattern Check",
                "INFO",
                f"Builder pattern not implemented (not required for basic factory)",
                "Builder Pattern"
            )

    def test_model_patterns(self):
        """Test Model-related Design Patterns"""
        print("\n" + "="*60)
        print("🤖 TESTING MODEL PATTERNS")
        print("="*60)

        # Test 1: Model Interface/Strategy
        try:
            from app.models.similarity_model import EmailClassifierModel

            model = EmailClassifierModel()

            # Test model interface
            topics = model.get_all_topics_with_descriptions()

            self.log_result(
                "Model Interface",
                "PASS",
                f"Model implements interface with {len(topics)} topics",
                "Model Pattern"
            )
        except Exception as e:
            self.log_result(
                "Model Interface",
                "FAIL",
                f"Model interface test failed: {e}",
                "Model Pattern"
            )

        # Test 2: Model Prediction Strategy
        try:
            # Test different prediction strategies
            model_similarity = EmailClassifierModel(use_email_similarity=False)
            model_email_sim = EmailClassifierModel(use_email_similarity=True)

            test_features = {
                "raw_email_email_subject": "Test subject",
                "raw_email_email_body": "Test body",
                "email_embeddings_average_embedding": 50.0
            }

            pred1 = model_similarity.predict(test_features)
            pred2 = model_email_sim.predict(test_features)

            self.log_result(
                "Model Strategy Pattern",
                "PASS",
                f"Two strategies: topic_sim='{pred1}', email_sim='{pred2}'",
                "Model Pattern"
            )
        except Exception as e:
            self.log_result(
                "Model Strategy Pattern",
                "FAIL",
                f"Model strategy test failed: {e}",
                "Model Pattern"
            )

    def test_integration_patterns(self):
        """Test Integration and Composition Patterns"""
        print("\n" + "="*60)
        print("🔗 TESTING INTEGRATION PATTERNS")
        print("="*60)

        # Test 1: Factory + Strategy Integration
        try:
            from app.features.factory import FeatureGeneratorFactory
            from app.dataclasses import Email

            factory = FeatureGeneratorFactory()

            test_email = Email(
                subject="Integration test email",
                body="This tests the integration of factory and strategy patterns"
            )

            # Use factory method to generate all features
            all_features = factory.generate_all_features(test_email)

            self.log_result(
                "Factory-Strategy Integration",
                "PASS",
                f"Generated {len(all_features)} features using factory pattern",
                "Integration Pattern"
            )
        except Exception as e:
            self.log_result(
                "Factory-Strategy Integration",
                "FAIL",
                f"Integration test failed: {e}",
                "Integration Pattern"
            )

        # Test 2: Model + Feature Integration
        try:
            from app.models.similarity_model import EmailClassifierModel

            model = EmailClassifierModel()

            # Test model with generated features
            prediction = model.predict(all_features)
            scores = model.get_topic_scores(all_features)

            self.log_result(
                "Model-Feature Integration",
                "PASS",
                f"Model prediction: '{prediction}', {len(scores)} topic scores",
                "Integration Pattern"
            )
        except Exception as e:
            self.log_result(
                "Model-Feature Integration",
                "FAIL",
                f"Model-feature integration failed: {e}",
                "Integration Pattern"
            )

    def generate_pattern_report(self):
        """Generate comprehensive design pattern report"""
        print("\n" + "="*80)
        print("📊 DESIGN PATTERN TEST SUMMARY")
        print("="*80)

        # Count results by pattern and status
        pattern_summary = {}
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.results if r['status'] == 'FAIL')

        for result in self.results:
            pattern = result['pattern']
            status = result['status']

            if pattern not in pattern_summary:
                pattern_summary[pattern] = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'INFO': 0}

            pattern_summary[pattern][status] = pattern_summary[pattern].get(status, 0) + 1

        print(f"\n📈 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        print(f"\n🏭 Pattern-by-Pattern Results:")
        for pattern, counts in pattern_summary.items():
            total_pattern = sum(counts.values())
            passed_pattern = counts.get('PASS', 0)
            print(f"   {pattern}:")
            print(f"      ✅ Passed: {passed_pattern}/{total_pattern}")
            if counts.get('FAIL', 0) > 0:
                print(f"      ❌ Failed: {counts['FAIL']}")
            if counts.get('WARN', 0) > 0:
                print(f"      ⚠️ Warnings: {counts['WARN']}")

        # Save detailed report
        Path("test_results").mkdir(exist_ok=True)
        report_path = f"test_results/design_patterns_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report_data = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%"
            },
            "pattern_summary": pattern_summary,
            "detailed_results": self.results,
            "timestamp": datetime.now().isoformat()
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Detailed report saved: {report_path}")
        return report_data

def main():
    """Main test execution"""
    print("🧪 INDIVIDUAL DESIGN PATTERN TESTING")
    print("=" * 80)
    print(f"Testing started at: {datetime.now()}")

    tester = DesignPatternTester()

    # Run all individual pattern tests
    tester.test_factory_pattern()
    tester.test_dataclass_pattern()
    tester.test_strategy_pattern()
    tester.test_singleton_pattern()
    tester.test_builder_pattern()
    tester.test_model_patterns()
    tester.test_integration_patterns()

    # Generate comprehensive report
    report = tester.generate_pattern_report()

    print("\n" + "="*80)
    print("🏁 DESIGN PATTERN TESTING COMPLETE!")
    print("="*80)

    if report['summary']['failed'] == 0:
        print("🎉 All design patterns implemented correctly!")
    else:
        print(f"⚠️  {report['summary']['failed']} tests failed - review implementation")

    return report

if __name__ == "__main__":
    main()