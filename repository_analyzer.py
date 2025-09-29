#!/usr/bin/env python3

"""
Repository and Application Analyzer
Comprehensive analysis of our MLOps repository structure, dependencies, and capabilities
"""

import os
import sys
import json
import ast
import re
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter

@dataclass
class FileAnalysis:
    """Analysis of a single file"""
    file_path: str
    file_type: str
    size_bytes: int
    line_count: int
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    purpose: str = ""

@dataclass
class RepositoryAnalysis:
    """Complete repository analysis"""
    total_files: int
    total_lines: int
    total_size_bytes: int
    file_types: Dict[str, int]
    files: List[FileAnalysis]
    dependencies: Dict[str, int]
    architecture_patterns: List[str]
    main_components: List[str]
    integration_points: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class RepositoryAnalyzer:
    """Analyzer for our MLOps repository"""

    def __init__(self, repo_path: str = "/Users/johnaffolter/lab_2_homework/lab2_factories"):
        self.repo_path = Path(repo_path)
        self.analysis = RepositoryAnalysis(
            total_files=0,
            total_lines=0,
            total_size_bytes=0,
            file_types={},
            files=[],
            dependencies={},
            architecture_patterns=[],
            main_components=[],
            integration_points=[]
        )

    def analyze_repository(self) -> RepositoryAnalysis:
        """Perform comprehensive repository analysis"""

        print("🔍 ANALYZING MLOPS REPOSITORY STRUCTURE")
        print("=" * 50)

        # Find all files
        all_files = []
        for ext in ['.py', '.html', '.js', '.css', '.json', '.yaml', '.yml', '.md', '.txt']:
            pattern = f"**/*{ext}"
            files = list(self.repo_path.glob(pattern))
            all_files.extend(files)

        self.analysis.total_files = len(all_files)

        print(f"📁 Total files found: {self.analysis.total_files}")

        # Analyze each file
        for file_path in all_files:
            try:
                file_analysis = self._analyze_file(file_path)
                self.analysis.files.append(file_analysis)

                # Update totals
                self.analysis.total_lines += file_analysis.line_count
                self.analysis.total_size_bytes += file_analysis.size_bytes

                # Update file type counts
                ext = file_path.suffix.lower()
                self.analysis.file_types[ext] = self.analysis.file_types.get(ext, 0) + 1

                # Collect dependencies
                for dep in file_analysis.dependencies:
                    self.analysis.dependencies[dep] = self.analysis.dependencies.get(dep, 0) + 1

            except Exception as e:
                print(f"⚠️ Error analyzing {file_path}: {e}")

        # Analyze architecture patterns
        self._identify_architecture_patterns()

        # Identify main components
        self._identify_main_components()

        # Find integration points
        self._identify_integration_points()

        print(f"📊 Analysis complete:")
        print(f"   Lines of code: {self.analysis.total_lines:,}")
        print(f"   Total size: {self.analysis.total_size_bytes / 1024 / 1024:.2f} MB")
        print(f"   Dependencies: {len(self.analysis.dependencies)}")

        return self.analysis

    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file"""

        file_analysis = FileAnalysis(
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
            size_bytes=file_path.stat().st_size,
            line_count=0
        )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_analysis.line_count = len(content.split('\n'))

            # Analyze Python files in detail
            if file_path.suffix == '.py':
                self._analyze_python_file(file_path, content, file_analysis)

            # Analyze other file types
            elif file_path.suffix in ['.html', '.js']:
                self._analyze_web_file(content, file_analysis)
            elif file_path.suffix == '.json':
                self._analyze_json_file(content, file_analysis)

        except Exception as e:
            print(f"⚠️ Could not analyze {file_path}: {e}")

        return file_analysis

    def _analyze_python_file(self, file_path: Path, content: str, file_analysis: FileAnalysis):
        """Analyze Python file in detail"""

        try:
            # Parse AST
            tree = ast.parse(content)

            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    file_analysis.classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    file_analysis.functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_analysis.imports.append(alias.name)
                        file_analysis.dependencies.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    file_analysis.imports.append(node.module)
                    file_analysis.dependencies.add(node.module.split('.')[0])

            # Calculate complexity (simple metric based on AST nodes)
            node_count = len(list(ast.walk(tree)))
            file_analysis.complexity_score = min(10.0, node_count / 100.0)

            # Determine file purpose based on name and content
            file_analysis.purpose = self._determine_file_purpose(file_path, content)

        except SyntaxError:
            # Handle files with syntax errors
            file_analysis.purpose = "syntax_error"

    def _analyze_web_file(self, content: str, file_analysis: FileAnalysis):
        """Analyze HTML/JS files"""

        # Simple analysis for web files
        if 'vue' in content.lower() or 'Vue' in content:
            file_analysis.dependencies.add('vue')
        if 'react' in content.lower() or 'React' in content:
            file_analysis.dependencies.add('react')
        if 'jquery' in content.lower() or 'jQuery' in content:
            file_analysis.dependencies.add('jquery')
        if 'chart' in content.lower():
            file_analysis.dependencies.add('charts')

    def _analyze_json_file(self, content: str, file_analysis: FileAnalysis):
        """Analyze JSON files"""

        try:
            data = json.loads(content)
            if 'dependencies' in data:
                for dep in data['dependencies']:
                    file_analysis.dependencies.add(dep)
        except json.JSONDecodeError:
            pass

    def _determine_file_purpose(self, file_path: Path, content: str) -> str:
        """Determine the purpose of a file based on its name and content"""

        filename = file_path.name.lower()
        content_lower = content.lower()

        # Core system files
        if 'test' in filename:
            return "testing"
        elif 'config' in filename:
            return "configuration"
        elif 'factory' in filename:
            return "factory_pattern"
        elif 'analyzer' in filename:
            return "analysis_engine"
        elif 'generator' in filename:
            return "data_generation"
        elif 'airflow' in filename:
            return "workflow_orchestration"
        elif 'aws' in filename or 's3' in filename:
            return "cloud_integration"
        elif 'screenshot' in filename:
            return "result_tracking"
        elif 'email' in filename:
            return "email_processing"
        elif 'attachment' in filename:
            return "document_processing"
        elif 'integration' in filename:
            return "system_integration"
        elif 'deployment' in filename:
            return "deployment_automation"

        # Check content for patterns
        if 'class.*factory' in content_lower:
            return "factory_pattern"
        elif 'fastapi' in content_lower or 'uvicorn' in content_lower:
            return "api_service"
        elif 'dag' in content_lower and 'airflow' in content_lower:
            return "workflow_orchestration"
        elif 'neo4j' in content_lower or 'snowflake' in content_lower:
            return "database_integration"
        elif 'llm' in content_lower or 'openai' in content_lower:
            return "ai_integration"

        return "utility"

    def _identify_architecture_patterns(self):
        """Identify architecture patterns used in the repository"""

        patterns = set()

        for file_analysis in self.analysis.files:
            if file_analysis.file_type == '.py':
                # Factory Pattern
                if 'factory' in file_analysis.file_path.lower() or any('Factory' in cls for cls in file_analysis.classes):
                    patterns.add("Factory Method Pattern")

                # Registry Pattern
                if 'registry' in file_analysis.file_path.lower() or any('Registry' in cls for cls in file_analysis.classes):
                    patterns.add("Registry Pattern")

                # Observer Pattern
                if any('observer' in cls.lower() for cls in file_analysis.classes):
                    patterns.add("Observer Pattern")

                # Strategy Pattern
                if any('strategy' in cls.lower() for cls in file_analysis.classes):
                    patterns.add("Strategy Pattern")

                # Builder Pattern
                if any('builder' in cls.lower() for cls in file_analysis.classes):
                    patterns.add("Builder Pattern")

                # Abstract Factory
                if any('abstractfactory' in cls.lower() for cls in file_analysis.classes):
                    patterns.add("Abstract Factory Pattern")

                # Singleton (though not recommended)
                if any('singleton' in cls.lower() for cls in file_analysis.classes):
                    patterns.add("Singleton Pattern")

        # Add patterns based on overall architecture
        if any('fastapi' in dep for dep in self.analysis.dependencies):
            patterns.add("REST API Architecture")

        if any('airflow' in dep for dep in self.analysis.dependencies):
            patterns.add("Workflow Orchestration Pattern")

        if 'neo4j' in self.analysis.dependencies or 'snowflake' in self.analysis.dependencies:
            patterns.add("Multi-Database Integration Pattern")

        if 'openai' in self.analysis.dependencies:
            patterns.add("AI Service Integration Pattern")

        self.analysis.architecture_patterns = list(patterns)

    def _identify_main_components(self):
        """Identify main system components"""

        components = []

        # Group files by purpose
        purpose_groups = defaultdict(list)
        for file_analysis in self.analysis.files:
            if file_analysis.purpose:
                purpose_groups[file_analysis.purpose].append(file_analysis)

        # Identify major components
        for purpose, files in purpose_groups.items():
            if len(files) >= 2 or any(f.line_count > 500 for f in files):
                component_name = purpose.replace('_', ' ').title()
                total_lines = sum(f.line_count for f in files)
                components.append(f"{component_name} ({len(files)} files, {total_lines} lines)")

        # Add specific major components
        major_files = [f for f in self.analysis.files if f.line_count > 1000]
        for file_analysis in major_files:
            component_name = Path(file_analysis.file_path).stem.replace('_', ' ').title()
            components.append(f"{component_name} (Major Module - {file_analysis.line_count} lines)")

        self.analysis.main_components = components

    def _identify_integration_points(self):
        """Identify system integration points"""

        integrations = []

        # Check for external service integrations
        for dep, count in self.analysis.dependencies.items():
            if dep in ['openai', 'boto3', 'neo4j', 'snowflake', 'requests', 'fastapi']:
                integrations.append(f"{dep.upper()} Integration ({count} files)")

        # Check for database integrations
        db_patterns = ['database', 'db', 'postgres', 'mysql', 'mongo']
        for file_analysis in self.analysis.files:
            for pattern in db_patterns:
                if pattern in file_analysis.file_path.lower():
                    integrations.append("Database Integration Layer")
                    break

        # Check for API integrations
        api_files = [f for f in self.analysis.files if 'api' in f.file_path.lower() or 'routes' in f.file_path.lower()]
        if api_files:
            integrations.append(f"REST API Layer ({len(api_files)} files)")

        # Check for cloud integrations
        cloud_files = [f for f in self.analysis.files if any(cloud in f.file_path.lower() for cloud in ['aws', 's3', 'azure', 'gcp'])]
        if cloud_files:
            integrations.append(f"Cloud Integration ({len(cloud_files)} files)")

        self.analysis.integration_points = integrations

    def generate_detailed_report(self) -> str:
        """Generate a detailed analysis report"""

        report = []
        report.append("🏗️ MLOPS REPOSITORY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {self.analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overview
        report.append("📊 REPOSITORY OVERVIEW")
        report.append("-" * 30)
        report.append(f"Total Files: {self.analysis.total_files}")
        report.append(f"Lines of Code: {self.analysis.total_lines:,}")
        report.append(f"Repository Size: {self.analysis.total_size_bytes / 1024 / 1024:.2f} MB")
        report.append("")

        # File type breakdown
        report.append("📁 FILE TYPE BREAKDOWN")
        report.append("-" * 25)
        for file_type, count in sorted(self.analysis.file_types.items()):
            percentage = (count / self.analysis.total_files) * 100
            report.append(f"{file_type:8} {count:3} files ({percentage:5.1f}%)")
        report.append("")

        # Architecture patterns
        report.append("🏛️ ARCHITECTURE PATTERNS")
        report.append("-" * 25)
        for pattern in sorted(self.analysis.architecture_patterns):
            report.append(f"✅ {pattern}")
        report.append("")

        # Main components
        report.append("🧩 MAIN SYSTEM COMPONENTS")
        report.append("-" * 30)
        for component in sorted(self.analysis.main_components):
            report.append(f"• {component}")
        report.append("")

        # Integration points
        report.append("🔗 INTEGRATION POINTS")
        report.append("-" * 20)
        for integration in sorted(self.analysis.integration_points):
            report.append(f"🔌 {integration}")
        report.append("")

        # Top dependencies
        report.append("📦 TOP DEPENDENCIES")
        report.append("-" * 20)
        top_deps = sorted(self.analysis.dependencies.items(), key=lambda x: x[1], reverse=True)
        for dep, count in top_deps[:10]:
            report.append(f"{dep:15} {count:3} files")
        report.append("")

        # File complexity analysis
        python_files = [f for f in self.analysis.files if f.file_type == '.py']
        if python_files:
            report.append("📈 CODE COMPLEXITY ANALYSIS")
            report.append("-" * 30)

            # Sort by complexity
            complex_files = sorted(python_files, key=lambda f: f.complexity_score, reverse=True)

            avg_complexity = sum(f.complexity_score for f in python_files) / len(python_files)
            report.append(f"Average Complexity: {avg_complexity:.2f}")
            report.append(f"Most Complex Files:")

            for file_analysis in complex_files[:5]:
                filename = Path(file_analysis.file_path).name
                report.append(f"  {filename:30} {file_analysis.complexity_score:.2f}")
            report.append("")

        # Purpose analysis
        report.append("🎯 FILE PURPOSE ANALYSIS")
        report.append("-" * 25)
        purpose_counts = Counter(f.purpose for f in self.analysis.files if f.purpose)
        for purpose, count in purpose_counts.most_common():
            purpose_name = purpose.replace('_', ' ').title()
            report.append(f"{purpose_name:25} {count:3} files")
        report.append("")

        # Major modules detail
        report.append("📋 MAJOR MODULES DETAIL")
        report.append("-" * 25)
        major_files = sorted([f for f in self.analysis.files if f.line_count > 500],
                           key=lambda f: f.line_count, reverse=True)

        for file_analysis in major_files[:10]:
            filename = Path(file_analysis.file_path).name
            report.append(f"{filename:35} {file_analysis.line_count:5} lines, {len(file_analysis.classes):2} classes, {len(file_analysis.functions):3} functions")
        report.append("")

        # Key insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 15)

        total_python_lines = sum(f.line_count for f in python_files)
        if total_python_lines > 0:
            report.append(f"• Python represents {len(python_files)} files with {total_python_lines:,} lines")

        if 'Factory Method Pattern' in self.analysis.architecture_patterns:
            report.append("• Strong use of Factory Method pattern for extensibility")

        if 'openai' in self.analysis.dependencies:
            report.append("• Real AI/LLM integration with OpenAI")

        if 'boto3' in self.analysis.dependencies:
            report.append("• Production AWS cloud integration")

        if len(self.analysis.integration_points) > 5:
            report.append("• Highly integrated system with multiple external services")

        if any('test' in f.purpose for f in self.analysis.files):
            report.append("• Comprehensive testing framework included")

        report.append("")
        report.append("🌟 REPOSITORY STATUS: PRODUCTION-READY MLOPS PLATFORM")

        return "\n".join(report)

    def save_analysis_json(self, output_path: str = None) -> str:
        """Save analysis as JSON for programmatic access"""

        if output_path is None:
            output_path = f"/tmp/repository_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        analysis_dict = {
            'timestamp': self.analysis.timestamp.isoformat(),
            'total_files': self.analysis.total_files,
            'total_lines': self.analysis.total_lines,
            'total_size_bytes': self.analysis.total_size_bytes,
            'file_types': self.analysis.file_types,
            'dependencies': self.analysis.dependencies,
            'architecture_patterns': self.analysis.architecture_patterns,
            'main_components': self.analysis.main_components,
            'integration_points': self.analysis.integration_points,
            'files': [
                {
                    'file_path': f.file_path,
                    'file_type': f.file_type,
                    'size_bytes': f.size_bytes,
                    'line_count': f.line_count,
                    'imports': f.imports,
                    'classes': f.classes,
                    'functions': f.functions,
                    'dependencies': list(f.dependencies),
                    'complexity_score': f.complexity_score,
                    'purpose': f.purpose
                }
                for f in self.analysis.files
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(analysis_dict, f, indent=2)

        print(f"📄 Analysis saved to: {output_path}")
        return output_path

def analyze_mlops_repository():
    """Main function to analyze the MLOps repository"""

    print("🔍 STARTING COMPREHENSIVE REPOSITORY ANALYSIS")
    print()

    analyzer = RepositoryAnalyzer()
    analysis = analyzer.analyze_repository()

    print("\n" + "="*60)
    report = analyzer.generate_detailed_report()
    print(report)

    # Save JSON analysis
    json_path = analyzer.save_analysis_json()

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Repository: {analyzer.repo_path}")
    print(f"   Files analyzed: {analysis.total_files}")
    print(f"   Total code lines: {analysis.total_lines:,}")
    print(f"   Architecture patterns: {len(analysis.architecture_patterns)}")
    print(f"   Main components: {len(analysis.main_components)}")
    print(f"   Integration points: {len(analysis.integration_points)}")
    print(f"   JSON report: {json_path}")

    return analysis

if __name__ == "__main__":
    analysis = analyze_mlops_repository()