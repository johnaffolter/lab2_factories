#!/usr/bin/env python3
"""
Comprehensive Screenshot Capture System
Captures screenshots at different viewports and scroll levels for documentation
"""

import os
import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Screenshot configuration
SCREENSHOTS_DIR = "homework_screenshots"
METADATA_FILE = f"{SCREENSHOTS_DIR}/screenshot_metadata.json"

# Viewport configurations
VIEWPORTS = {
    "desktop_full": {"width": 1920, "height": 1080, "name": "Desktop Full HD"},
    "desktop_hd": {"width": 1280, "height": 720, "name": "Desktop HD"},
    "tablet_landscape": {"width": 1024, "height": 768, "name": "Tablet Landscape"},
    "tablet_portrait": {"width": 768, "height": 1024, "name": "Tablet Portrait"},
    "mobile_large": {"width": 414, "height": 896, "name": "Mobile Large (iPhone)"},
}

# Files to capture
CAPTURE_FILES = [
    {
        "file": "HOMEWORK_VISUAL_DEMONSTRATION.html",
        "name": "Visual Demonstration",
        "scroll_positions": [0, 500, 1000, 2000, 3000, 4000],
        "description": "Step-by-step homework demonstration"
    },
    {
        "file": "homework_tracking/visualization.html",
        "name": "Tracking Dashboard",
        "scroll_positions": [0, 500, 1000],
        "description": "Interactive test results dashboard"
    },
    {
        "file": "frontend/graph_visualization.html",
        "name": "Graph Visualization",
        "scroll_positions": [0, 500],
        "description": "Knowledge graph interface"
    }
]

class ScreenshotCapture:
    """Capture screenshots at various viewports and scroll levels"""

    def __init__(self):
        self.screenshots = []
        self.setup_driver()
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    def setup_driver(self):
        """Initialize Chrome WebDriver with options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=chrome_options)
        print("Chrome WebDriver initialized")

    def capture_screenshot(self, file_path, viewport_name, scroll_position):
        """Capture a single screenshot"""

        viewport = VIEWPORTS[viewport_name]

        # Set window size
        self.driver.set_window_size(viewport["width"], viewport["height"])

        # Load file
        full_path = f"file://{os.path.abspath(file_path)}"
        self.driver.get(full_path)

        # Wait for page load
        time.sleep(2)

        # Scroll to position
        self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
        time.sleep(1)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_info = next((f for f in CAPTURE_FILES if f["file"] == file_path), {})
        file_name = file_info.get("name", "unknown").replace(" ", "_")

        screenshot_filename = f"{SCREENSHOTS_DIR}/{file_name}_{viewport_name}_scroll{scroll_position}_{timestamp}.png"

        # Capture screenshot
        self.driver.save_screenshot(screenshot_filename)

        # Get page dimensions
        total_height = self.driver.execute_script("return document.body.scrollHeight")

        screenshot_info = {
            "filename": screenshot_filename,
            "source_file": file_path,
            "file_name": file_info.get("name", "Unknown"),
            "description": file_info.get("description", ""),
            "viewport": viewport_name,
            "viewport_size": f"{viewport['width']}x{viewport['height']}",
            "viewport_description": viewport["name"],
            "scroll_position": scroll_position,
            "total_height": total_height,
            "timestamp": datetime.now().isoformat(),
            "capture_id": len(self.screenshots) + 1
        }

        self.screenshots.append(screenshot_info)

        print(f"  Captured: {viewport['name']} at scroll {scroll_position}px")

        return screenshot_filename

    def capture_all(self):
        """Capture all configured screenshots"""

        print("\n" + "="*80)
        print("COMPREHENSIVE SCREENSHOT CAPTURE")
        print("="*80)

        total_captures = sum(
            len(f["scroll_positions"]) * len(VIEWPORTS)
            for f in CAPTURE_FILES
        )

        print(f"\nPlanned captures: {total_captures}")
        print(f"Files: {len(CAPTURE_FILES)}")
        print(f"Viewports: {len(VIEWPORTS)}")
        print(f"Output directory: {SCREENSHOTS_DIR}/")

        capture_count = 0

        for file_config in CAPTURE_FILES:
            file_path = file_config["file"]

            if not os.path.exists(file_path):
                print(f"\nWarning: File not found - {file_path}")
                continue

            print(f"\n{'='*80}")
            print(f"FILE: {file_config['name']}")
            print(f"Source: {file_path}")
            print(f"Description: {file_config['description']}")
            print(f"{'='*80}")

            for viewport_name in VIEWPORTS:
                print(f"\n  Viewport: {VIEWPORTS[viewport_name]['name']}")

                for scroll_pos in file_config["scroll_positions"]:
                    try:
                        self.capture_screenshot(file_path, viewport_name, scroll_pos)
                        capture_count += 1
                    except Exception as e:
                        print(f"    Error at scroll {scroll_pos}: {e}")

        print(f"\n{'='*80}")
        print(f"CAPTURE COMPLETE")
        print(f"{'='*80}")
        print(f"Total screenshots captured: {capture_count}")
        print(f"Saved to: {SCREENSHOTS_DIR}/")

        # Save metadata
        self.save_metadata()

        # Generate index
        self.generate_index()

    def save_metadata(self):
        """Save screenshot metadata to JSON"""

        metadata = {
            "capture_session": {
                "timestamp": datetime.now().isoformat(),
                "total_screenshots": len(self.screenshots),
                "viewports": VIEWPORTS,
                "files_captured": [f["name"] for f in CAPTURE_FILES]
            },
            "screenshots": self.screenshots
        }

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {METADATA_FILE}")

    def generate_index(self):
        """Generate HTML index of all screenshots"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Homework Screenshots - Comprehensive Documentation</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 10px;
        }}
        .filters {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .filter-group {{
            display: inline-block;
            margin-right: 20px;
        }}
        .filter-group label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        .filter-group select {{
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .screenshot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 30px;
        }}
        .screenshot-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .screenshot-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .screenshot-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .screenshot-info {{
            padding: 20px;
        }}
        .screenshot-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .screenshot-meta {{
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }}
        .viewport-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        .scroll-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-top: 10px;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MLOps Homework 1 - Screenshot Documentation</h1>
        <p>Comprehensive visual documentation at multiple viewports and scroll positions</p>
        <p style="margin-top: 10px; opacity: 0.9;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{len(self.screenshots)}</div>
            <div class="stat-label">Total Screenshots</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(CAPTURE_FILES)}</div>
            <div class="stat-label">Files Documented</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(VIEWPORTS)}</div>
            <div class="stat-label">Viewports</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{sum(len(f['scroll_positions']) for f in CAPTURE_FILES)}</div>
            <div class="stat-label">Scroll Positions</div>
        </div>
    </div>

    <div class="filters">
        <div class="filter-group">
            <label for="fileFilter">File:</label>
            <select id="fileFilter" onchange="filterScreenshots()">
                <option value="all">All Files</option>
                {''.join(f'<option value="{f["name"]}">{f["name"]}</option>' for f in CAPTURE_FILES)}
            </select>
        </div>
        <div class="filter-group">
            <label for="viewportFilter">Viewport:</label>
            <select id="viewportFilter" onchange="filterScreenshots()">
                <option value="all">All Viewports</option>
                {''.join(f'<option value="{name}">{v["name"]}</option>' for name, v in VIEWPORTS.items())}
            </select>
        </div>
    </div>

    <div class="screenshot-grid" id="screenshotGrid">
"""

        for i, screenshot in enumerate(self.screenshots):
            html += f"""
        <div class="screenshot-card" data-file="{screenshot['file_name']}" data-viewport="{screenshot['viewport']}">
            <img src="{os.path.basename(screenshot['filename'])}" alt="{screenshot['file_name']} - {screenshot['viewport_description']}">
            <div class="screenshot-info">
                <div class="screenshot-title">{screenshot['file_name']}</div>
                <div class="screenshot-meta">{screenshot['description']}</div>
                <div class="screenshot-meta">Viewport: {screenshot['viewport_size']}</div>
                <div class="screenshot-meta">Scroll: {screenshot['scroll_position']}px / {screenshot['total_height']}px</div>
                <span class="viewport-badge">{screenshot['viewport_description']}</span>
                <span class="scroll-badge">Scroll: {screenshot['scroll_position']}px</span>
            </div>
        </div>
"""

        html += """
    </div>

    <script>
        function filterScreenshots() {
            const fileFilter = document.getElementById('fileFilter').value;
            const viewportFilter = document.getElementById('viewportFilter').value;
            const cards = document.querySelectorAll('.screenshot-card');

            cards.forEach(card => {
                const fileMatch = fileFilter === 'all' || card.dataset.file === fileFilter;
                const viewportMatch = viewportFilter === 'all' || card.dataset.viewport === viewportFilter;

                if (fileMatch && viewportMatch) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

        index_file = f"{SCREENSHOTS_DIR}/index.html"
        with open(index_file, 'w') as f:
            f.write(html)

        print(f"Index generated: {index_file}")

    def cleanup(self):
        """Close browser and cleanup"""
        if self.driver:
            self.driver.quit()
        print("\nBrowser closed")

def main():
    """Main execution"""

    print("\n" + "="*80)
    print("HOMEWORK SCREENSHOT CAPTURE SYSTEM")
    print("="*80)
    print("\nThis will capture screenshots at multiple viewports and scroll positions")
    print("for comprehensive documentation of the homework demonstration.")
    print("\nViewports configured:")
    for name, config in VIEWPORTS.items():
        print(f"  - {config['name']}: {config['width']}x{config['height']}")

    input("\nPress Enter to start capture...")

    capturer = ScreenshotCapture()

    try:
        capturer.capture_all()
    except Exception as e:
        print(f"\nError during capture: {e}")
    finally:
        capturer.cleanup()

    print("\n" + "="*80)
    print("CAPTURE SESSION COMPLETE")
    print("="*80)
    print(f"\nScreenshots saved to: {SCREENSHOTS_DIR}/")
    print(f"View index at: {SCREENSHOTS_DIR}/index.html")
    print(f"Metadata at: {METADATA_FILE}")

    # Open index
    import subprocess
    subprocess.run(['open', f"{SCREENSHOTS_DIR}/index.html"])

if __name__ == "__main__":
    main()