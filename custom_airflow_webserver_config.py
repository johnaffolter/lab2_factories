# Custom Airflow webserver configuration
from airflow.www.app import create_app
import os

# Increase Flask/Werkzeug limits
os.environ['WERKZEUG_MAX_CONTENT_LENGTH'] = str(100 * 1024 * 1024)  # 100MB
os.environ['WERKZEUG_MAX_FORM_MEMORY_SIZE'] = str(10 * 1024 * 1024)  # 10MB

app = create_app(testing=False)

# Configure app for large headers
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['MAX_COOKIE_SIZE'] = 16384  # 16KB

if __name__ == "__main__":
    app.run()
