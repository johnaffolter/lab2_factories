"""
Streamlit UI for Email Classification System
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Email Classification System")
st.markdown("### Factory Pattern ML Classification with Dynamic Topics")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
classification_mode = st.sidebar.radio(
    "Classification Mode",
    ["Topic Similarity", "Email Similarity (if available)"],
    help="Choose how to classify emails"
)
use_email_similarity = classification_mode == "Email Similarity (if available)"

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📨 Classify Email",
    "🏷️ Manage Topics",
    "💾 Stored Emails",
    "🔧 Feature Generators",
    "📊 Analytics"
])

# Helper functions
def make_request(method, endpoint, json_data=None):
    """Make API request and handle errors"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=json_data)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Error: Cannot connect to API. Make sure the server is running on port 8000"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Tab 1: Email Classification
with tab1:
    st.header("Classify New Email")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Email")
        subject = st.text_input("Subject", placeholder="Enter email subject...")
        body = st.text_area("Body", placeholder="Enter email body...", height=200)

        if st.button("🔍 Classify Email", type="primary"):
            if subject or body:
                with st.spinner("Classifying..."):
                    data, error = make_request("POST", "/emails/classify", {
                        "subject": subject,
                        "body": body,
                        "use_email_similarity": use_email_similarity
                    })

                    if data:
                        st.session_state['last_classification'] = data
                    else:
                        st.error(error)
            else:
                st.warning("Please enter either a subject or body text")

    with col2:
        st.subheader("Classification Results")
        if 'last_classification' in st.session_state:
            result = st.session_state['last_classification']

            # Display prediction
            st.success(f"**Predicted Topic:** {result['predicted_topic'].upper()}")

            # Display scores
            st.write("**Topic Scores:**")
            scores_df = pd.DataFrame(
                list(result['topic_scores'].items()),
                columns=['Topic', 'Score']
            ).sort_values('Score', ascending=False)

            # Bar chart of scores
            st.bar_chart(scores_df.set_index('Topic')['Score'])

            # Show features in expander
            with st.expander("View Generated Features"):
                features = result['features']
                for feature_name, value in features.items():
                    if isinstance(value, (int, float)):
                        st.metric(feature_name, f"{value:.2f}" if isinstance(value, float) else value)
                    elif len(str(value)) > 50:
                        st.text_area(feature_name, value, height=100, disabled=True)
                    else:
                        st.text_input(feature_name, value, disabled=True)

# Tab 2: Topic Management
with tab2:
    st.header("Topic Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add New Topic")
        new_topic_name = st.text_input("Topic Name", placeholder="e.g., finance")
        new_topic_desc = st.text_area(
            "Topic Description",
            placeholder="e.g., Financial statements, invoices, and banking emails",
            height=100
        )

        if st.button("➕ Add Topic"):
            if new_topic_name and new_topic_desc:
                data, error = make_request("POST", "/topics", {
                    "topic_name": new_topic_name.lower(),
                    "description": new_topic_desc
                })

                if data:
                    st.success(f"✅ {data['message']}")
                    st.rerun()
                else:
                    st.error(error)
            else:
                st.warning("Please fill in both topic name and description")

    with col2:
        st.subheader("Current Topics")
        data, error = make_request("GET", "/topics")

        if data:
            topics = data['topics']

            # Get full topic info
            pipeline_data, _ = make_request("GET", "/pipeline/info")
            if pipeline_data:
                topics_with_desc = pipeline_data['topics_with_descriptions']

                for topic in topics:
                    with st.container():
                        st.write(f"**{topic.capitalize()}**")
                        if topic in topics_with_desc:
                            st.caption(topics_with_desc[topic])
                        st.divider()
        else:
            st.error(error)

# Tab 3: Email Storage
with tab3:
    st.header("Email Storage")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Store New Email")
        store_subject = st.text_input("Email Subject", key="store_subject")
        store_body = st.text_area("Email Body", height=150, key="store_body")

        # Get available topics for ground truth
        topics_data, _ = make_request("GET", "/topics")
        topics = topics_data['topics'] if topics_data else []

        store_ground_truth = st.selectbox(
            "Ground Truth Label (Optional)",
            ["None"] + topics,
            help="Select the correct classification for training"
        )

        if st.button("💾 Store Email"):
            if store_subject or store_body:
                email_data = {
                    "subject": store_subject,
                    "body": store_body
                }
                if store_ground_truth != "None":
                    email_data["ground_truth"] = store_ground_truth

                data, error = make_request("POST", "/emails", email_data)

                if data:
                    st.success(f"✅ {data['message']} (ID: {data['email_id']})")
                    st.rerun()
                else:
                    st.error(error)
            else:
                st.warning("Please enter either subject or body")

    with col2:
        st.subheader("Stored Emails")
        data, error = make_request("GET", "/emails")

        if data and data['emails']:
            st.info(f"Total stored emails: {data['count']}")

            for email in data['emails']:
                with st.expander(f"Email #{email['id']}: {email.get('subject', 'No subject')[:50]}..."):
                    st.write(f"**Subject:** {email.get('subject', 'N/A')}")
                    st.write(f"**Body:** {email.get('body', 'N/A')}")
                    if 'ground_truth' in email:
                        st.write(f"**Ground Truth:** {email['ground_truth']}")
        elif data:
            st.info("No emails stored yet")
        else:
            st.error(error)

# Tab 4: Feature Generators
with tab4:
    st.header("Feature Generators")
    st.write("View all available feature generators in the system")

    data, error = make_request("GET", "/features")

    if data:
        generators = data['available_generators']

        # Create a DataFrame for better visualization
        gen_data = []
        for gen in generators:
            gen_data.append({
                "Generator Name": gen['name'],
                "Features": ", ".join(gen['features']),
                "Feature Count": len(gen['features'])
            })

        df = pd.DataFrame(gen_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show details in expanders
        st.subheader("Generator Details")
        for gen in generators:
            with st.expander(f"📦 {gen['name'].replace('_', ' ').title()} Generator"):
                st.write(f"**Features Generated:** {len(gen['features'])}")
                for feature in gen['features']:
                    st.write(f"• {feature}")
    else:
        st.error(error)

# Tab 5: Analytics
with tab5:
    st.header("System Analytics")

    col1, col2, col3 = st.columns(3)

    # Get statistics
    topics_data, _ = make_request("GET", "/topics")
    emails_data, _ = make_request("GET", "/emails")
    features_data, _ = make_request("GET", "/features")

    with col1:
        if topics_data:
            st.metric("Total Topics", len(topics_data['topics']))
        else:
            st.metric("Total Topics", "N/A")

    with col2:
        if emails_data:
            st.metric("Stored Emails", emails_data['count'])
        else:
            st.metric("Stored Emails", "0")

    with col3:
        if features_data:
            st.metric("Feature Generators", len(features_data['available_generators']))
        else:
            st.metric("Feature Generators", "N/A")

    # Show classification distribution if emails exist
    if emails_data and emails_data['emails']:
        st.subheader("Email Classification Distribution")

        # Count ground truth labels
        label_counts = {}
        unlabeled = 0
        for email in emails_data['emails']:
            if 'ground_truth' in email:
                label = email['ground_truth']
                label_counts[label] = label_counts.get(label, 0) + 1
            else:
                unlabeled += 1

        if label_counts:
            # Create pie chart
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Pie chart for labeled emails
            ax1.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%')
            ax1.set_title("Labeled Email Distribution")

            # Bar chart including unlabeled
            all_counts = label_counts.copy()
            if unlabeled > 0:
                all_counts['unlabeled'] = unlabeled

            ax2.bar(all_counts.keys(), all_counts.values())
            ax2.set_title("All Emails by Category")
            ax2.set_xlabel("Category")
            ax2.set_ylabel("Count")

            st.pyplot(fig)
        else:
            st.info("No labeled emails available for analytics")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>
    Email Classification System | Factory Pattern Implementation<br>
    API Status: <span id="api-status">🟢 Connected</span> |
    Classification Mode: {mode}
    </small>
</div>
""".format(mode="Email Similarity" if use_email_similarity else "Topic Similarity"),
unsafe_allow_html=True)

# Instructions in sidebar
st.sidebar.divider()
st.sidebar.subheader("📖 Instructions")
st.sidebar.markdown("""
1. **Classify Emails**: Enter subject and body to classify
2. **Add Topics**: Create new classification categories
3. **Store Emails**: Save emails with ground truth for training
4. **View Features**: Explore available feature generators
5. **Analytics**: View system statistics and distributions

**Note**: Make sure the API server is running:
```bash
uvicorn app.main:app --reload
```
""")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()