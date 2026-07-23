"""
ui.py

This module contains reusable Streamlit UI components
for the Spam Detection application.
"""
import streamlit as st

# Page Configuration

# Sidebar
def configure_page():
    """Configure the Streamlit page."""
    st.markdown("""
<style>
    .main h1{font-size:42px;}
    .main h2{font-size:30px;}
    .main h3{font-size:24px;}        
</style>
""", unsafe_allow_html=True)
    st.set_page_config(
        page_title="Spam Detection System",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Header
def display_header():
    """Display the page header."""

    st.title(" 📧 Spam Detection System")

    st.markdown(
        """
The application uses **Natural Language Processing (NLP)** and a
**Machine Learning** to classify SMS messages as eithe:

- ✅ Ham (Safe)
- 🚨 Spam
Simply enter a message below and click **Detect Message**.
"""
    )

# Example Messages

def display_examples():
    with st.expander(" 💡 Example Messages"):
        st.code(
            """Spam Example:
            Congratulations!
            You have won Rs50,000.
            Click here now to claim your reward.

            Ham Example:
            Hi Athira,
            Are we meeting tomorrow at 10 AM? 
            """
        )

# Input Area

def message_input():
    return st.text_area(
        "Enter your message",
        height=180,
        placeholder="Type or paste an SMS message..."
    )

# Detect Button

def detect_button():
    return st.button(
        " 🔍 Detect Message",
        use_container_width=True
    )

# Sidebar
def display_sidebar(best_model, dataset_stats):
    """Display the spplication sidebar."""

    st.sidebar.title(" 📋 Project Details")

    st.sidebar.success(f" 🏆 Best Model\n\n{best_model}")

    st.sidebar.markdown("---")

    st.sidebar.subheader(" 📊 Dataset")

    st.sidebar.metric(
        "Total Messages",
        dataset_stats["Total Messages"]
    )

    st.sidebar.metric(
        "Spam Messages",
        dataset_stats["Spam Messages"]
    )

    st.sidebar.metric(
        "Ham Messages",
        dataset_stats["Ham Messages"]
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader(" ⚙️ Feature Extraction")

    st.sidebar.info("TF-IDF Vectorizer")

    st.sidebar.subheader(" 🤖 Machine Learning")

    st.sidebar.info("Support Vector Machine")

# Dataset Statistics

def display_dataset_statistics(dataset_stats):
    """Display dataset statistics."""

    st.subheader(" 📊 Dataset Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Spam Messages",
            dataset_stats["Spam Messages"]
        )

        st.metric(
            "Spam %",
            f"{dataset_stats['Spam Percentage']}%"
        )

    with col2:
        st.metric(
            "Ham Messages",
            dataset_stats["Ham Messages"],
        )
        st.metric(
                "Ham %",
                f"{dataset_stats['Ham Percentage']}%"
        )
        

# Prediction Results
def display_prediction(result):
    """Display prediction results.
    
    Parameters
    ----------
    result : dict
        Prediction dictionary returned by predict.py
    """

    st.divider()

    st.subheader("Prediction Result")

    #Prediction Label
    if result["label"] == "Spam":
        st.error(" 🚨 This message is classified as SPAM")

    else:
        st.success(" ✅ This message is classified as HAM")
    
    # Confidence
    col1, col2 = st.columns(2)
    with col1:

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Confidence Level",
            result["confidence_level"]
        )
    
    # Confidence Bar

    st.progress(result["confidence"] / 100)

    #Explanation
    st.subheader("💡 Why was this classified?")
    for reason in result["explanation"]:
        st.success(reason)
# Footer
def display_footer():
    st.divider()

    st.caption(
        "Spam Detection System | MCA Major Project | "
        "Built using Python, Scikit-learn and Streamlit"
    )




