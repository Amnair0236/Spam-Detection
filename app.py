"""
app.py

Main Streamlit application for the Spam Detection System.
"""
import pandas as pd

from src.predict import predict_message

from src.utils import (
    get_dataset_statistics,
    load_model_comparison,
    get_best_model
)

from src.ui import (
    configure_page,
    display_sidebar,
    display_header,
    display_examples,
    message_input,
    detect_button,
    display_prediction,
    display_dataset_statistics,
    display_footer
)
# Configure Page
configure_page()

#Load Data
dataset_stats = get_dataset_statistics()
comparison = load_model_comparison()
best_model = get_best_model()

# Sidebar
display_sidebar(best_model, dataset_stats)

# Header
display_header()

import streamlit as st

# Dashboard Metrics
st.subheader("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📩 Messages",
        dataset_stats["Total Messages"]
    )
with col2:
    st.metric(
        "🚨 Spam",
        dataset_stats["Spam Messages"]
    )
with col3:
    st.metric(
        "✅ Ham",
        dataset_stats["Ham Messages"]
    )
with col4:
    best_accuracy = (comparison["Accuracy"].max() * 100)
    st.metric(
        "🏆 Accuracy",
        f"{best_accuracy:.2f}%"
    )



# Examples
display_examples()

# User Input
message = message_input()

# Prediction

if detect_button():
    if message.strip() == "":
        import streamlit as st
        st.warning("Please enter a message.")
    else:
        result = predict_message(message)
        display_prediction(result)

# Model Comparison
import streamlit as st

st.divider()

st.subheader("📊 Model Comparison")

comparison_display = comparison.copy()
numeric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

for column in numeric_columns:
    comparison_display[column] = (
        comparison_display[column] * 100
    ).round(2)

comparison_display["Accuracy"] = comparison_display["Accuracy"].map(lambda x: f"{x:.2f}%")
comparison_display["Precision"] = comparison_display["Precision"].map(lambda x: f"{x:.2f}%")
comparison_display["Recall"] = comparison_display["Recall"].map(lambda x: f"{x:.2f}%")
comparison_display["F1 Score"] = comparison_display["F1 Score"].map(lambda x: f"{x:.2f}%")

st.dataframe(comparison_display, use_container_width=True, hide_index=True)

st.success(f"🏆 Best Performing Model: {best_model}")
        
#Dataset Statistics
display_dataset_statistics(dataset_stats)

st.divider()

st.subheader("⚙️ How the System Works")

st.markdown("""
1. 📝 **Input Message**

↓

2. 🧹 **Text Preprocessing**

↓

3. 📊 **TF-IDF Feature Extraction**

↓

4. 🤖 **Support Vector Machine (SVM)**

↓

5. 📧 **Spam / Ham Prediction**

↓

6. 💡 **Confidence Score & Explanation**
""")

#Footer
display_footer()


