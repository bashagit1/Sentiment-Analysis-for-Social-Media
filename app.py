import os
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

# Set up OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit app setup
st.set_page_config(page_title="Sentiment Analysis Tool", page_icon="📊")

# Sidebar with options
st.sidebar.markdown("<h2>✨ Options Panel</h2>", unsafe_allow_html=True)
analysis_type = st.sidebar.selectbox("📝 Select Analysis Type", ["Text Sentiment", "Social Media Post"])

# Main title
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis for Social Media 🌟</h1>", unsafe_allow_html=True)

# User input for sentiment analysis
st.markdown("<h3>📥 Enter Your Text or Post for Analysis:</h3>", unsafe_allow_html=True)
user_input = st.text_area("💬 Text/Social Media Post", placeholder="Type your text here...")

# Function to analyze sentiment
def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Analyze the sentiment of the following text: '{text}'."}
        ]
    )
    return response.choices[0].message.content.strip()

# Generate sentiment analysis
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        sentiment_result = analyze_sentiment(user_input)
        
        # Simulated sentiment scores for demonstration
        sentiments = ["Positive", "Neutral", "Negative"]
        scores = np.random.rand(3)  # Random scores for illustration
        scores /= scores.sum()  # Normalize to sum to 1 for percentage
        
        st.markdown("<h3>🔍 Sentiment Analysis Result:</h3>", unsafe_allow_html=True)
        st.write(sentiment_result)

        # Plotting the sentiment scores
        plt.figure(figsize=(8, 4))
        plt.bar(sentiments, scores, color=['green', 'gray', 'red'])
        plt.title('Sentiment Distribution')
        plt.ylabel('Percentage')
        plt.ylim(0, 1)
        st.pyplot(plt)

    else:
        st.warning("Please enter text to analyze.")

# User guide
st.info(""" 
### How to Use:
1. **Enter your text** in the provided area.
2. **Select the analysis type** if necessary.
3. **Click Analyze Sentiment** to get your analysis results.
""")
