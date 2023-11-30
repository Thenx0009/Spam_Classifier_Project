import pickle
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Load stopwords outside the function
stop_words = set(stopwords.words('english'))

def preprocess_text(text, stop_words):
    ps = PorterStemmer()
    # Remove everything apart from A~Z ,a~z
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    # Tokenize using NLTK
    review = word_tokenize(review)
    # Using stemming technique and removing stopwords
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    return review

# Load models
tfidf = pickle.load(open('C:\\Users\\INDIA\\Desktop\\ML_Projects\\sms-spam-classifier\\vectorizer.pkl', 'rb'))
model = pickle.load(open('C:\\Users\\INDIA\\Desktop\\ML_Projects\\sms-spam-classifier\\model.pkl', 'rb'))

# Set page title, favicon, and background color
st.set_page_config(page_title="Spam Classifier App", page_icon="📧", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set app title and description
st.title("Email/SMS Spam Classifier")
st.markdown("This app predicts whether a given message is spam or not.")

# Input for user to enter the message
input_sms = st.text_area("Enter the message", "Type your message here...")

# Button to trigger prediction
if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = preprocess_text(input_sms, stop_words)

    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])

    # Make prediction
    result = model.predict(vector_input)[0]

    # Display the result
    st.subheader("Prediction Result:")
    if result == 1:
        st.error("🚨 This message is classified as **Spam**.")
    else:
        st.success("✅ This message is classified as **Not Spam**.")
