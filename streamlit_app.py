
import streamlit as st
import joblib
import os

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# --------------------------------
# LOAD TRAINED MODELS
# --------------------------------

@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return svm_model, tfidf

try:
    svm_model, tfidf = load_models()
except Exception as e:
    st.error(f"Unable to load the ML models: {e}")
    st.stop()


# --------------------------------
# RULE-BASED CORRECTION
# --------------------------------

def improved_prediction(review, svm_prediction):

    text = review.lower()

    negative_context = [
        "disappointed",
        "terrible",
        "awful",
        "regret buying",
        "did not like",
        "do not like",
        "not recommend",
        "waste of money",
        "not worth it",
        "poor quality",
        "waste",
        "horrible"
    ]

    if "not bad" in text:
        if not any(word in text for word in negative_context):
            return 1

    return int(svm_prediction)


# --------------------------------
# CUSTOM PAGE DESIGN
# --------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #2563eb;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------
# WEBSITE CONTENT
# --------------------------------

st.markdown(
    '<div class="title">🛍️ Product Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze customer reviews using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

st.subheader("Enter your product review")

review = st.text_area(
    "Write your review below:",
    placeholder="Example: This product is really good. I love it!",
    height=160
)

analyze = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True
)


# --------------------------------
# SENTIMENT PREDICTION
# --------------------------------

if analyze:

    if not review.strip():
        st.warning("Please enter a review first.")

    else:

        review_vector = tfidf.transform([review])

        svm_prediction = svm_model.predict(review_vector)[0]

        final_prediction = improved_prediction(
            review,
            svm_prediction
        )

        st.divider()
        st.subheader("Analysis Result")

        if final_prediction == 1:

            st.success("😊 Positive Sentiment")
            st.write("This review expresses a positive opinion.")

        else:

            st.error("😞 Negative Sentiment")
            st.write("This review expresses a negative opinion.")


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Powered by TF-IDF and Support Vector Machine (SVM)"
)

st.caption(
    "Academic Project | Sentiment Analysis of Product Reviews"
)
