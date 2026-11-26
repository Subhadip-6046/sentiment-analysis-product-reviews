# Sentiment Analysis of Product Reviews 🛍️

A Machine Learning project that analyzes customer product reviews and classifies them as **Positive** or **Negative**.

🌐 **Live Demo:** [Try the Web App](https://sentiment-analysis-appuct-reviews-meatfjpyh9z3tkdzyzryjr.streamlit.app/)

## About the Project

This project uses Natural Language Processing (NLP) and Machine Learning to understand customer opinions from product reviews.

Users can enter a review and get its predicted sentiment through a simple web application.

## Features

- Positive and Negative sentiment prediction.
- Interactive web interface.
- Text processing using TF-IDF.
- Machine Learning models for classification.
- Simple rule-based correction for selected expressions.

## Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- TF-IDF
- Naive Bayes
- Support Vector Machine (SVM)
- LSTM
- Streamlit
- Google Colab

## Dataset

**Amazon Polarity Dataset**

Initial experiment:
- Training reviews: 10,000
- Testing reviews: 2,000
- Sentiment classes: Positive and Negative

## Model Performance

| Model | Accuracy |
|---|---:|
| Naive Bayes | 85.35% |
| SVM | **87.60%** |
| LSTM | 61.30% |

SVM achieved the highest accuracy and is currently used in the web application.

*Note: These are the original model evaluation results, before the additional rule-based correction.*

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Subhadip-6046/sentiment-analysis-product-reviews.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

## Future Improvements

- Train using a larger dataset.
- Improve prediction accuracy.
- Handle complex sentences and negations better.
- Experiment with advanced NLP models.

## Author

**Subhadip Biswas**

B.Tech – Information Technology  
Jalpaiguri Government Engineering College

GitHub: [Subhadip-6046](https://github.com/Subhadip-6046)
