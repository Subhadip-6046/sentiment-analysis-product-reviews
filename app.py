from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

nb_model = joblib.load("models/naive_bayes.pkl")
svm_model = joblib.load("models/svm.pkl")
tfidf = joblib.load("models/tfidf.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    nb_result = ""
    svm_result = ""
    review = ""

    if request.method == "POST":

        review = request.form["review"]

        review_tfidf = tfidf.transform([review])

        nb_prediction = nb_model.predict(review_tfidf)
        svm_prediction = svm_model.predict(review_tfidf)

        if nb_prediction[0] == 1:
            nb_result = "Positive"
        else:
            nb_result = "Negative"

        if svm_prediction[0] == 1:
            svm_result = "Positive"
        else:
            svm_result = "Negative"

    return render_template(
        "index.html",
        nb_result=nb_result,
        svm_result=svm_result,
        review=review
)



if __name__ == "__main__":
    app.run()
