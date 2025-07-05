# from flask import Flask, render_template, request
# import pickle
# import os
# from scripts.feature_extraction import extract_features
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# app = Flask(__name__)

# # Load the model and TF-IDF vectorizer
# model_file_path = os.path.join('models', 'sentiment_model.pkl')
# tfidf_file_path = os.path.join('models', 'tfidf_vectorizer.pkl')

# with open(model_file_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# with open(tfidf_file_path, 'rb') as tfidf_file:
#     tfidf_vectorizer = pickle.load(tfidf_file)

# # Define the route for the homepage
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Define the route for handling prediction requests
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the review from the form input
#         new_review = request.form['review']
        
#         # Preprocess and vectorize the review
#         review_tfidf = tfidf_vectorizer.transform([new_review])
        
#         # Make the prediction
#         sentiment = model.predict(review_tfidf)
        
#         # Map sentiment to readable format
#         sentiment_label = "Positive" if sentiment[0] == 1 else "Negative"
        
#         return render_template('index.html', prediction=sentiment_label, review_text=new_review)

# if __name__ == '__main__':
#     app.run(debug=True)


#new


import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
with open("models/sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form input
        new_review = request.form['review']
        
        # Preprocess and vectorize the review
        review_tfidf = tfidf_vectorizer.transform([new_review])
        
        # Make the prediction
        sentiment = model.predict(review_tfidf)
        
        # Map sentiment to readable format
        sentiment_label = "Positive" if sentiment[0] == 1 else "Negative"
        
        # Render the result in the index.html
        return render_template('index.html', sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
