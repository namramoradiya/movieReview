# # import joblib
# # import pandas as pd
# # from scripts.feature_extraction import extract_features
# # from scripts.preprocessing import preprocess_text

# # def predict_sentiment(review):
# #     """
# #     Predict the sentiment of a new movie review.

# #     Args:
# #         review (str): The movie review text.

# #     Returns:
# #         str: 'positive' or 'negative' based on the prediction.
# #     """
# #     # Load the saved model
# #     model = joblib.load('models/sentiment_model.pkl')

# #     # Preprocess the review
# #     cleaned_review = preprocess_text(review)

# #     # Extract features (TF-IDF)
# #     tfidf_features = extract_features([cleaned_review])  # Make sure it's in list format

# #     # Make prediction
# #     prediction = model.predict(tfidf_features)
    
# #     # Convert prediction to sentiment label
# #     sentiment = 'positive' if prediction == 1 else 'negative'
# #     return sentiment

# # # Example of prediction
# # if __name__ == '__main__':
# #     new_review = "The movie was absolutely amazing! The plot was great and the acting was superb."
# #     print(f"Sentiment: {predict_sentiment(new_review)}")


# import joblib
# import nltk
# from scripts.preprocessing import preprocess_text

# # Download necessary NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# def predict_sentiment(review):
#     """
#     Predicts the sentiment of a given review (positive or negative).
#     """
#     # Load the saved model and vectorizer
#     model = joblib.load('models/sentiment_model.pkl')
#     vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
#     # Preprocess the review (cleaning the text)
#     cleaned_review = preprocess_text(review)
    
#     # Transform the review into the format expected by the model
#     tfidf_features = vectorizer.transform([cleaned_review])
    
#     # Make a prediction using the trained model
#     prediction = model.predict(tfidf_features)
    
#     # Map the prediction to a human-readable sentiment
#     if prediction == 1:
#         return "Positive"
#     else:
#         return "Negative"

# if __name__ == "__main__":
#     # Example review for prediction 
#     new_review = "Movie was good but need some better writing and need some fast paced scened instead of slow and unnecessary scenes else it was good"
    
#     # Print the sentiment prediction
#     print(f"Review: {new_review}")
#     print(f"Sentiment: {predict_sentiment(new_review)}")


# new 

# import pickle
# from scripts.feature_extraction import preprocess_text
# import sys
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load the pre-trained model and TF-IDF vectorizer
# with open("models/sentiment_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("models/tfidf_vectorizer.pkl", "rb") as f:
#     tfidf_vectorizer = pickle.load(f)

# # Function to predict sentiment
# def predict_sentiment(review):
#     review = preprocess_text(review)  # Preprocess the text
#     tfidf_features = tfidf_vectorizer.transform([review])  # Transform the review to TF-IDF features
#     prediction = model.predict(tfidf_features)
#     return "Positive" if prediction == 1 else "Negative"

# # Get review from command line
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         review = sys.argv[1]
#         print(f"Sentiment: {predict_sentiment(review)}")
#     else:
#         print("Please provide a review to predict sentiment.")



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
        return render_template('index.html', prediction=sentiment_label, review_text=new_review)

if __name__ == '__main__':
    app.run(debug=True)
