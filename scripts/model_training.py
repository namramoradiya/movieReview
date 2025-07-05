# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from scripts.feature_extraction import extract_features

# # Load preprocessed data
# data_path = 'data/cleaned_IMDB_reviews.csv'  # Adjust path as needed
# data = pd.read_csv(data_path)

# # Extract features (TF-IDF)
# X = extract_features(data['cleaned_reviews'])

# # Convert sentiment labels to binary values (0 = negative, 1 = positive)
# y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# # Split data into training and testing sets (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predictions and Evaluation
# y_pred = model.predict(X_test)

# # Print Evaluation Metrics
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Classification Report: \n", classification_report(y_test, y_pred))

# # Save the model (optional)
# import joblib
# joblib.dump(model, 'models/sentiment_model.pkl')



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from scripts.feature_extraction import extract_features
# import joblib

# def model_training():
#     # Load preprocessed data
#     data_path = 'data/cleaned_IMDB_reviews.csv'  # Adjust path if necessary
#     data = pd.read_csv(data_path)

#     # Extract features (TF-IDF)
#     X = extract_features(data['cleaned_reviews'])

#     # Convert sentiment labels to binary values (0 = negative, 1 = positive)
#     y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

#     # Split data into training and testing sets (80% training, 20% testing)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train a Logistic Regression model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)

#     # Predictions and Evaluation
#     y_pred = model.predict(X_test)

#     # Print Evaluation Metrics
#     print("Accuracy: ", accuracy_score(y_test, y_pred))
#     print("Classification Report: \n", classification_report(y_test, y_pred))

#     # Save the model (optional)
#     joblib.dump(model, 'models/sentiment_model.pkl')



# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib
# from scripts.preprocessing import preprocess_text

# def model_training():
#     """
#     This function trains the sentiment analysis model and saves the model and vectorizer.
#     """
#     # Load the cleaned dataset
#     data_path = 'data/cleaned_IMDB_reviews.csv'  # Adjust path if necessary
#     data = pd.read_csv(data_path)
    
#     # Preprocess reviews (clean text)
#     data['cleaned_reviews'] = data['review'].apply(preprocess_text)
    
#     # Extract features using TF-IDF vectorizer
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X = vectorizer.fit_transform(data['cleaned_reviews'])
    
#     # Save the vectorizer for later use in prediction
#     joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
#     # Convert sentiment labels to binary (0 = negative, 1 = positive)
#     data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
#     y = data['sentiment']
    
#     # Split the data into training and test sets (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train a Logistic Regression model
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
    
#     # Evaluate the model on the test set
#     y_pred = model.predict(X_test)
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
    
#     # Save the trained model
#     joblib.dump(model, 'models/sentiment_model.pkl')

#     print("Model and vectorizer have been saved successfully!")

# # Call model_training function when this script runs
# if __name__ == "__main__":
#     model_training()


#new 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from feature_extraction import extract_features

# Function to train the model
def train_model():
    # Load and preprocess data
    X, y = extract_features("data/IMDB Dataset.csv")  # Raw data used here

    # Vectorize the data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Train the model
    model = LogisticRegression(max_iter=100)
    model.fit(X_tfidf, y)

    # Save the model and vectorizer
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    print("Model training and saving completed.")

# Entry point to run model training
if __name__ == "__main__":
    train_model()
