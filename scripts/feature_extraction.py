# from sklearn.feature_extraction.text import TfidfVectorizer

# def extract_features(data):
#     """
#     Convert the cleaned text reviews into numerical features using TF-IDF.

#     Args:
#         data (pd.Series): A pandas Series containing the cleaned reviews.

#     Returns:
#         tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.
#     """
#     vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
#     tfidf_matrix = vectorizer.fit_transform(data)
#     return tfidf_matrix



#new
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to clean the text (optional, depending on your preprocessing steps)
def clean_text(text):
    # Add text cleaning steps (remove HTML tags, stopwords, etc.)
    return text

# Function to load and preprocess data
def extract_features(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Clean the text column (optional)
    data['review'] = data['review'].apply(clean_text)
    
    # Extract features (X) and labels (y)
    X = data['review']
    y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert sentiment to binary
    
    return X, y
