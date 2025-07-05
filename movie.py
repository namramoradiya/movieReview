# import pandas as pd

# # Load the dataset
# data = pd.read_csv('IMDB Dataset.csv')

# # Display the first few rows
# print(data.head())

# # Check for missing values
# print(data.isnull().sum())

# # Check the distribution of sentiments
# print(data['sentiment'].value_counts())
import pandas as pd
from scripts.preprocessing import preprocess_text
from scripts.feature_extraction import extract_features
from scripts.model_training import model_training

# Load Dataset
data_path = 'data/IMDB Dataset.csv'  # Adjust path as per your setup
data = pd.read_csv(data_path)

# Apply Preprocessing
data['cleaned_reviews'] = data['review'].apply(preprocess_text)

# Save Preprocessed Data (Optional)
data.to_csv('data/cleaned_IMDB_reviews.csv', index=False)
model_training()
print("Model training completed and saved!")

# Check Results
# print(data.head())
