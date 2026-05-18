from data.preprocess import DataPreprocessor

processor = DataPreprocessor()

# Load
df = processor.load_data("data/raw/superstore.csv")

# Clean
df = processor.clean_data(df)

# Feature engineering
df = processor.feature_engineering(df)

# Save processed data
processor.save_processed(df, "data/processed/superstore_processed.csv")

print("Preprocessing Completed Successfully")