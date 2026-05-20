import pandas as pd
import numpy as np

class DataPreprocessor:

    def __init__(self):
        pass

    def load_data(self, path):
        df = pd.read_csv(path, encoding="latin1")
        return df

    def clean_data(self, df):

        # explicit copy
        df = df.copy()

        # convert dates safely
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')

        df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')

        # remove invalid rows
        df = df.dropna(
        subset=['Order Date']
        ).copy()

        return df

    def feature_engineering(self, df):

        # Time features
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Day'] = df['Order Date'].dt.day
        df['WeekDay'] = df['Order Date'].dt.dayofweek

        # Business feature
        df['Discount'] = np.random.choice([0,0.05,0.1,0.15,0.2,0.25], size=len(df))
        df['Estimated_Profit'] = df['Sales'] * np.random.uniform(0.1, 0.3, len(df))
        df['Profit_Margin'] = df['Estimated_Profit'] / df['Sales']

        return df

    def save_processed(self, df, path):
        df.to_csv(path, index=False)