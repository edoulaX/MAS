import pandas as pd
import numpy as np

# Read the dataset (you'll need to specify your data file path)
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify with your data file path

# Separate the dataset into training and testing dataset
df_train = df[df['trainTestLabel'].isin(['Training'])]
df_test = df[df['trainTestLabel'].isin(['Test'])]
df_None = df[df['trainTestLabel'].isin(['[0 0]'])]

# Print dataset sizes to verify the split
print("Training set size:", len(df_train))
print("Test set size:", len(df_test))
print("None set size:", len(df_None)) 