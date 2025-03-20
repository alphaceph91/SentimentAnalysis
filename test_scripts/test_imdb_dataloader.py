import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataloader.imdb_dataloader import IMDBDataset

def main():
    csv_path = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")

    if not os.path.exists(csv_path):
        print(f"Processed CSV file not found at {csv_path}. Please run imdb_preprocessing.py first")
        return

    #loading the processed CSV to verify its contents
    df = pd.read_csv(csv_path)
    print("Processed IMDB dataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print("Column names:", list(df.columns))
    print("Distribution of splits:")
    print(df["split"].value_counts())
    print("Distribution of sentiment labels:")
    print(df["sentiment"].value_counts())
    
    #testing the IMDB Dataset class using the processed CSV
    print("\nTesting the IMDBDataset class...")
    dataset = IMDBDataset(csv_file=csv_path)
    print(f"IMDB Dataset loaded with {len(dataset)} samples.")
    
    #retrieving a sample and print details
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Sentiment label:", sample["sentiment"])

if __name__ == "__main__":
    main()