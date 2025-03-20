import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class IMDBDataset(Dataset):
    """
    PyTorch Dataset for the IMDB reviews
    
    Each sample includes:
      - Tokenized review text (input_ids and attention_mask)
      - Sentiment label (binary, e.g., 0 for negative, 1 for positive)
    """
    def __init__(self, csv_file, tokenizer=None, max_length=128):
        """
        Initializes the IMDB dataset.
        
        Args:
            csv_file (str): Path to the processed IMDB CSV file
            tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer for encoding review text
                If None, the default 'bert-base-uncased' tokenizer is loaded
            max_length (int): Maximum sequence length for tokenization
        """
        self.data = pd.read_csv(csv_file)
        # Use provided tokenizer or load default
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        
        # Ensure necessary columns exist
        required_columns = ['review_text', 'sentiment']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #getting the review text and sentiment label from the dataframe
        review_text = self.data.loc[idx, "review_text"]
        sentiment = self.data.loc[idx, "sentiment"]
        
        #Tokenizing the review text using the provided tokenizer
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        #squeeze the batch dimension and create tensors
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentiment": torch.tensor(sentiment, dtype=torch.long)
        }

#testing the dataset
if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    csv_path = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")
    
    # Initialize the dataset
    dataset = IMDBDataset(csv_file=csv_path)
    print(f"IMDB Dataset loaded with {len(dataset)} samples.")
    
    #printing the details of the first sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Sentiment label:", sample["sentiment"])