import os
import sys
import torch
from models.BERT_model import load_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_dummy_data():
    """
    Testing the BERT model with dummy data
    """
    batch_size = 2
    seq_length = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    # Load the BERT model
    model = load_model(num_game_labels=10, num_sentiment_labels=2)
    model.eval()
    
    with torch.no_grad():
        game_logits, sentiment_logits = model(dummy_input_ids, attention_mask=dummy_attention_mask)
    
    print("Dummy game logits shape:", game_logits.shape)
    print("Dummy sentiment logits shape:", sentiment_logits.shape)

def test_imdb_reviews():
    """
    Test the BERT model using a sample from the IMDB processed CSV.
    """
    from dataloader.imdb_dataloader import IMDBDataset
    csv_path = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")
    dataset = IMDBDataset(csv_file=csv_path)
    
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)
    
    model = load_model(num_game_labels=10, num_sentiment_labels=2)
    model.eval()
    
    with torch.no_grad():
        game_logits, sentiment_logits = model(input_ids, attention_mask=attention_mask)
    
    print("IMDB reviews - Processed CSV sentiment logits shape:", sentiment_logits.shape)

def main():
    print("Testing BERT model with dummy data...")
    test_dummy_data()
    print("\nTesting BERT model with IMDB processed CSV...")
    test_imdb_reviews()

if __name__ == "__main__":
    main()