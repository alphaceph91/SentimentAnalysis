import os
import sys
import argparse
import torch
from transformers import BertTokenizer
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def tokenize_text(review_text, tokenizer, max_length=128):
    """
    Tokenizes the input review text using the provided tokenizer
    Returns input_ids and attention_mask tensors
    """
    encoding = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict(review_text, model, tokenizer, device):
    """
    Performs inference on the input review text
    Returns predicted sentiment index and its probabilities
    """
    model.eval()
    input_ids, attention_mask = tokenize_text(review_text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        #using only the sentiment logits for prediction
        sentiment_logits = model(input_ids, attention_mask=attention_mask)
    
    pred_sentiment_idx = torch.argmax(sentiment_logits, dim=1).item()
    sentiment_probs = F.softmax(sentiment_logits, dim=1).cpu().numpy()[0]
    
    return pred_sentiment_idx, sentiment_probs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from models.BERT_model import load_model as load_model_bert
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #loading the sentiment-only BERT model for IMDB
    model = load_model_bert(num_sentiment_labels=2)
    model = model.to(device)
    
    #loading checkpoint weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    #performing the prediction on the input review text
    pred_sentiment_idx, sentiment_probs = predict(args.review_text, model, tokenizer, device)
    predicted_sentiment = "Good" if pred_sentiment_idx == 1 else "Bad"
    
    #sentiment predictions print statements
    print("\n--- Prediction ---")
    print(f"Input Review Text: {args.review_text}")
    print(f"Predicted Sentiment: {predicted_sentiment} (Probabilities: {sentiment_probs})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment using a saved BERT model checkpoint")
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert"], help="Model type to use for prediction ('bert')")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model checkpoint (.pth file)")
    parser.add_argument("--review_text", type=str, required=True, help="Review text for which to predict sentiment")
    
    args = parser.parse_args()
    main(args)