import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig

class BERTSentimentModel(BertPreTrainedModel):
    """
    Custom BERT-based model for sentiment analysis
    This model includes only a sentiment classification head (good vs. bad sentiment)
    """
    def __init__(self, config, num_sentiment_labels):
        super(BERTSentimentModel, self).__init__(config)
        
        #loading the pre-trained BERT model
        self.bert = BertModel(config)
        #dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #classifier head for sentiment analysis for binary classification
        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)
        
        #initialize weights using BertPreTrainedModel's init_weights method
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass for the model
        
        Args:
            input_ids (torch.Tensor): Tokenized input with shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Mask to avoid attention on padding tokens
            token_type_ids (torch.Tensor): Segment token indices to indicate different portions of the input
        
        Returns:
            torch.Tensor: Logits for sentiment classification
        """
        #getting outputs from BERT model. outputs.pooler_output is used
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output

        #applying dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        #getting logits from the sentiment classifier head
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        return sentiment_logits

def load_model(model_name='bert-base-uncased', num_sentiment_labels=2):
    """
    Loads a pre-trained BERT configuration and initializes the custom BERTSentimentModel
    
    Args:
        model_name (str): Name of the pre-trained BERT model (default is 'bert-base-uncased')
        num_sentiment_labels (int): Number of sentiment classes (typically 2 for binary classification)
    
    Returns:
        BERTSentimentModel: The initialized sentiment-only model
    """
    #loading the pre-trained BERT configuration from Hugging Face
    config = BertConfig.from_pretrained(model_name)
    
    #initializing the sentiment-only model with the provided configuration and number of sentiment classes
    model = BERTSentimentModel(config, num_sentiment_labels=num_sentiment_labels)
    
    return model