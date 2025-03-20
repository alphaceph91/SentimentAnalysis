# IMDB Sentiment Analysis with BERT
The core idea of this project was to try to develop a **binary sentiment analysis** model (positive vs negative) using the **IMDB Movie Reviews** [dataset] (https://ai.stanford.edu/~amaas/data/sentiment/). The project uses **BERT-based** model architecture to classify whether a given movie review is **Good** (a positive sentiment) or **Bad** (a negative sentiment). 

## Loss Plot
![Image](https://github.com/alphaceph91/SentimentAnalysis/blob/main/loss_plot.png)

## ROC AUC CURVE (BEST)
![Image](https://github.com/alphaceph91/SentimentAnalysis/blob/main/roc_curve_epoch_5.png)

## Project Structure
```
.
├── checkpoints
│   └── ...                  	#Model checkpoints, plots, metrics (gets created automatically)
├── config
│   └── ...                  	#YAML config files for runs is generated automatically once train.py is executed
├── data
│   ├── processed	     	#create this folder
│   └── raw		     	#create this folder, download the dataset with the provided link and extract, paste the aclImdb folder here
│       └── aclImdb		#after pasting the folder should look like this	
│           ├── train/pos/...   #IMDB training positive reviews
│           ├── train/neg/...   #IMDB training negative reviews
│           ├── test/pos/...    #IMDB test positive reviews
│           └── test/neg/...    #IMDB test negative reviews
├── dataloader
│   ├── imdb_dataloader.py      #Dataset class for IMDB reviews
│   └── imdb_preprocessing.py   #Preprocessing script to build imdb_reviews_processed.csv
├── models
│   └── BERT_model.py           #BERT model with a single sentiment classification head
├── test_scripts
│   └── test_BERT_model.py 	#script for testing BERT_model using dummy data and imdb_reviews_processed.csv dataset
│   └── test_imdb_dataloader.py #script to verify IMDB dataloader is working correctly or not
├── environment.yml		#tested for linux 
├── train.py 			#primary training script with performance metrics, plots etc
├── predict.py
```

## Future Enhancements
- Integrating another dataset for example a Game Review dataset
- Implementing advanced data augmentation for back-translation or paraphrasing to expand the dataset and improve generalization
- Experiment with Larger Models: Implmentation of BERT-large or RoBERTa models might yield higher accuracy at the cost of increased computation
- Adding a validation curve for monitoring not just final metrics but also learning rate scheduling, memory usage, etc., providing deeper insights

## Pretrained Model
A pretrained model checkpoint trained for 5 epochs with batch_size=32 could be downloaded [here](https://drive.google.com/file/d/1ehGbUmuoNl4tUSYOeQ1QlqsRP9sAl1z-/view?usp=drive_link)

## References
- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [BERT](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Optuna](https://optuna.org/)
