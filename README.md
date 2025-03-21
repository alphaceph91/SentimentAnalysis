# IMDB Sentiment Analysis with BERT
The core idea of this project was to try to develop a **binary sentiment analysis** model (positive vs negative) using the **IMDB Movie Reviews** [dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The project uses **BERT-based** model architecture to classify whether a given movie review is **Good** (a positive sentiment) or **Bad** (a negative sentiment). 

## Loss Plot
![Image](https://github.com/alphaceph91/SentimentAnalysis/blob/main/loss_plot.png)

## ROC AUC CURVE (BEST)
![Image](https://github.com/alphaceph91/SentimentAnalysis/blob/main/roc_curve_epoch_5.png)

## Project Structure
```
.
├── checkpoints
│   └── ...                  	  #Model checkpoints, plots, metrics (gets created automatically)
├── config
│   └── ...                  	  #YAML config files for runs is generated automatically once train.py is executed
├── data
│   ├── processed	     	        #create this folder
│   └── raw		     	            #create this folder, download the dataset with the provided link and extract, paste the aclImdb folder here
│       └── aclImdb		          #after pasting the folder should look like this	
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
│   └── test_BERT_model.py 	    #script for testing BERT_model using dummy data and imdb_reviews_processed.csv dataset
│   └── test_imdb_dataloader.py #script to verify IMDB dataloader is working correctly or not
├── environment.yml		          #tested for linux 
├── train.py 			              #primary training script with performance metrics, plots etc
├── predict.py                  #this script is used for loading trained model checkpoint and inference 
```

## Dataset
The project uses IMDB dataset
- 50,000 reviews split into train (25,000) and test (25,000)
- Each review is labeled as positive (label = 1) or negative (label = 0)
- In aclImdb/, pos/ holds positive reviews, while neg/ holds negative reviews
- We ignore the unsup/ folder since it’s unlabeled
- We merge the data into a single CSV by run [imdb_preprocessing.py](https://github.com/alphaceph91/SentimentAnalysis/blob/main/dataloader/imdb_preprocessing.py), which collects reviews from train and test directories into **imdb_reviews_processed.csv**

## Model Architecture
The project uses a BERT model, defined in [BERT_model.py](https://github.com/alphaceph91/SentimentAnalysis/blob/main/models/BERT_model.py) for sentiment classification
- ```BERTSentimentModel``` extends ```BertPreTrainedModel```
- Pooled Output from BERT (```outputs.pooler_output```) passes through a dropout layer and then a Linear layer (```sentiment_classifier```) with output dimension = 2
- Forward Method returns only sentiment logits

## Training and Evaluation
**Training Script** [train.py](https://github.com/alphaceph91/SentimentAnalysis/blob/main/train.py) handles data loading,model creation, training loops, and evaluation
- ```get_model``` loads the BERT-based sentiment-only model (```load_model_bert(num_sentiment_labels=2)```) and optionally freezes all BERT layers initially
- ```train_epoch``` performs one epoch of training. Minimizes cross-entropy loss on the sentiment logits
- ```validate_epoch``` evaluates on the validation split, computing loss, F1, and ROC-AUC
- ```plot_roc_curve / plot_losses``` saves training curves and the ROC curve
- train.py tracks the best validation loss and saves the best model checkpoint in **checkpoints/**
- During training ee freeze the pretrained layers at first, then gradually unfreeze them in subsequent epochs (aggressive gradual unfreezing). This helped for improving the model's performance significantly

For training the model:
```sh
python train.py --model_type bert --batch_size 32 --epochs 5 --lr_pretrained 1e-5 --lr_classifier 2e-5
```

**Data Splits**
We split 80% for training and 20% for validation using random_split. During training:
- ```train_epoch``` updates weights on the training set
- ```validate_epoch``` checks performance on the validation set

**Performance Metrics**
- Loss: Standard cross-entropy.
- F1 Score (weighted): Evaluates classification performance.
- ROC-AUC: Measures separability between positive and negative.
- The train.py script logs these metrics for each epoch
- Thr train.py script saves train/val loss plot, roc_auc_curve during each epoch and saves a performance_metrics.csv
- performance_metrics.csv consists of: epoch, train_loss, train_f1, val_loss, val_f1, val_roc_auc

**Best Score**
- Epoch 5
- val_f1: 0.718965871977108
- val_roc_auc: 0.79667576

## Hyperparameter Optimization (Optuna)
- train.py supports hyperparameter search using Optuna module
- Hyperparameter search was not performed due to hardware limitation
- For advanced experiments, you can enable Optuna with the ```--optuna``` flag
- Optuna systematically searches for the combination of hyperparameters yielding the lowest validation loss
```sh
python train_optuna.py --optuna --n_trials 20
```

## Inference
The script [predict.py](https://github.com/alphaceph91/SentimentAnalysis/blob/main/predict.py):
- Load a trained BERT checkpoint (```.pth```) from **checkpoints/**
- Tokenize an input review text with BertTokenizer
- Forward pass to get sentiment logits
- Print the classification result: “Good” vs. “Bad” and its probabilities

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
