import os
import sys
import time
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar

#importing optuna module for hyperparameter optimization. Might be useful later for hyperparameter search
import optuna

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#importing and using the IMDB dataset
from dataloader.imdb_dataloader import IMDBDataset

#setting CUDA for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_config(config: dict, config_dir: str = "config", run_index: int = 0):
    """
    Saves the configuration dictionary to a YAML file
    """
    os.makedirs(config_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"config_{run_index}_{timestamp}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    print(f"Configuration saved to {config_path}")


def train_epoch(model, dataloader, criterion, optimizer):
    """
    Performs one epoch of training; returns average loss and weighted F1 score
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_labels = batch["sentiment"].to(device)
        optimizer.zero_grad()
        # Forward pass: use only the sentiment head
        sentiment_logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(sentiment_logits, sentiment_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
        labels = sentiment_labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_f1


def validate_epoch(model, dataloader, criterion):
    """
    Performs one epoch of validation; returns loss, F1, ROC-AUC, labels and probabilities
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_labels = batch["sentiment"].to(device)
        with torch.no_grad():
            sentiment_logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(sentiment_logits, sentiment_labels)
        running_loss += loss.item() * input_ids.size(0)
        probs = torch.softmax(sentiment_logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
        labels = sentiment_labels.cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        epoch_roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_roc_auc = 0.0
    return epoch_loss, epoch_f1, epoch_roc_auc, all_labels, all_probs


def plot_roc_curve(labels, probs, save_path):
    """
    Plots and saves the ROC curve
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_losses(train_losses, val_losses, save_path):
    """
    Plots and saves training and validation loss curves
    """
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def get_model(model_type):
    """
    Loads the BERT model for sentiment analysis
    This version is tailored solely for the IMDB dataset
    """
    from models.BERT_model import load_model as load_model_bert
    #calling the load_model_bert with only num_sentiment_labels
    model = load_model_bert(num_sentiment_labels=2)
    #freezing pretrained transformer layers. Performing this step increased the model's performance
    for param in model.bert.parameters():
        param.requires_grad = False
    return model.to(device)


def train_model(args, n_epochs):
    """
    Standard training loop for a given number of epochs
    """
    #saving configuration including dataset info
    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "run_index": args.run_index,
        "dataset": "IMDB"
    }
    save_config(config, config_dir="config", run_index=args.run_index)

    #loading the processed IMDB dataset
    dataset_path = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")
    dataset = IMDBDataset(csv_file=dataset_path)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model_type)

    #setting up optimizer with differential learning rates
    pretrained_params = list(model.bert.parameters())
    #updating only the sentiment classifier
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": args.lr_pretrained},
        {"params": model.sentiment_classifier.parameters(), "lr": args.lr_classifier}
    ])

    best_val_loss = float("inf")
    best_epoch = -1
    performance_records = []
    train_losses = []
    val_losses = []
    criterion = nn.CrossEntropyLoss()

    total_layers = len(model.bert.encoder.layer)

    for epoch in range(1, n_epochs + 1):
        #aggressive Gradual Unfreezing: unfreeze (epoch-1) layers from the end
        n = min(epoch - 1, total_layers)
        if n > 0:
            for i in range(total_layers - n, total_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        print(f"\nEpoch {epoch}/{n_epochs}")
        start_time = time.time()
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_roc_auc, val_labels, val_probs = validate_epoch(model, val_loader, criterion)
        end_time = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | ROC-AUC: {val_roc_auc:.4f}")
        print(f"Epoch time: {end_time - start_time:.2f} seconds")

        performance_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc
        })

        #saving the checkpoint based on the current validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            roc_plot_path = os.path.join(checkpoint_dir, f"roc_curve_epoch_{epoch}.png")
            plot_roc_curve(val_labels, val_probs, roc_plot_path)

    loss_plot_path = os.path.join(project_root, "checkpoints", "loss_plot.png")
    plot_losses(train_losses, val_losses, loss_plot_path)
    performance_df = pd.DataFrame(performance_records)
    performance_csv_path = os.path.join(project_root, "checkpoints", "performance_metrics.csv")
    performance_df.to_csv(performance_csv_path, index=False)
    print(f"\nTraining complete. Best model found at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    print(f"Performance metrics saved to {performance_csv_path}")
    return best_val_loss


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    """
    #suggested hyperparameters for optuna:
    lr_pretrained = trial.suggest_loguniform("lr_pretrained", 1e-6, 1e-4)
    lr_classifier = trial.suggest_loguniform("lr_classifier", 1e-5, 1e-4)
    dropout_rate = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    #setting up arguments for a short training run
    trial_args = argparse.Namespace(
        model_type="bert",
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,  #This is not used directly instead lr_classifier is used
        run_index=trial.number,
        lr_pretrained=lr_pretrained,
        lr_classifier=lr_classifier
    )
    
    #loading the model and set dropout rate
    from models.BERT_model import load_model as load_model_bert
    model = load_model_bert(num_sentiment_labels=2)
    model.dropout.p = dropout_rate
    #freezing pretrained layers
    for param in model.bert.parameters():
        param.requires_grad = False
    model = model.to(device)
    
    #setting up optimizer with differential learning rates using suggested values
    pretrained_params = list(model.bert.parameters())
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": lr_pretrained},
        {"params": model.sentiment_classifier.parameters(), "lr": lr_classifier}
    ])
    
    #loading IMDB dataset
    dataset_path = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")
    dataset = IMDBDataset(csv_file=dataset_path)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=trial_args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trial_args.batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    total_layers = len(model.bert.encoder.layer)
    
    #running training for a few epochs
    for epoch in range(1, trial_args.epochs + 1):
        n = min(epoch - 1, total_layers)
        if n > 0:
            for i in range(total_layers - n, total_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, _, _, _, _ = validate_epoch(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss


def main_wrapper(args):
    if args.optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)
        print("Best trial:")
        print(study.best_trial)
    else:
        train_model(args, n_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the IMDB Sentiment model with differential learning rates, aggressive gradual unfreezing, and hyperparameter optimization with Optuna (BERT model)"
    )
    #other model such as RoBerta could be implemented similar to Bert model
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert"], help="Model type to use for training (only 'bert' is supported)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (full training mode)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Base learning rate for classifier heads")
    parser.add_argument("--run_index", type=int, default=0, help="Run index for config file naming")
    parser.add_argument("--optuna", action="store_true", help="If set, run hyperparameter optimization instead of standard training")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for Optuna study (if --optuna is set)")
    
    #Additional hyperparameters for optimizer:
    parser.add_argument("--lr_pretrained", type=float, default=1e-5, help="Learning rate for pretrained layers (used in standard training)")
    parser.add_argument("--lr_classifier", type=float, default=2e-5, help="Learning rate for classifier layers (used in standard training)")
    args = parser.parse_args()
    
    main_wrapper(args)