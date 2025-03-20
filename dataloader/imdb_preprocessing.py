import os
import pandas as pd

def process_imdb_dataset(aclimdb_path, output_csv):
    """
    Processes the ACL IMDB dataset and saves a combined CSV file
    
    The ACL IMDB dataset structure:
        aclImdb/
            train/
                pos/  (positive reviews, label=1)
                neg/  (negative reviews, label=0)
                unsup/ (optional unsupervised reviews, usually ignored for sentiment tasks)
            test/
                pos/
                neg/
                
    This function reads all reviews from the 'train' and 'test' splits (ignoring the 'unsup' folder),
    assigns a sentiment label (1 for positive, 0 for negative), and adds a column indicating the split
    
    Args:
        aclimdb_path (str): Path to the aclImdb folder
        output_csv (str): Path where the combined CSV will be saved
    """
    data = []
    
    #processing both train and test splits
    for split in ["train", "test"]:
        for sentiment in ["pos", "neg"]:
            folder_path = os.path.join(aclimdb_path, split, sentiment)
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} does not exist. Skipping.")
                continue
            
            #Setting label: 1 for pos, 0 for neg
            label = 1 if sentiment == "pos" else 0
            
            #Processing each text file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            review = f.read().strip()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    data.append({
                        "review_text": review,
                        "sentiment": label,
                        "split": split
                    })
    
    #Creating DataFrame and shuffle data
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    #Making sure that the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed IMDB dataset saved to {output_csv}")

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    aclimdb_path = os.path.join(project_root, "data", "raw", "aclImdb")
    output_csv = os.path.join(project_root, "data", "processed", "imdb_reviews_processed.csv")
    
    process_imdb_dataset(aclimdb_path, output_csv)