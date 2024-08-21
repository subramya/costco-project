import torch, os
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset

grocery_df = pd.read_csv("/Users/Ramya_1/Desktop/costco-project/GroceryDataset.csv")

grocery_df.rename(columns={'Sub Category': 'category'}, inplace=True)
grocery_df = grocery_df.drop(columns=['Price', 'Discount', 'Rating', 'Currency', 'Feature', 'Product Description'])


labels = grocery_df['category'].unique().tolist()
labels = [s.strip() for s in labels]

NUM_LABELS = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# print("ID to Label mapping:", id2label)
# print("Label to ID mapping:", label2id)

grocery_df["labels_num"] = grocery_df["category"].map(label2id)
# print(grocery_df.head())

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f"Model is on {device}")
 
