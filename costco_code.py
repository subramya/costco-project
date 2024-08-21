import pytesseract
from PIL import Image, ImageEnhance
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

image_path = '/Users/Ramya_1/Desktop/costco-project/costco_receipt.png'
image = Image.open(image_path)
image = image.convert('L')
width, height = image.size
image = image.resize((width * 2, height * 2), Image.LANCZOS)

# Increase contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)

# Apply thresholding
image = image.point(lambda x: 0 if x < 150 else 255, '1')

# Extract text after preprocessing
text = pytesseract.image_to_string(image, lang='eng')

# Split text into lines
lines = text.split('\n')

# Keywords to exclude from item processing
exclude_keywords = ['SUBTOTAL', 'TAX', 'TOTAL', 'CHANGE', 'CHECK', 'MEMBER', 'NUMBER', 'SOLD', 'ITEMS', 'THANK']
replacements = {
    '£': 'E',
    'HONT': 'MONT',
    '3 @': '' 
}

# Process lines and extract items with prices
processed_items = []
for line in lines:
    for wrong, correct in replacements.items():
        line = line.replace(wrong, correct)
    if any(keyword in line.upper() for keyword in exclude_keywords):
        continue
    line = line.replace('€', '$')
    match = re.search(r'\d+\.\d{2}', line)
    if match:
        price = match.group()
        item_name = line.split(price)[0].strip()
        item_name = re.sub(r'\s+\d+\.?\d*$', '', item_name)
        if len(item_name) > 2 and not item_name.replace('.', '', 1).isdigit():
            processed_items.append((item_name, price))

print(processed_items)

categories = ["Groceries", "Household Items", "Electronics", "Clothing", "Miscellaneous"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

item_names = [item[0] for item in processed_items]
inputs = tokenizer(item_names, return_tensors="pt", padding=True, truncation=True, max_length=64)

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=1)
categorized_items = [(item, categories[prediction]) for item, prediction in zip(processed_items, predictions)]

for item, category in categorized_items:
    print(f'Item: {item[0]}, Price: {item[1]}, Category: {category}')
