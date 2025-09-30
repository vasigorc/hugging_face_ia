from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load dataset in memory (not streaming)
print("Loading IMDB dataset...")
dataset = load_dataset("stanfordnlp/imdb", split="train")

# 2. Skip the first two reviews and keep the rest
print("Skipping the first two reviews...")
dataset_tail = dataset.select(range(2, len(dataset)))

# 3. Load tokenizer
print("Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# 4. Tokenize the text field with truncation + padding
print("Tokenizing reviews...")
tokenized = dataset_tail.map(
    lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length")
)

# 5. Inspect the very first review after skipping
print("\n--- First example after skipping ---")
example = dataset_tail[0]
print("Raw review text:", example["text"])
print("Sentiment label:", example["label"])  # 0=negative, 1=positive

# 6. Look at its tokenized version
tokenized_example = tokenized[0]
print("\nTokenized (first example):")
print("Input IDs:", tokenized_example["input_ids"])

# 7. Convert IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(tokenized_example["input_ids"])
print("\nDecoded tokens (first 30 shown):")
print(tokens[:30])  # truncate for readability

# 8. Convert IDs back to full string (skip special tokens like [CLS], [SEP])
decoded_text = tokenizer.decode(
    tokenized_example["input_ids"], skip_special_tokens=True
)
print("\nDecoded back to string:")
print(decoded_text)
