from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
input_text = "What is unhappiness?"
tokens = tokenizer.tokenize(input_text, return_tensors="pt")

print(f"{tokens = }")
