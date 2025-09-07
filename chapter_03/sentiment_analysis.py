import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.auto.configuration_auto import model_type_to_module_name

"""
Usting DistilBERT to perform sentiment analysis on a piece of text.
Sentiment analysis is a NLP technique to analyze texttual data to categorize
the sentiment as a positive, negative, or neutral, indicating the overall emotional
tone or polarity of the text.
"""

# getting the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
# load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
# tokenize the text
text = "I loved the movie, it was fantastic!"

inputs = tokenizer(text, return_tensors="pt")
print(inputs)

# pass the tookenized representation to the model
# ** - unpacks a dictionary
# the below is equivalent of
# model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
outputs = model(**inputs)
print(outputs)

# move the resolution from probabilitic into deterministic mode
predicted_label = torch.argmax(outputs.logits)
sentiment = "positive" if predicted_label == 1 else "negative"

print(f"Predicted sentiment: {sentiment}")
