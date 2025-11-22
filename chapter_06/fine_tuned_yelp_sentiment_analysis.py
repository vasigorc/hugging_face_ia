from typing import cast
import torch
from torch import device
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import Dataset, DatasetDict, load_dataset
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

dataset = load_dataset("yelp_polarity")  # includes 560K rows!
train_dataset: Dataset = cast(Dataset, dataset["train"])
test_dataset: Dataset = cast(Dataset, dataset["test"])

restaurant_train_reviews = train_dataset.filter(
    lambda x: "restaurant" in x["text"].lower()
)

restaurant_test_reviews = test_dataset.filter(
    lambda x: "restaurant" in x["text"].lower()
)

number_of_reviews = 5000
subset_train_reviews = restaurant_train_reviews.shuffle(seed=42).select(
    range(number_of_reviews)
)
subset_test_reviews = restaurant_test_reviews.shuffle(seed=42).select(
    range(number_of_reviews)
)

subset_dataset = {"train": subset_train_reviews, "test": subset_test_reviews}

yelp_restaurant_dataset = DatasetDict(subset_dataset)  # type: ignore

# tokenizing the reduced dataset
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


tokenized_datasets = yelp_restaurant_dataset.map(tokenize_function, batched=True)

# set-up a pretrained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

model.to(device)

# configure and initialize the trainer
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()

"""
When the model is trained, save it to disk so you can use it later
without going through the training press again
"""
model.save_pretrained(".results/final_model")
tokenizer.save_pretrained("./results/final_tokenizer")

# evaluat the model and print the result
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

"""
Using the fine-tuned model

With the fine-tuned model trained and saved, you can use it to
perform sentiment analysis on a new restaurant review.
"""

new_model = AutoModelForSequenceClassification.from_pretrained("./results/final_model")
new_tokenizer = AutoTokenizer.from_pretrained("./results/final_tokenizer")

new_model.to(device)

sentence = """
I had an amazing experience dining at this restaurant last night.
From the moment we walked in, the staff made us feel welcomed and
were incredibly attentive. Our server was friendly, knowledgeable,
and made great recommendations from the menu.

The food was absolutely delicious. I had the grilled salmon, and
it was cooked to perfection—tender, flavorful, and served with a
lovely citrus glaze that complemented it beautifully. The roasted 
vegetables on the side were fresh and perfectly seasoned. My
partner had the pasta, which was creamy and rich in flavor, with
just the right amount of spice.

The ambiance was warm and inviting, with cozy lighting and tasteful
decor. It was the perfect place to relax and enjoy a nice meal. The 
dessert, a decadent chocolate lava cake, was the perfect way to end
the meal.

Overall, this restaurant exceeded my expectations in every way.
Excellent food, exceptional service, and a wonderful atmosphere.
I'll definitely be back and highly recommend it to anyone looking
for a great dining experience.
"""

inputs = new_tokenizer(
    sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
)

# Move inputs to GPU/MPS
inputs = {key: value.to(device) for key, value in inputs.items()}

new_model.eval()

with torch.no_grad():
    outputs = new_model(**inputs)
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

if predicted_class == 1:
    print(
        f"Sentiment: Positive (Confidence: \
            {probabilities[0][1].item():.2f})"
    )
else:
    print(
        f"Sentiment: Negative (Confidence: \
            {probabilities[0][0].item():.2f})"
    )
