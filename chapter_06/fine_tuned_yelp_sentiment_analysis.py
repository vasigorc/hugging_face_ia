"""
Fine-tuned sentiment analysis for restaurant reviews using the Yelp Polarity dataset.

This module demonstrates how to fine-tune a DistilBERT model for binary sentiment
classification on restaurant reviews. It filters the Yelp dataset for restaurant-related
reviews, trains a model, and uses it for inference.
"""

from pathlib import Path
from typing import Dict, Tuple, cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Constants
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_REVIEWS = 5000
NUM_LABELS = 2
MAX_LENGTH = 512
RESULTS_DIR = "./results"
FINAL_MODEL_PATH = "./results/final_model"
FINAL_TOKENIZER_PATH = "./results/final_tokenizer"
RANDOM_SEED = 42


def get_device() -> torch.device:
    """
    Determine the best available device for training and inference.

    Returns:
        torch.device: CUDA if available, MPS if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_and_filter_dataset() -> Tuple[Dataset, Dataset]:
    """
    Load the Yelp Polarity dataset and filter for restaurant reviews.

    Returns:
        Tuple[Dataset, Dataset]: Filtered train and test datasets containing only
                                 restaurant-related reviews.
    """
    print("Loading Yelp Polarity dataset...")
    dataset = load_dataset("yelp_polarity")
    train_dataset: Dataset = cast(Dataset, dataset["train"])
    test_dataset: Dataset = cast(Dataset, dataset["test"])

    print("Filtering for restaurant reviews...")
    restaurant_train = train_dataset.filter(lambda x: "restaurant" in x["text"].lower())
    restaurant_test = test_dataset.filter(lambda x: "restaurant" in x["text"].lower())

    return restaurant_train, restaurant_test


def create_dataset_subset(
    train_dataset: Dataset, test_dataset: Dataset, num_reviews: int = NUM_REVIEWS
) -> DatasetDict:
    """
    Create a subset of the dataset with a specified number of reviews.

    Args:
        train_dataset: Training dataset to sample from.
        test_dataset: Test dataset to sample from.
        num_reviews: Number of reviews to include in each subset.

    Returns:
        DatasetDict: A dictionary containing train and test subsets.
    """
    print(f"Creating subset with {num_reviews} reviews per split...")
    subset_train = train_dataset.shuffle(seed=RANDOM_SEED).select(range(num_reviews))
    subset_test = test_dataset.shuffle(seed=RANDOM_SEED).select(range(num_reviews))

    return DatasetDict({"train": subset_train, "test": subset_test})  # type: ignore


def tokenize_dataset(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizer
) -> DatasetDict:
    """
    Tokenize the dataset using the provided tokenizer.

    Args:
        dataset: The dataset to tokenize.
        tokenizer: The tokenizer to use for processing text.

    Returns:
        DatasetDict: Tokenized dataset ready for training.
    """
    print("Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_function, batched=True)


def model_exists() -> bool:
    """
    Check if a fine-tuned model and tokenizer already exist on disk.

    Returns:
        bool: True if both model and tokenizer exist, False otherwise.
    """
    model_path = Path(FINAL_MODEL_PATH)
    tokenizer_path = Path(FINAL_TOKENIZER_PATH)
    return model_path.exists() and tokenizer_path.exists()


def train_model(
    tokenized_datasets: DatasetDict, device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Trainer]:
    """
    Train a DistilBERT model for sequence classification.

    Args:
        tokenized_datasets: The tokenized training and evaluation datasets.
        device: The device to train on (CPU, CUDA, or MPS).

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer, Trainer]: The trained model,
                                                               tokenizer, and trainer.
    """
    print("Loading pretrained model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=NUM_LABELS
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    model.to(device)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
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

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained(FINAL_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_TOKENIZER_PATH)

    return model, tokenizer, trainer


def load_trained_model(
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a previously trained model and tokenizer from disk.

    Args:
        device: The device to load the model onto.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer.
    """
    print("Loading existing fine-tuned model from disk...")
    model = AutoModelForSequenceClassification.from_pretrained(
        FINAL_MODEL_PATH, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        FINAL_TOKENIZER_PATH, local_files_only=True
    )
    model.to(device)
    return model, tokenizer


def evaluate_model(trainer: Trainer) -> Dict:
    """
    Evaluate the trained model on the test dataset.

    Args:
        trainer: The trainer object containing the model and evaluation dataset.

    Returns:
        Dict: Evaluation results including loss and metrics.
    """
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    return eval_results


def predict_sentiment(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> None:
    """
    Predict the sentiment of a given text using the fine-tuned model.

    Args:
        text: The text to analyze.
        model: The fine-tuned model.
        tokenizer: The tokenizer corresponding to the model.
        device: The device the model is on.
    """
    print("\nPredicting sentiment for sample review...")

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if predicted_class == 1:
        sentiment = "Positive"
        confidence = probabilities[0][1].item()
    else:
        sentiment = "Negative"
        confidence = probabilities[0][0].item()

    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    if model_exists():
        print("\nFine-tuned model already exists. Skipping training.")
        model, tokenizer = load_trained_model(device)
        trainer = None
    else:
        print("\nNo existing model found. Starting training process.")
        train_dataset, test_dataset = load_and_filter_dataset()
        dataset_subset = create_dataset_subset(train_dataset, test_dataset)

        temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        tokenized_datasets = tokenize_dataset(dataset_subset, temp_tokenizer)

        model, tokenizer, trainer = train_model(tokenized_datasets, device)

        if trainer:
            evaluate_model(trainer)

    sample_review = """
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

    predict_sentiment(sample_review, model, tokenizer, device)


if __name__ == "__main__":
    main()
