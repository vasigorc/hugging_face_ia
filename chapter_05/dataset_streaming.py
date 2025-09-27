from datasets import load_dataset

"""
The IMDB dataset is a popular dataset used for natural language processing (NLP) tasks,
specifically for sentiment analysis. It consists of movie reviews from the Internet Movie
Database (IMDB) and includes both positive and negative reviews.
"""
dataset_id = "stanfordnlp/imdb"
# only get training split
# instead of returning the entire dataset, return
# an IterableDatasetDict
# shuffling the downloaded set with a fixed seed for reproductibility
dataset = load_dataset(dataset_id, streaming=True, split="train").shuffle(seed=42)
print(dataset)

# to fetch the dataset, enumerate through it and fetch one row at a time
# the following code snippet shows how to print first two rows in a
# fetched dataset
print("First two rows of the fetched dataset:")
for i, example in enumerate(dataset):
    if i < 2:
        print(example)
    else:
        break
