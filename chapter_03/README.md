# Chapter 3. Using Hugging Face Transformers and Pipelines for NLP Tasks

## Approached notions

### Tokenization

In the context of NLP and machine learning, a "token" is a chunk of text that a model processes as a single unit. Tokens can represent an individual words, punctuation marks, or other linguistic elements, depending on the specific tokenization strategy employed. Tokenization is the process of converting a text document or sentence down into smaller units.

### Tensor

A **tensor** is a multi-dimensional array used in frameworks like PyTorch to efficiently handle numerical data, especially for GPU-accelerated computations.

### BERT

BERT stands for Bidirectional Encoder Representations from Transformers. It is a transformer-based machine learning model designed for natural language processing (NLP) tasks.

It is commonly used in natural language processing tasks such as question answering, text classification, named entity recognition, part-of-speech tagging, text summarization, sentiment analysis, language translation, text generation, coreference resolution, paraphrase detection, semantic search, textual entailment, and dialog systems.

### DistilBERT

It is a lighter, smaller, and faster version of BERT, suitable for deployment in resource-constrained environments.

### Token Embeddings

Token embeddings converts tokens into numerical vectors. These embeddings capture semantic and syntactic information about the tokens, enabling machine learning models to understand the underlying meaning and relationships between words in natural language text. In the context of NLP tasks, word embeddings allow you to see which words are often used together. It captures semantic
relationships between words based on their patterns in large text corpora. The embeddings are learned based on the co-occurrence and contextual relationships between words in the training corpus. As a result, words that have similar meanings or appear in similar contexts tend to have similar representations in the embedding space.

### t-SNE

It is a dimensionality reduction technique.

From ["Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow. O'Reilly. Third Edition"](https://www.amazon.ca/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975/):

> _t-distributed stochastic neighbor embedding_... reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space.

### Positional Encoding

Plays a crucial role in transformer-based models by providing positional information on the order of tokens within a sequence.
It is vital for model to grasp the meaning and context of the input accurately. By incorporating this encoding into the embeddings,
the model gains ability to discern between tokens based on their positions in the sequence. Positional encoding is typically
added to the token embeddings before they are input to the transformer model.

E.g.: The sentence "The cat sat on the sofa" has a different meaning than "The sofa set on the cat"

### Softmax

Softmax is a mathematical function that converts a vector of numbers into a probability distribution, where the probability
of each element is proportional to the exponentiation of that element's value relative to the sum of all the exponentiated values in the vector. In neural networks, Softmax is often used as the final activation function in classification tasks.

## Code examples

### Token Embeddings

In [this example](./token_embeddings.py) we are given code that extracts tokens from a sample short text, passes them to the BERT model, generates the embeddings and prints each word (decoded token) next to its embeddings:

```bash
uv run python chapter_03/token_embedddings.py
...
life: tensor([ 3.6462e-01, -5.7927e-02, -1.3655e-02, -1.0991e-01,  5.0892e-01,
        -7.5323e-03,  4.4139e-01,  7.5105e-01, -4.6468e-01,  4.3893e-01,
        -9.9271e-03, -8.2384e-01, -2.4836e-01,  9.2323e-01, -1.5780e-01,
         5.8997e-01,  4.1004e-01,  1.4265e-01, -2.5426e-01,  3.9373e-01,
...
```

We are using token embeddings to visualize the relationship between tokens. Since the embeddings have a high dimension cardinality, we are using `t-SNE` first to reduce dimensionality to 2 dimensions per token.  
We further use `matplotlib` to plot the tokens' graph:

![Tokens relationship graph](./token_embeddings.png)

### Sentiment Analysis

As in the previous example, in [this exercise](./sentiment_analysis.py) we kick off by tokenizing the input:

```bash
uv run python chapter_03/sentiment_analysis.py
tokenizer_config.json: 100%|...| 48.0/48.0 [00:00<00:00, 406kB/s]
config.json: 100%|...| 629/629 [00:00<00:00, 5.20MB/s]
vocab.txt: 232kB [00:00, 8.20MB/s]
model.safetensors: 100%|█...8M [00:03<00:00, 82.3MB/s]
{'input_ids': tensor([[  101,  1045,  3866,  1996,  3185,  1010,  2009,  2001, 10392,   999,
           102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

What gets printed next is the model's output:

```bash
...
SequenceClassifierOutput(loss=None, logits=tensor([[-4.3428,  4.6955]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
Predicted sentiment: positive
```

The `logit` key points to a value of shape (1, 2), where the first dimension corresponds to the batch size (number of texts), and the second dimension corresponds to the number of classes (2, 0 for "negative", and 1 for "positive"). Looking at the scores suggests that model correctly identified the text as highly likely to have positive tone. Hence, the predicted sentiment.
