# Chapter 6. Fine-tuning pretrained models and working with multimodal models

## Fine-tuning pretrained models

> Fine-tuning is a machine learning technique in which a pretrained model, which
> has already learned general patterns from a large dataset, is further trained on
> a smaller, domain-specific dataset to adapt for a particular task. This process
> uses the knowledge the model gained from the initial training, enabling to perform
> well with less data and computational resources.

### Sentiment analysis for a pretrained model of restaurant reviews

[Yelp polarity dataset](https://huggingface.co/datasets/fancyzhx/yelp_polarity) is used for this example. This dataset includes ~560K reviews. Noteworthy, not only for
restaurants. Therefore in [fine_tuned_yelp_sentiment_analysis.py](./fine_tuned_yelp_sentiment_analysis.py) we start by filtering the dataset to include only rows containing the word _restaurant_, and then
extract a subset of 5000 rows only.

For this exercise, the book employs `AutoModelForSequenceClassification` class from HF's Transformers library.

> _Sequence classification tasks_ involve assigning a single label or category to an entire
> sequence of data, such as a sentence, a paragraph, or a longer sequence of tokens.

To fine-tune a pretrained model on a dataset, you can use a `Trainer` and `TrainingArguments` classes from HF.

Output on the first run of the program:

```bash
uv run python chapter_06/fine_tuned_yelp_sentiment_analysis.py
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6765, 'grad_norm': 1.0631293058395386, 'learning_rate': 1.9808306709265177e-05, 'epoch': 0.03}
{'loss': 0.6369, 'grad_norm': 1.474997878074646, 'learning_rate': 1.959531416400426e-05, 'epoch': 0.06}
...
{'eval_loss': 0.21187028288841248, 'eval_runtime': 20.3412, 'eval_samples_per_second': 245.807, 'eval_steps_per_second': 15.388, 'epoch': 3.0}
{'train_runtime': 246.7572, 'train_samples_per_second': 60.789, 'train_steps_per_second': 3.805, 'train_loss': 0.14555772379659615, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████| 939/939 [04:06<00:00,  3.81it/s]
100%|███████████████████████████████████████████████████████████████████████| 313/313 [00:20<00:00, 15.42it/s]
Evaluation results: {'eval_loss': 0.1611873060464859, 'eval_runtime': 20.3063, 'eval_samples_per_second': 246.229, 'eval_steps_per_second': 15.414, 'epoch': 3.0}
Sentiment: Positive (Confidence:             0.99)
```

And we are supposed to skip training on subsequent runs due to saving the updated `final_model`:

```bash
uv run python chapter_06/fine_tuned_yelp_sentiment_analysis.py
Using device: cuda

Fine-tuned model already exists. Skipping training.
Loading existing fine-tuned model from disk...

Predicting sentiment for sample review...
Sentiment: Positive (Confidence: 0.99)
```
