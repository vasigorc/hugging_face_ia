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
