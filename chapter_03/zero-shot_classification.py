import pandas as pd
from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

# Initialize the classifier
zero_shot_classifier = pipeline(
    "zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device="cuda"
)

# Define the texts
text1 = """
    "In the intricate realm of global affairs, the interplay of power, 
    diplomacy, and governance stands as a defining force in the 
    trajectory of nations. Amidst fervent debates in legislative 
    chambers and pivotal dialogues among world leaders, ideologies 
    clash and policies take shape, shaping the course of societies. 
    Issues such as economic disparity, environmental stewardship, and 
    human rights take precedence, driving conversations and shaping 
    public sentiment. In an age of digital interconnectedness, social 
    media platforms have emerged as influential channels for discourse 
    and activism, amplifying voices and reshaping narratives with 
    remarkable speed and breadth. As citizens grapple with the 
    complexities of contemporary governance, the pursuit of accountable 
    and transparent leadership remains paramount, reflecting an 
    enduring quest for fairness and inclusivity in societal governance."""

text2 = """
    In the tender tapestry of human connection, romance weaves its 
delicate threads, binding hearts in a dance of passion and longing. 
    From the flutter of a first glance to the warmth of an intimate 
embrace, love blooms in the most unexpected places, transcending 
barriers of time and circumstance. In the gentle caress of a hand 
and the whispered promises of affection, two souls find solace in 
    each other's embrace, navigating the complexities of intimacy with 
    tender care. As the sun sets and stars illuminate the night sky, 
    lovers share stolen moments of intimacy, lost in the intoxicating 
    rhythm of each other's presence. In the symphony of love, every 
glance, every touch, speaks volumes of a shared bond that defies 
explanation, leaving hearts entwined in an eternal embrace.
    """

# Print the texts
print("=" * 50)
print("Texts for Analysis")
print("=" * 50)
print("Text 1:")
print(text1)
print("-" * 50)
print("Text 2:")
print(text2)
print("=" * 50)

# Define candidate labels
candidate_labels = ["technology", "politics", "business", "romance"]

# Process text1
prediction1 = zero_shot_classifier(text1, candidate_labels, multi_label=True)
prediction1_df = pd.DataFrame(prediction1).drop(["sequence"], axis=1)

print("\n" + "=" * 50)
print("Zero-Shot Classification for Text 1")
print("=" * 50)
print("Predictions:")
print(prediction1_df)
print("=" * 50)

# Process both texts
prediction_both = zero_shot_classifier([text1, text2], candidate_labels, multi_label=True)
prediction_both_df = pd.DataFrame(prediction_both)

print("\n" + "=" * 50)
print("Zero-Shot Classification for Both Texts")
print("=" * 50)
print("Predictions:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(prediction_both_df)
print("=" * 50)