from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

print("Question detection examples:")

question_classifier = pipeline(
    "text-classification", model="huaen/question_detection", device="cuda"
)

response = question_classifier(
    """Have you ever pondered the mysteries that lie beneath the surface of every day life?"""
)
print(response)

response = question_classifier(
    """"Life is a journey that must be traveled, no matter how bad the roads and accomodations." - Olive GoldSmith"""
)
print(response)

print("Language detection example:")

language_classifier = pipeline(
    "text_classification",
    model="papluca/xlm-roberta-base-language-detection",
    device="cudaq",
)

response = language_classifier("日本の桜は美しいです。")
print(response)

print("Spam classification examples:")

spam_classifier = pipeline(
    "text_classification", model="Delphia/twitter-spam-classifier", device="cuda"
)

response = spam_classifier(
    """Congratulations! You've been selected as the winener of our exclusive prize draw.
    Claim your reward now by clicking on the link below!"""
)
print(response)

response = spam_classifier(
    """Hi Jimmy, I hope you're doing well. I just wanted to remind
    you about our meeting tomorrow at 10 AM in conference room A.
    Please let me know if you have any questions or need any
    further information. Looking forward to seeing you there!"""
)
print(response)

print("Text generation example:")

generator = pipeline("text-generation", model="openai-community/gpt2", device="cuda")
generator("In this course, we will teach you how to")
generator(
    "In this course, we will teach you how to", max_length=50, num_return_sequences=3
)
