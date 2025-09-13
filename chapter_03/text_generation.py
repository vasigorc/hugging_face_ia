from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

print("Text generation example:")

generator = pipeline("text-generation", model="openai-community/gpt2")
generator("In this course, we will teach you how to")
generator(
    "In this course, we will teach you how to",
    max_length=50,
    num_return_sequences=3,
)
