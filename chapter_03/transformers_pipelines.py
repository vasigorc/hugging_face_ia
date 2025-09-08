from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

try:
    # running this just to get the list of all available Pipeline tasks
    dummy_pipeline = pipeline(task="dummy")
except Exception as e:
    print(e)
    """
        This prints:
        "Unknown task dummy, available tasks are ['audio-classification', 'automatic-speech-recognition', 
        'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 
        'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-text-to-text',
        'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 'question-answering',
        'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 'text-generation',
        'text-to-audio', 'text-to-speech', 'text2text-generation', 'token-classification', 'translation',
        'video-classification', 'visual-question-answering', 'vqa', 'zero-shot-audio-classification',
        'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 'translation_XX_to_YY']"
    """

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda",
)

review1 = """From the warm welcome to the exquisite dishes and impeccable service,
dining at Gourmet Haven is an unforgettable experience that leaves you
eager to return.
    """

review2 = """Despite high expectations, our experience at Savor Bistro
fell short; the food was bland, service was slow, and the overall
atmosphere lacked charm, leaving us disappointed and unlikely
to revisit.
"""

print(classifier([review1, review2]))
