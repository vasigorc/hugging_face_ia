import gradio as gr
from transformers import pipeline  # type: ignore[reportPrivateImportUsage]
from PIL import Image

model = pipeline(
    "image-segmentation",
    model="nvidia/segformer-b0-finetuned-ade-512-512",
    device="cuda",
)


def segmentation(image, label):
    """
    Performs image segmentation, finds a specific label, and overlays a
    colored, semi-transparent mask on the corresponding area of the original image.
    """
    if image is None:
        return None
    if not label or not label.strip():
        return Image.fromarray(image)

    pil_image = Image.fromarray(image)
    results = model(pil_image)

    if not results or not isinstance(results, list):
        return pil_image

    for result in results:
        # Match label case-insensitively
        if result.get("label") == label.strip().lower():
            base_image = pil_image.copy()
            mask = result["mask"]

            # Create a red, semi-transparent overlay
            colored_overlay = Image.new("RGB", base_image.size, (255, 0, 0))
            blended_image = Image.blend(base_image, colored_overlay, alpha=0.4)

            # Paste the blended image onto the original, but only in the masked area
            base_image.paste(blended_image, mask=mask)
            return base_image

    # If the specified label is not found, return the original image
    return pil_image


# Define the Gradio interface
image_input = gr.Image(label="Image to Segment")
label_input = gr.Dropdown(
    label="Label to Highlight",
    choices=list(model.model.config.id2label.values()),
    value="person",
)
image_output = gr.Image(label="Segmented Image")

# Create and launch the Gradio app
iface = gr.Interface(
    fn=segmentation,
    inputs=[image_input, label_input],
    outputs=image_output,
    title="Image Segmentation with Segformer",
    description="Upload an image and enter a label to see the segmentation mask applied. The model is nvidia/segformer-b0-finetuned-ade-512-512.",
    examples=[["images/hugging_face_ia.png", "person"]],
)

if __name__ == "__main__":
    iface.launch()
