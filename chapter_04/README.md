# Chapter 4. Using Hugging Face for Computer Vision Tasks

The computer vision models hosted on Hugging Face are grouped into the following tasks:

- Object Detection
- Image Classification
- Image Segmentation
- Video Classification
- Depth Estimation
- Image-to-Image
- Unconditional Image Generation
- Zero-Shot Image Classification

## Object Detection

> The primary goal of object detection is to not only classify the objects in the image
> or video but also to determine their precise positions by drawing bounding boxes around them.

## Image Segmentation

> Image segmentation is a technique that envolves separating an image into multiple segments,
> or regions. Each segment corresponds to a particular object of interest. Using image segmentation,
> you can analyze an image and extract valuable information from it.

Here are some of the use cases of this technique:

- Medical imaging - used to identify and segment tumors in MRI or CT scans
- Object detection and recognition
- Document processing - used to segment text regions in scanned documents
- Biometrics - used to identify and localize faces in images or video frames

Please explore [computer_vision_tasks.ipynb](computer_vision_tasks.ipynb) Jupyter Notebook for an example of
segmenting different objects (plane, person, wall) from an image using `nvidia/segformer-b0-finetuned-ade-512-512`
model ([link](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)).
