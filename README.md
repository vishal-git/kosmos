# Use `kosmos-2` to create image caption and identify main entities 

Use the code in the Notebook to select a random image from the `images` folder and run the `kosmos-2` model to get image caption and identify main themes/objects in that image.

Here's the model link on HuggingFace: https://huggingface.co/docs/transformers/main/model_doc/kosmos-2

This looks very promising to me for identifying tags/objects from an image because the other two alternatives, CLIP and YOLO, require a fixed set of target tags. While `kosmos-2` relies on text completion, which makes it open ended in terms of what types of objects, tags, or activities it can be identified within an image.
