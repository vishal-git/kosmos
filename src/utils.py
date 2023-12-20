from glob import glob
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from matplotlib import pyplot as plt


# randomly select a file from a directory
def random_file(path):
    files = glob(path)
    index = random.randrange(0, len(files))
    return files[index]


model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")


def get_tags(img, plot=False, prompt="<grounding>An image of"):
    """Get entities from an image"""
    image = Image.open(img)

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generate_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )

    generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    processed_text = processor.post_process_generation(
        generated_text, cleanup_and_extract=False
    )

    processed_text, entities = processor.post_process_generation(generated_text)
    if plot:
        # plot the image
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    return processed_text, entities


if __name__ == "__main__":
    img = random_file("./images/*.jpg")
    print(img)
    processed_text, entities = get_tags(img)
    print(processed_text)
    print(entities)
