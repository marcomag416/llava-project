from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch import cuda

if cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

train_path_img = "./dataset/train/images/"
val_path_img = "./dataset/validation/images/"



def load_img(name, split="train"):
    if split == "train":
        path = train_path_img
    elif split == "val":
        path = val_path_img
    else:
        raise ValueError("Invalid split name")
    return Image.open(path + name)


def load_model(model="qwen2vl"):
    if model == "qwen2vl":
        return Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
    else:
        raise ValueError("Invalid model name")


def load_processor(model="qwen2vl"):
    if model == "qwen2vl":
        min_pixels = 256*28*28
        max_pixels = 512*28*28 
        return AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        raise ValueError("Invalid model name")