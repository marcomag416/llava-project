from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch import cuda
import gc

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

def clean_cuda_cache():
    cuda.empty_cache()
    gc.collect()