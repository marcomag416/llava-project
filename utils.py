from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch import cuda
import gc
import pandas as pd

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

def implement_majority_voting(file_names):
    merged = pd.read_csv(file_names[0])
    for idx, perm in enumerate(file_names[1:]):
        df = pd.read_csv(perm)
        merged = pd.merge(merged, df, on="file_name")
    
    # Function to check for ties
    def check_ties(row):
        modes = row.mode()
        return len(modes) > 1

    maj_voting = merged.mode(axis=1)
    maj_voting["file_name"] = merged["file_name"]
    ties = maj_voting[maj_voting.drop(columns="file_name").apply(check_ties, axis=1)]
    ties.drop(columns=0, inplace=True)
    maj_voting= maj_voting[maj_voting["file_name"].isin(ties["file_name"]).apply(lambda x: not x)]
    maj_voting["answer"] = maj_voting[0].apply(lambda x: int(x))

    return maj_voting[["file_name", "answer"]], ties