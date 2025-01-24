import pandas as pd
from PIL import Image


class loader():
    def __init__(self, path, img_path, filter=None):
        self.path = path
        self.img_path = img_path
        self.df = pd.read_csv(path)
        if filter is not None:
            self.df = self.df[self.df["file_name"].isin(filter)]

    def load_img(self, i):
        return Image.open(self.img_path + self.df.iloc[i]["file_name"])

    def load_prompt(self, i):
        return self.df.iloc[i]

    def get_len(self):
        return len(self.df)

    def iter(self, batch_size=1, start_from=0):
        M = (self.get_len() - start_from) % batch_size
        for i in range(start_from, self.get_len()-M, batch_size):
            yield [self.load_img(i+j) for j in range(batch_size)], [self.load_prompt(i+j) for j in range(batch_size)]
        if M > 0:
            yield [self.load_img(i+j) for j in range(M)], [self.load_prompt(i+j) for j in range(M)]