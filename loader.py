import pandas as pd

class loader():
    def __init__(self, path, img_path):
        self.path = path
        self.img_path = img_path
        delf.df = train_set = pd.read_csv(path + "train_with_answers.csv")

    def load_img(self, i):
        return Image.open(self.img_path + self.df.iloc[i]["image_name"])

    def load_prompt(self, i):
        return self.df.iloc[i]

    def get_len(self):
        return len(self.df)

    def iter():
        for i in range(self.get_len()):
            yield self.load_img(i), self.load_prompt(i)