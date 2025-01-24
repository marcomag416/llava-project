import loader
import utils
import pandas as pd
from tqdm.autonotebook import tqdm
import os
from math import ceil


def infer(model, promptgen, path, img_path, file_out, batch_size=1, check_point_every=-1, start_from=0):
    val_set = loader.loader(path, img_path)

    submission = []
    invalid_results = 0
    file_idx = 1

    for imgs, xs in tqdm(val_set.iter(batch_size=batch_size, start_from=start_from), total=ceil(val_set.get_len()//batch_size)):
        prompt = [promptgen.generate_prompt(x, template=0) for x in xs]
        result = model.infer(imgs, prompt)
        for idx, r in enumerate(result):
            answer, parsed_right = promptgen.parse_response(r, template=0)
            if not parsed_right:
                invalid_results += 1
                print(f"Invalid result. Prompt: {prompt}. Result: {result}")
            submission.append({"file_name": xs[idx]["file_name"], "answer": answer})
        utils.clean_cuda_cache()
        if check_point_every > 0 and len(submission) % check_point_every == 0:
            df = pd.DataFrame(submission)
            df.to_csv(f"{file_out}_{file_idx}.csv", index=False)
            file_idx += 1
            submission = []

    if len(submission) > 0:
        df = pd.DataFrame(submission)
        df.to_csv(f"{file_out}_{file_idx}.csv", index=False)
    print(f"Invalid results: {invalid_results}")

    return df


if __name__ == "__main__":
    from models import qwen2vl
    from prompts import Promptgenerator

    model = qwen2vl()
    promptgen = Promptgenerator(template=0, permutation={1:1, 2:2, 3:3, 4:4})

    infer(model, promptgen, "./dataset/validatio/validation_without_answers.csv", "./dataset/validation/images/", "./qwen2vl_t0", batch_size=1, check_point_every=50)