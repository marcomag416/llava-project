import loader
import utils
import pandas as pd
from tqdm.autonotebook import tqdm
import os


def infer(model, promptgen, path, img_path, file_out, batch_size=1, check_point_every=-1):
    val_set = loader.loader(path, img_path)

    submission = []
    invalid_results = 0
    file_idx = 1

    for imgs, xs in tqdm(val_set.iter(batch_size=batch_size), total=val_set.get_len()//batch_size +1):
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

    df = pd.DataFrame(submission)
    df.to_csv(f"{file_out}_{file_idx}.csv", index=False)
    print(f"Invalid results: {invalid_results}")

    return df