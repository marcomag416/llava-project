import loader
import prompts
import utils
import pandas as pd
from tqdm.autonotebook import tqdm


def infer(model, path, img_path, file_out, batch_size=1):
    val_set = loader.loader(path, img_path)

    submission = []
    invalid_results = 0

    for imgs, xs in tqdm(val_set.iter(batch_size=batch_size), total=val_set.get_len()//batch_size +1):
        prompt = [prompts.generate_prompt(x, template=0) for x in xs]
        result = model.infer(imgs, prompt)
        for r in result:
            answer, parsed_right = prompts.parse_response(r, template=0)
            if not parsed_right:
                invalid_results += 1
                print(f"Invalid result. Prompt: {prompt}. Result: {result}")
            submission.append({"file_name": x["file_name"], "answer": answer})
        utils.clean_cuda_cache()

    df = pd.DataFrame(submission)
    df.to_csv(file_out, index=False)
    print(f"Invalid results: {invalid_results}")

    return df