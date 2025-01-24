import loader
import utils
import pandas as pd
from tqdm.autonotebook import tqdm
import os
from math import ceil
from prompts import Promptgenerator



def infer(model, promptgen, path, img_path, file_out, batch_size=1, check_point_every=-1, start_from=0, first_file_idx=1, filter=None):
    val_set = loader.loader(path, img_path, filter=filter)

    submission = []
    invalid_results = 0
    file_idx = first_file_idx

    for imgs, xs in tqdm(val_set.iter(batch_size=batch_size, start_from=start_from), total=ceil(val_set.get_len()//batch_size)):
        prompt = [promptgen.generate_prompt(x) for x in xs]
        result = model.infer(imgs, prompt)
        for idx, r in enumerate(result):
            answer, parsed_right = promptgen.parse_response(r)
            if not parsed_right:
                invalid_results += 1
                print(f"Invalid result. Prompt: {prompt}. Result: {result}")
            submission.append({"file_name": xs[idx]["file_name"], "answer": answer})
        utils.clean_cuda_cache()
        if check_point_every > 0 and len(submission) % (check_point_every*batch_size) == 0:
            df = pd.DataFrame(submission)
            df.to_csv(f"{file_out}_{file_idx}.csv", index=False)
            file_idx += 1
            submission = []

    if len(submission) > 0:
        df = pd.DataFrame(submission)
        df.to_csv(f"{file_out}_{file_idx}.csv", index=False)
    print(f"Invalid results: {invalid_results}")

    return df, file_idx

def infer_majority_voting(model, csv_path, img_path, root_name, batch_size=1):
    permutations = [
        {1:1, 2:2, 3:3, 4:4},
        {1:3, 2:1, 3:4, 4:2},
        {1:2, 2:3, 3:4, 4:1},
        {1:4, 2:2, 3:1, 4:3},
        {1:3, 2:4, 3:1, 4:2}
    ]
    for iternum in range(5):
        filter = None
        if(perm >= 3):
            files = []
            for file in range(iternum):
                files.append(f"{root_name}{str(file)}.csv")
            res, ties = implement_majority_voting(files)
            if len(ties) == 0:
                break
            filter = ties["file_name"]

        promptgen = Promptgenerator(template=0, permutation=permutations[iternum])

        df, num_files = infer(model, promptgen, csv_path, img_path, root_name+str(iternum), batch_size=batch_size, check_point_every=50, filter=filter) 
        results = pd.DataFrame({"file_name":[], "answer":[]})

        for t in range(1, num_files + 1):
            file = f"{root_name+str(iternum)}_{str(t)}.csv"
            tmp = pd.read_csv(file)
            results = pd.concat([results, tmp])

        if filter is not None:
            results = pd.concat([results, res])

        results.to_csv(f"{root_name+str(iternum)}.csv", index=False)
    
    return results, f"{root_name+str(iternum)}.csv"


if __name__ == "__main__":
    from models import qwen2vl
    from prompts import Promptgenerator
    import argparse

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--images_path", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--template", type=int, default=0, help="Template number")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--check_point_every", type=int, default=-1, help="Checkpoint every N batches")
    args = parser.parse_args()

    model = qwen2vl()
    promptgen = Promptgenerator(template=args.template, permutation={1:1, 2:2, 3:3, 4:4})

    infer(model, promptgen, args.csv_path, args.images_path, args.output_path, batch_size=args.batch_size, check_point_every=args.check_point_every)