import re

def generate_prompt(x, template=0, permutation={1:1, 2:2, 3:3, 4:4}):
    options = [x["option1"], x["option2"], x["option3"], x["option4"]]
    templates = [
        f"""Question: {x["question"]}
Answer options:
A. {options[permutation[1]-1]}
B. {options[permutation[2]-1]}
C. {options[permutation[3]-1]}
D. {options[permutation[4]-1]}
Answer with the option’s letter from the given choices directly.""",

    ]
    return templates[template]

def parse_response(response, template=0, permutation={1:1, 2:2, 3:3, 4:4}):
    templates = [
        {"A":1, "B":2, "C":3, "D":4},
    ]

    regex = [
        r'[^A-D]',
    ]

    try:
        ind = templates[template][re.sub(regex[template], '', response.upper())]
        out = permutation[ind]
    except:
        return 1, False
    return out, True


