
def generate_prompt(x, template=0):
    templates = [
        f"""Question: {x["question"]}
Answer options:
A. {x["option1"]}
B. {x["option2"]}
C. {x["option3"]}
D. {x["option4"]}
Answer with the option’s letter from the given choices directly.""",

    ]
    return templates[template]

def parse_response(response, template=0):
    templates = [
        {"A":1, "B":2, "C":3, "D":4},
    ]

    return templates[template][response]


