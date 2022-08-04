import text_model as tm

path_to_prompts = "prompts.txt"

with open(path_to_prompts, "r") as f:
    prompts = f.read()

def train_model(n=3):
    lm = tm.train_lm(prompts, 3)
    return lm

def generate(model, nletters=100):
    result = tm.generate_text(model, nletters)
    print(result)