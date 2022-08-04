import text_model as tm
import io 

path_to_prompts = r"prompts.txt"

# +

with open(path_to_prompts, encoding = "utf-8") as f:
    prompts = f.read()

# -

def train_model(n=3):
    lm = tm.train_lm(prompts, 3)
    return lm

def generate(model, nletters=100):
    result = tm.generate_text(model, nletters)
    print(result)

# +
#print(prompts)
# -


