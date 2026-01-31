from datasets import load_dataset

OPTION_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

def load_data(name,split):
    dataset=load_dataset(name,split=split)
    samples = []
    for ex in dataset:
        sample = {
            "question": ex["question"],
            "options": {
                "A": ex["opa"],
                "B": ex["opb"],
                "C": ex["opc"],
                "D": ex["opd"]
            },
            "correct_option": OPTION_MAP[ex["cop"]]
        }
        samples.append(sample)
    return samples
    
    
    