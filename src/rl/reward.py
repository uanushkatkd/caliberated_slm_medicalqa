import math
from src.utils.parsing import parse_answer_and_confidence

EPS = 1e-6

def compute_reward(model_output_text: str, correct_option: str):
    pred, conf, valid = parse_answer_and_confidence(model_output_text)

    # Bad format penalty
    if not valid or conf is None or pred is None:
        return -1.0, False

    correct = (pred == correct_option)
    

    if correct:
        r = math.log(conf + EPS)
    else:
        r = math.log(1.0 - conf + EPS)

    return r, True
