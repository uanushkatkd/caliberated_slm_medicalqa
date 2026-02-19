import re

def parse_answer_and_confidence(text: str):
    """
    Expects format:
    Answer: C
    Confidence: 0.73

    Returns:
        pred_option (str or None)
        confidence (float or None)
        valid (bool)
    """
    # Normalize
    t = text.strip()

    # Regex
    ans_match = re.search(r"Answer\s*:\s*([A-D])", t, re.IGNORECASE)
    conf_match = re.search(r"Confidence\s*:\s*([0-9]*\.?[0-9]+)", t, re.IGNORECASE)

    if not ans_match:
        return None, None, False

    pred = ans_match.group(1).upper()

    if conf_match:
        try:
            conf = float(conf_match.group(1))
            # clamp
            conf = max(1e-6, min(1 - 1e-6, conf))
        except ValueError:
            return pred, None, False
        return pred, conf, True

    # If confidence missing, still return answer but mark invalid format
    return pred, None, False
