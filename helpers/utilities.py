import numpy as np


def calculate_text_accuracy(pred_texts, target_texts):
    pred_chars = np.array(pred_texts)
    target_chars = np.array(target_texts)
    char_count = pred_chars.size
    token_count = target_chars.size // target_chars.shape[-1]
    char_matches = np.sum(pred_chars == target_chars).item()
    token_matches = np.sum(np.all(pred_chars == target_chars, axis=2)).item()
    return {
        "char_count": char_count,
        "token_count": token_count,
        "char_matches": char_matches,
        "token_matches": token_matches,
    }
