import Levenshtein
import numpy as np
import re


def find_preference(options, user_utterance, max_levensthein: int = 3):
    smallest_distance = np.inf
    best: str = None
    current = False
    pref = None
    for option in options:
        if re.search(r"^.*" + re.escape(option) + r".*", user_utterance):
            # an exact match has been found
            pref = option
            current = True
            break
        else:
            for word in user_utterance.split():
                distance = Levenshtein.distance(word, option)
                if distance < smallest_distance:
                    smallest_distance = distance
                    best = option
    if best and not current:
        if smallest_distance < max_levensthein:
            pref = best
    return pref