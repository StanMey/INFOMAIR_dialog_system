import Levenshtein
import numpy as np
import re

from typing import List, Union



def find_preference(options: List[str], user_utterance: str, max_levensthein: int = 3) -> Union[str, None]:
    """Finds the preference in a sentence out of a list with options and the levensthein distance.

    Args:
        options (List[str]): a list of options for which we search in the text.
        user_utterance (str): The input from the user.
        max_levensthein (int, optional): the length of the levensthein distance. Defaults to 3.

    Returns:
        Union[str, None]: returns either the best found match or None.
    """
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