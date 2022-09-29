from tkinter.tix import Tree
import Levenshtein
import numpy as np

from typing import List, Union



def find_preference(options: List[str], user_utterance: str, patterns: List[str], max_levenshtein: int = 0) -> Union[str, None]:
    """Finds the preference in a sentence out of a list with options and the levenshtein distance.

    Args:
        options (List[str]): a list of options for which we search in the text.
        user_utterance (str): The input from the user.
        max_levenshtein (int, optional): the length of the levenshtein distance, if this is 0 than a dynamic levenshtein value is used. Defaults to 0.

    Returns:
        Union[str, None]: returns either the best found match or None.
    """
    smallest_distance = np.inf
    best: str = None
    current = False
    pref = None
    prev_word = ""
    for option in options:
        if option in user_utterance:
            # an exact match has been found
            pref = option
            current = True
            return pref
            # return option
            break

        # else:
        #     for word in user_utterance.split():
        #         distance = Levenshtein.distance(word, option)
        #         if distance < smallest_distance:
        #             # if not defined
        #             if not max_levenshtein:
        #                 max_levenshtein = len(word)/2
        #             smallest_distance = distance
        #             best = option

    for word in user_utterance.split():
        if word in patterns:
            break
        prev_word = word
    # not completely certain
    for option in options:
        distance = Levenshtein.distance(prev_word, option)
        if distance < smallest_distance:
            # if not defined
            if max_levenshtein == 0:
                max_levenshtein = len(prev_word)/2
            smallest_distance = distance
            best = option
    if best and not current:
        if smallest_distance < max_levenshtein:
            pref = best
    return pref


