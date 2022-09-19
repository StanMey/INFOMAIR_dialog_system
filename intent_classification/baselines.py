import json

from operator import itemgetter
from typing import Dict, List


class MostOccuringBaselinePredictor:
  def __init__(self):
    ...
  
  def train(self, X, y):
    self.most_occ = get_max_occurence(y)
  
  def predict(self, X):
    return most_occuring_baseline(X, max_occ=self.most_occ)


class RuleBasedBaselinePredictor:
  def __init__(self):
    ...
  
  def train(self, X, y):
    self.most_occ = get_max_occurence(y)
    # load in the keywords
    with open('./intent_classification/manual_rules.json') as f:
      self.keyword_rules = json.load(f)
  
  def predict(self, X):
    return rule_based_baseline(X, self.keyword_rules, default_dialog=self.most_occ)


def get_max_occurence(train: List[str]):
    """Get the most occuring label from the train list.

    args:
    train (list[str]): A list with all the labels.

    returns:
    str: The most occuring label.
    """
    return max(set(train), key=train.count)


def most_occuring_baseline(X: List[str], max_occ: str = "inform") -> List[str]:
    """A function that represents the first baseline.
    It assigns the majority class in the data (which is the 'inform' label)

    args:
    X (list[str]): A list with all the features.
    max_occ (str): Optional; The most occuring label ( default is 'inform')

    returns:
    List[str]: The predicted labels.
    """
    return [max_occ for _ in X]


def rule_based_baseline(X: List[str], key_words: Dict, default_dialog: str = "inform") -> List[str]:
  """A function that represents the second baseline.
  Uses a keyword based search to retrieve the corresponding label.
  When multiple keywords are correct, it chooses the largest corresponding keyword.
  
  args:
    X (list[str]): A list with all the features.
    keywords (Dict): A dictionary containing all the rules for the classification.
    default_dialog (str): Optional; The most occuring label ( default is 'inform').

  returns:
    List[str]: The predicted labels.
  """
  output = []

  # loop over the input sentences
  for sentence in X:
    
    # 
    possible_acts = []
    for dialog_act, keywords in key_words.items():

      # loop over all keywords connected to a dialog act
      for word in keywords:
        # check if the keywords exists in the sentence
        if word in sentence:
          # found a matching keyword, add it to the possible acts
          possible_acts.append((len(word), dialog_act))
  
    # check if any keyword has been found, if not take the default_dialog option
    if possible_acts:
      # select the dialog act with the most similarity
      output.append(max(possible_acts,key=itemgetter(0))[1])
      possible_acts = []
    else:
      output.append(default_dialog)
  
  return output