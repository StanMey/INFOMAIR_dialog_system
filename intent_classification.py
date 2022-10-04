import json
import pandas as pd
import spacy

from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, List
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


class IntentClassifier(ABC):
    """Abstract class for defining the behaviour of the Intent classifiers.
    """

    @abstractmethod
    def train(self, X, y):
        ...

    @abstractmethod
    def predict(self, x) -> str:
        ...

    @abstractmethod
    def predict_batch(self, X) -> List[str]:
        ...


class MostOccuringBaselinePredictor(IntentClassifier):
    """This classifier uses the most occuring class in the data to predict the intent.
    """
    # class variable
    name = "Most occuring baseline"

    def train(self, X, y):
        self.most_occ = get_max_occurence(y)

    def predict(self, x) -> str:
        return most_occuring_baseline([x], max_occ=self.most_occ)[0]

    def predict_batch(self, X) -> List[str]:
        return most_occuring_baseline(X, max_occ=self.most_occ)


class RuleBasedBaselinePredictor(IntentClassifier):
    """This classifier uses a keyword based approach to predict the intent.
    """
    # class variable
    name = "Rule-based baseline"

    def train(self, X, y):
        self.most_occ = get_max_occurence(y)
        # load in the keywords
        with open('./data/manual_rules.json') as f:
            self.keyword_rules = json.load(f)

    def predict(self, x) -> str:
        return rule_based_baseline([x], self.keyword_rules, default_dialog=self.most_occ)[0]

    def predict_batch(self, X) -> List[str]:
        return rule_based_baseline(X, self.keyword_rules, default_dialog=self.most_occ)


class NaiveBayesPredictor(IntentClassifier):
    """This classifier uses the Naive Bayes approach to predict the intent.
    """
    # class variable
    name = "Naive Bayes"

    def train(self, X, y) -> None:
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('model', MultinomialNB()),
            ])
        self.model.fit(X, y)
    
    def predict(self, x):
        return self.model.predict([x])[0]
    
    def predict_batch(self, X) -> List[str]:
        return self.model.predict(X)


class KNNPredictor(IntentClassifier):
    """This classifier uses a KNN and spaCy's word vectors to predict the intent.
    """
    # class variable
    name = "KNN"

    def train(self, X, y) -> None:
        # transform the sentences into vectors
        self.nlp = spacy.load("en_core_web_sm")
        X_train = [doc.vector for doc in self.nlp.pipe(X)]
        # setup the model
        self.neighbors = 5
        self.model = KNeighborsClassifier(n_neighbors=self.neighbors)
        # fit the model
        self.model.fit(X_train, y)
    
    def predict(self, x):
        return self.model.predict([self.nlp(x).vector])[0]
    
    def predict_batch(self, X) -> List[str]:
        return self.model.predict([doc.vector for doc in self.nlp.pipe(X)])


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


if __name__ == "__main__":
    # load in the training and test dialog data
    train_path = Path("./data/training_dialog.pkl")
    test_path = Path("./data/test_dialog.pkl")

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    # setup the data
    X_train, y_train = train_df["utterance_content"].to_list(), train_df["dialog_act"].to_list()
    X_test, y_test = test_df["utterance_content"].to_list(), test_df["dialog_act"].to_list()

    # analyse all the models above
    models = [MostOccuringBaselinePredictor(), RuleBasedBaselinePredictor(), NaiveBayesPredictor(), KNNPredictor()]
    results = []
    for model in models:
        # train the model
        model.train(X_train, y_train)
        # use the model to predict from the test set
        y_pred = model.predict_batch(X_test)
        # compute the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # compute the prf scores
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="micro")
        results.append((model.name, accuracy, precision, recall, fscore))
    
    headers = ["Method", "accuracy", "precision", "recall", "fscore"]
    df_models = pd.DataFrame(results, columns=headers)
    print(df_models)