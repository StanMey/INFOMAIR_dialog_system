from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesPredictor:
    def __init__(self):
        ...

    def train(self, X, y):
        model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('model', MultinomialNB()),
            ])

        model.fit(X, y)
        self.model = model
        return model

    def predict(self, X):
        return self.model.predict(X)