import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB


# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

class Model():
    nb = GaussianNB()
    bow_converter = None
    tfidf_transformer = None

def process_news(news):
    # Convert to lower case
    news = news.lower()
    
    # Replace contractions with their longer forms 
    news = news.split()
    new_news = []
    for word in news:
        if word in contractions:
            new_news.append(contractions[word])
        else:
            new_news.append(word)
    news = " ".join(new_news)
    
    # Remove stop words
    news = news.split()
    stops = set(stopwords.words("english"))
    new_news = []
    for word in news:
        if word not in stops:
            new_news.append(word)
    news = " ".join(new_news)

    # Tokenize each word
    news =  nltk.WordPunctTokenizer().tokenize(news)
        
    return news

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.bow_converter = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    model.tfidf_transformer = text.TfidfTransformer(norm=None)
    
    X_train = list(map(process_news, X_train))
    X_train_bow = model.bow_converter.fit_transform(X_train)
    X_train_tfidf = model.tfidf_transformer.fit_transform(X_train_bow)
    model.nb.fit(X_train_tfidf.toarray(), y_train)

def predict(model, X_test):
    ''' TODO: make your prediction here '''
    X_test = list(map(process_news, X_test))
    X_test_bow = model.bow_converter.transform(X_test)
    X_test_tfidf = model.tfidf_transformer.transform(X_test_bow)
    y_pred = model.nb.predict(X_test_tfidf.toarray())
    return y_pred

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('fulltrain.csv', header=0, names=['Verdict', 'News'])
    mask = train['Verdict'] < 3
    train = train.loc[mask]
    X_train = train['News']
    y_train = train['Verdict']
    model = Model()
    train_model(model, X_train, y_train)
    y_pred = predict(model, X_train)
    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('balancedtest.csv', header=0, names=['Verdict', 'News'])
    X_test = test['News']
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, "news_predicted.csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()