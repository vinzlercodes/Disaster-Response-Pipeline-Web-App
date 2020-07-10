# %% [code]
#importing the libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
from sqlalchemy import create_engine
from sklearn.externals import joblib
import re

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% [code]
#defining the database function
def loading_data(database_path):
    engine = create_engine('sqlite:///'+database_path)
    data = pd.read_sql('Processed_Message',engine)
    X = data['message']
    y = data.iloc[:,4:]
    category_labels = list(data.columns[4:])
    return X,y,category_labels

# %% [code]
#defining the toenization function
#function returns the lemmatized labels
def tokenizing(text):

    # normalizing, tokenizing and lemmatizing the sentence
    text_input = text.lower()
    text_input = re.sub(r"[^a-zA-Z0-9]", " ", text_input)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_tokens = []
    
    for token in tokens:
        cleaned_tokens = lemmatizer.lemmatize(token).lower().strip()
        lemmatized_tokens.append(cleaned_tokens)
        
        
    return lemmatized_tokens

# %% [code]
#model build function 
#function utilises grid search method
def building_model():

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenizing)),('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    params = {'vect__stop_words': ['english',None],'tfidf__use_idf' :[True, False]}

    cv = GridSearchCV(pipeline, params, n_jobs=-1)
    return cv

# %% [code]
#model evaluation function
def evaluating_model(model, X_test, y_test, category_labels):
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test.columns): 
        print('\t\t',col,'\t\t')
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))

# %% [code]
def saving_model(model, model_path):
    joblib.dump(model.best_estimator_, model_path)

# %% [code]
def main():
    if len(sys.argv) == 3:
        database_path, model_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_path))
        X, y, category_labels = loading_data(database_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = building_model()

        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluating_model(model, X_test, y_test, category_labels)

        print('Saving model...\n    MODEL: {}'.format(model_path))
        saving_model(model, model_path)

        print('Trained model has been saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()