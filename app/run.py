# %% [code]
#importing libraries
import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re

# %% [code]
app = Flask(__name__)

#tokenization function
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
#loading the data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
data = pd.read_sql_table('Processed_Message', engine)

# %% [code]
#loading the model
model = joblib.load("../model/classifier.pkl")

# %% [code]
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    count_genre = data.groupby('genre').count()['message']
    unique_genre = list(count_genre.index)

    # Show distribution of different category
    category = list(data.columns[4:])
    count_category = []
    for column_name in category:
        count_category.append(np.sum(data[column_name]))
    
    categories = data.iloc[:,4:]
    last_10_mean_categories = categories.mean().sort_values()[0:10]
    last_10_categories_names = list(last_10_mean_categories.index)
    
    
    # extract data exclude related
    categories = data.iloc[:,4:]
    top_10_mean_categories = categories.mean().sort_values(ascending=False)[0:10]
    top_10_categories_names = list(top_10_mean_categories.index)

    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=unique_genre,
                    y=count_genre
                )
            ],

            'layout': {
                'title': ' Message Genres Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_10_categories_names,
                    y=top_10_mean_categories
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=last_10_categories_names,
                    y=last_10_mean_categories
                )
            ],

            'layout': {
                'title': 'Least Searched Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=count_category
                )
            ],

            'layout': {
                'title': 'Message Categories Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
      
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# %% [code]
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(data.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# %% [code]
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
