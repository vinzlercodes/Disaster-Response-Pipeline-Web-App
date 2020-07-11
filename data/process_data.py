# %% [code]
'''
importing libraries
'''
import sys
import pandas as pd
from sqlalchemy import create_engine

# %% [code]
'''
loading the data 
'''
def loading_data(messages_path, categories_path):

    try:
        messages = pd.read_csv(messages_path)
        categories = pd.read_csv(categories_path, sep = ',')
        data = pd.merge(messages, categories, on = 'id')
        return data
    except:
        print("file not found!")

# %% [code]
'''
cleaning the data
'''
def cleaning_data(data):

    data_categories = data['categories'].str.split(';',expand = True)
    data_row = data_categories.loc[0]
    category_cols = data_row.apply(lambda x: x[:-2]).values.tolist()
    data_categories.columns = category_cols

    '''
    changing the cloumn shape
    '''
    for column in data_categories:
        
        '''
        setting each value to the last character of the string
        '''
        
        data_categories[column] = data_categories[column].astype(str).str[-1]
        
        '''
        converting the column from string to numeric
        '''
        
        data_categories[column] = data_categories[column].astype(int)
    
    data.drop('categories',axis = 1, inplace = True)
    data = pd.concat([data, data_categories], axis = 1)
    if data.duplicated().sum() != 0:
        data.drop_duplicates(inplace = True)
        
    return data

# %% [code]
'''
saving the cleaned data to a SQLite Database
'''
def store_data(data, database_name):
    engine = create_engine('sqlite:///'+database_name)
    data.to_sql('Processed_Message', engine, index=False)

# %% [code]
def main():
    if len(sys.argv) == 4:

        messages_path, categories_path, database_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_path, categories_path))
        data = loading_data(messages_path, categories_path)

        print('Cleaning data...')
        data = cleaning_data(data)
        
        print('Saving data...\n    DATABASE: {}'.format(database_path))
        store_data(data, database_path)
        
        print('Cleaned the data and saved to the database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'data/disaster_messages.csv data/disaster_categories.csv '\
              'data/DisasterResponse.db')


if __name__ == '__main__':
    main()
