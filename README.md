# Disaster Response Pipeline Web App
An end-end ETL pipeline utilising both an NLP and a Machine Learning Pipeline systems to create a web application that on typing a form of disaster-related message, categorizes it into categories for various disaster relief teams

-------

## Project Inspiration
Applying concepts and techniches of Data Engeering (ETL Pipelines, especially Machine Learning and NLP Pipelies) on a disaster messages dataset by Figure Eight to build a model for an API that classifies disaster messages.

------

## Repository Structure
![structure](https://user-images.githubusercontent.com/34100245/87125302-e4130480-c2a7-11ea-856d-68cd8de12cf0.PNG)

------

## Implementation
1. Setting up the database and model
   - Run the ETL pipeline that cleans the raw data (csv) and stores it in a database: ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```

2. 
