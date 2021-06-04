# Disaster Response Pipeline Web App
An end-end ETL pipeline utilising both an NLP and a Machine Learning Pipeline systems to create a web application that on typing a form of disaster-related message, categorizes it into categories for various disaster relief teams

-------

## Project Inspiration
Applying concepts and techniques of Data Engeering (ETL Pipelines, especially Machine Learning and NLP Pipelies) on a disaster messages dataset by Figure Eight to build a model for an API that classifies disaster messages.

------

## Repository Structure
![structure](https://user-images.githubusercontent.com/34100245/87125302-e4130480-c2a7-11ea-856d-68cd8de12cf0.PNG)

------

## Implementation
1. Setting up the database and model
   - Run the ETL pipeline that cleans (process_data.py) the raw data (disaster_messages.csv) and stores it in a database(DisasterResponse.db): ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```
   - Run ML pipeline that trains the classifier (train_classifier.py) and saves it (classifier.pkl): ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl ```
   - The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1EbkXfpzmQSO7tkAK8N164c-Fo2-bHoWK/view?usp=sharing)
   
2. Running the web app: ```python run.py```
3. Goto the link: [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

Below are some screenshots of how the web application looks:
![app1](https://user-images.githubusercontent.com/34100245/87163183-94066300-c2e4-11ea-8eca-954161b70370.PNG)

--------

## Example
Type a sample distress message: "We have a lot of problem at Delma 75 Avenue Albert Jode, those people need water and food"
![app3](https://user-images.githubusercontent.com/34100245/87163040-628d9780-c2e4-11ea-8aec-a1dee933cce6.png)

---

If you do find this repository useful, why not give a star and even let me know about it!

Feel free to express issues and feedback as well, cheers!

