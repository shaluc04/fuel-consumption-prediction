## Predicting fuel consumption using Machine Learning

This notebook looks into using various Python based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting the fuel consumption of a vehicle in the 70s and 80s based on its attributes.

### What is Regression?

Regression analysis consists of a set of machine learning methods that allow us to predict a **continuous outcome variable (y)** based on the value of **one or multiple predictor variables (x)**.

I am going to follow the life-cycle of a machine learning problem and take the following approach to tackle the problem:

* **Problem Definition** - What problem am I trying to solve?
* **Data** - What data do I have?
* **Evaluation** - What defines success?
* **Features** - What features should I model?
  * Exploratory Data Analysis with Pandas and NumPy
  * Data Preparation using Sklearn
* **Modelling** - What kind of model should I use?
  * Setting up data transformation pipeline ( to make it easier to integrate it into main product).
  * Selecting and Training a few Machine Learning Models
* **Experimentation** - What have I tried/ what else can I try?
  * Cross-Validation and Hyperparameter Tuning using Sklearn
* **Deployment**
  * Deploying the Final Trained Model on Heroku via a Flask App
  

 #### Problem Definition
> How well can we predict the fuel consumption of a vehicle (in mpg), given it's characteristics?

#### Data
* The data is downloaded from UCI Machine Learning repository: [Auto MPG UCI Dataset](http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
* The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes.

#### Deployment

* I packaged the trained model into a **web service** using the **Flask** web framework, that, when given the data through a POST request, returns the MPG (Miles per Gallon) predictions as a response.
* Deployed the final trained model on **Heroku**: https://predict-fuel-mpg.herokuapp.com/


