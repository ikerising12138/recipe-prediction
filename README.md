# A Follow-up Analysis on Recipe Ratings using SciKitLearn
Authors: Xiaoyan Zhang (xiz115@ucsd.edu), Kay Qu(kqu@ucsd.edu)

# About this project
This is a paired data exploration project which is originally assigned by UCSD's DSC80 offering during SP23. 
we will be focusing on cleaning, exploreing, and building a scikit pipeline to tranform and predict data originally scraped from [food.com](food.com). 
More information about the project's guidelines can be found [here](https://dsc80.com/project5/#overview).

# Part I: Analysis

## Prediction Problem

Our previous investigation question was on the relationship between Time (in minutues) to prepare a recipe and whether it correlates with the average rating that the recipe receives. We hypothesize that the longer time a recipe requires, the less rating it will receive. However, we could not establish a clear correlation between them. Therefore we would now like to build a **multiclass classification model** to predict recipes' ratings. 


## About our data
Our data is derived from [food.com](https://www.food.com), originally scraped and used by them. The raw csv files can be found [here](https://drive.google.com/file/d/1kIbMz6jlhleiZ9_3QthmUnifoSds_2EI/view).

Below are the breif description of the raw data that we are going to perform cleaning on: 

### raw_recipes (83782 rows, 12 columns)

| Column           | Description                                          |
|------------------|------------------------------------------------------|
| 'name'           | Name of the recipe                                          |
| 'id'             | Recipe's ID                                            |
| 'minutes'        | Time (in minutues) to prepare recipe                            |
| 'contributor_id' | Recipe contributor's ID                     |
| 'submitted'      | Date when the recipe was submitted                            |
| 'tags'           | Categorical tags for recipe                             |
| 'nutrition'      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| 'n_steps'        | Number of steps in recipe                            |
| 'steps'          | Description for 'n_steps', in order                       |
| 'description'    | additional description provided by user                             |

### raw_ratings (731927 rows, 5 columns)

| Column       | Description           |
|--------------|-----------------------|
| 'user_id'    | User's ID               |
| 'recipe_id'  | Recipe's ID             |
| 'date'       | Date when the rating was posted   |
| 'rating'     | rating, scaling from 1-5          |
| 'review'     | user's review message about the recipe           |

We transformed `submitted` column to `submitted_year`, `date` column to `interacted_date`, as per the data source. Then, we exapnded the nutrition column by assigning each nutrition to its separate columns:

### transformed dataframe

<div class="table-container">
  <table>
    | submitted_year | interacted_year | minutes | n_steps | n_ingredients | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | rating |
| -------------- | --------------- | ------- | ------- | ------------- | ------------ | --------------- | ----------- | ------------- | -------------- | ------------------- | -------------------- | ------ |
| 2008           | 2008            | 40      | 10      | 9             | 138.4        | 10.0            | 50.0        | 3.0           | 3.0            | 19.0                | 6.0                  | 4.0    |
| 2011           | 2012            | 45      | 12      | 11            | 595.1        | 46.0            | 211.0       | 22.0          | 13.0           | 51.0                | 26.0                 | 5.0    |
| 2008           | 2008            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2009            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2013            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
    
  </table>
</div>


| submitted_year | interacted_year | minutes | n_steps | n_ingredients | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | rating |
| -------------- | --------------- | ------- | ------- | ------------- | ------------ | --------------- | ----------- | ------------- | -------------- | ------------------- | -------------------- | ------ |
| 2008           | 2008            | 40      | 10      | 9             | 138.4        | 10.0            | 50.0        | 3.0           | 3.0            | 19.0                | 6.0                  | 4.0    |
| 2011           | 2012            | 45      | 12      | 11            | 595.1        | 46.0            | 211.0       | 22.0          | 13.0           | 51.0                | 26.0                 | 5.0    |
| 2008           | 2008            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2009            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2013            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |



We will be using `DecisionTreeClassifier` module which is built into `sklearn` to help us classify recipe ratings, categorized as discrete categorical data of 1-5. We use ratings as our response variable mainly because we want to be able to detect classification patterns from the other data entries

The metric we will be using to assess our model is accuracy, since mean acccuracy function is built-in to the `DecisionTreeClassifier` model and for the purpose of this question, we are only interested in how accurately it can make predictions, and there are no specific restrictions against any specific errors, such as false negatives.

At time of prediction, we are able to use all of our data aside from the `rating` column from our transformed dataframe to perform analysis, since `rating` is our response variable.


## Baseline Model

Before we train our model, we need to first define a training set and a testing set. We used 

Again, since we are dealing with a classification problem, we will mainly be using `DecisionTreeClassifier`. The features we will be using is `submitted_year` and `interacted_year`, and both of them are treated as dicrete categorical features and therefore we decided to use `OneHotEncoder` to vectoeize these two columns. The remainder columns are kept and passed down to our `DecisionTreeClassifier`.

This process is achieved via `Pipeline` module, in which it allows us to first apply necessary preprocessings to our dataframe and then apply the classifer to our target data. 

### Performace and Interpretation

The result of our `DecisionTreeClassifier` showed a mean accuracy on training data of 0.9039, and a mean accuracy on testing data of 0.5889. This is not an ideal result, since a high training-accuracy and a low testing-accuracy indicate that our model is overfitting the training data. We need to make necessary adjustments to account for overfitting. 


## Final Model

### Selecting More Feature

For the final model, we've decided to select more features and added them into our model. Namely, we chose `n_steps`, `n_ingredients`, and `minutes`. We have visualized their distributionn with regard to the ratings:

<iframe src="assets/Scatter_Plot_N_Steps.html" width=600 height=550 frameBorder=0></iframe>

<iframe src="assets/Scatter_Plot_N_Ingre.html" width=600 height=550 frameBorder=0></iframe>

<iframe src="assets/Scatter_Plot_Minutes.html" width=600 height=550 frameBorder=0></iframe>

From the plots above, we can see that there seems to be some association between `n_steps`, `n_ingredients`, `minutes` and ratings, some with a visually bimodal trends. 

We believe these features helped our model to make better predictions partially due to their bimodal relationships with the ratings, and also because we believe that these features are associated with a high-level concept of 'tediousness' of a recipe, as we hypothesized that either recipes that are easy to make, or have a delicate preparation process will gain higher ratings. 

### Engineering features

Now that we've decided on what features to use, we wanted to process these data so that they help with better prediction results.

For  `n_steps` and  `n_ingredients`, we used `KBinsDiscretizer` to transform the numerical values into discrete bins. 

For `minutes`, we've decided to apply `StandardScalar` to transform them to standardized units. This process enables us to better handle outliers, since for `minutes` column, the percentile distributuion is 
25th percentile: 20.0
50th percentile: 35.0
75th percentile: 60.0
100th percentile: 1051200.0
, 
thus we wanted to account for the outliers on the higher end of the spectrum. 


### Choosing Appropriate Hyperparameters





