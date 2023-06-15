# A Follow-up Analysis on Recipe Ratings using SciKitLearn
Authors: Xiaoyan Zhang (xiz115@ucsd.edu), Kay Qu(kqu@ucsd.edu)

# About this project
This is a paired EDA project which is originally assigned by UCSD's DSC80 offering during SP23. 
we will be focusing on cleaning, exploreing, and building a scikit pipeline to tranform and predict data originally scraped from [food.com](food.com). 
More information about the project's guidelines can be found [here](https://dsc80.com/project5/#overview).

# Part I: Analysis

## Prediction Problem

Our previous investigation question was on the relationship between Time (in minutues) to prepare a recipe and whether it correlates with the average rating that the recipe receives. We hypothesize that the longer time a recipe requires, the less rating it will receive. However, we could not establish a clear correlation between them. Therefore we would now like to build a **multiclass classification model** to predict recipes' ratings. We will be using `DecisionTreeClassifier` module which is built into `sklearn` to help us classify recipe ratings, categorized as discrete categorical data of 1-5.

The metric we will be using to assess our model is accuracy, since mean acccuracy function is built-in to the `DecisionTreeClassifier` model and for the purpose of this question, we are only interested in how accurately it can make predictions, and there are no specific restrictions against any specific errors, such as false negatives.


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

| submitted_year | interacted_year | minutes | n_steps | n_ingredients | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | rating |
| -------------- | --------------- | ------- | ------- | ------------- | ------------ | --------------- | ----------- | ------------- | -------------- | ------------------- | -------------------- | ------ |
| 2008           | 2008            | 40      | 10      | 9             | 138.4        | 10.0            | 50.0        | 3.0           | 3.0            | 19.0                | 6.0                  | 4.0    |
| 2011           | 2012            | 45      | 12      | 11            | 595.1        | 46.0            | 211.0       | 22.0          | 13.0           | 51.0                | 26.0                 | 5.0    |
| 2008           | 2008            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2009            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2013            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |


