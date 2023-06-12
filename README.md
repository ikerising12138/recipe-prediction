# Chop Chop! An Analysis on Recipe Preparation Times and Ratings
Authors: Xiaoyan Zhang (xiz115@ucsd.edu), Kay Qu(kqu@ucsd.edu)

# About this project
This is a paired EDA project which is originally assigned by UCSD's DSC80 offering during SP23. 
we will be focusing on cleaning, exploreing, and visualizing data scraped from [food.com](food.com). 
More information about the project's guidelines can be found [here](https://dsc80.com/project3/recipes-and-ratings/).

# Part I: Introduction

## Research Question

Follwoing up on our previous investigation on **the relationship between Time (in minutues) to prepare a recipe and whether it correlates with the average rating that the recipe receives.** We hypothesize that the longer time a recipe requires, the less rating it will receive. However, we could not establish a clear correlation between 

This is a question worth investigating because through analyzing the relationship between cooking time and ratings, we can potentially look into whether this trend (or the lack thereof) is related to years when the users provided their ratings. We can then preform meta-analysis on whether users are becoming less fond of lengthy recipes as time passes. This can be a future research topic that is worth investigating. 

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
