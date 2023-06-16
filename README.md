# A Follow-up Analysis on Recipe Rating Classification using SciKitLearn
Authors: Xiaoyan Zhang (xiz115@ucsd.edu), Kay Qu(kqu@ucsd.edu)

# About this project
This is a paired data exploration project which is originally assigned by UCSD's DSC80 offering during SP23. 
We will be focusing on cleaning, exploring, and building a scikit pipeline to tranform and predict data originally scraped from [food.com](food.com). 
More information about the project's guidelines can be found [here](https://dsc80.com/project5/#overview).


# Framing the Problem

Our previous investigation question was on the relationship between 'minutes' to prepare a recipe and the average 'rating' that the recipe receives. We hypothesized that the longer time a recipe requires, the lower rating it will receive. However, we could not establish a clear correlation between them. Therefore, in this project, we aim to build a **multiclass classification model** to predict recipes' ratings. 


## About our data
Our data is derived from [food.com](https://www.food.com), originally scraped and used by them. The raw csv files can be found [here](https://drive.google.com/file/d/1kIbMz6jlhleiZ9_3QthmUnifoSds_2EI/view).

Below are the brief description of the raw data that we are going to perform cleaning on: 

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

### raw_interactions (731927 rows, 5 columns)

| Column       | Description           |
|--------------|-----------------------|
| 'user_id'    | User's ID               |
| 'recipe_id'  | Recipe's ID             |
| 'date'       | Date when the rating was posted   |
| 'rating'     | rating, scaling from 1-5          |
| 'review'     | user's review message about the recipe           |

### Data Cleaning Process
We've splitted the `nutrition` column into 7 separate columns as each value in the `nutrition` column represents a nutrient in the recipe: `calories (#)`, `total fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated fat (PDV)`, and `carbohydrates (PDV)`.

Then, we wanted to extract the year information from the `submitted` column in `raw_recipes` dataset and the year information from the `date` column in `raw_interactions` dataset. We transformed the `submitted` column to `submitted_year` which represents the year when a recipe was posted, and the `date` column to `interacted_date` which represents the year when a rating for a recipe was posted.
We finally ended up selecting the following 13 columns that will be needed in this project.

### Cleaned dataframe (234428 rows, 13 columns)

<div markdown="1" style="
    display: block;
    /* background-color: blue; */
    width: 100%;
    overflow-x: auto
">
    
| submitted_year | interacted_year | minutes | n_steps | n_ingredients | calories (#) | total fat (PDV) | sugar (PDV) | sodium (PDV) | protein (PDV) | saturated fat (PDV) | carbohydrates (PDV) | rating |
| -------------- | --------------- | ------- | ------- | ------------- | ------------ | --------------- | ----------- | ------------- | -------------- | ------------------- | -------------------- | ------ |
| 2008           | 2008            | 40      | 10      | 9             | 138.4        | 10.0            | 50.0        | 3.0           | 3.0            | 19.0                | 6.0                  | 4.0    |
| 2011           | 2012            | 45      | 12      | 11            | 595.1        | 46.0            | 211.0       | 22.0          | 13.0           | 51.0                | 26.0                 | 5.0    |
| 2008           | 2008            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2009            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |
| 2008           | 2013            | 40      | 6       | 9             | 194.8        | 20.0            | 6.0         | 32.0          | 22.0           | 36.0                | 3.0                  | 5.0    |

</div>

We will be using `DecisionTreeClassifier` module which is built into `sklearn` to help us classify recipe ratings, categorized as discrete categorical data of 1-5. And we use `rating` as our response variable. In project 3, we used it as our targeted column and studied its relationship with other columns like `minutes`, `n_steps`, and `n_ingredients`. Now we take a step further and aim at being able to classification patterns based on information of other data entries.

The metric we will be using to assess our model is **accuracy**, which can be calculated by the score function built-in to the `DecisionTreeClassifier` model. We are choosing this metric because we are only interested in how accurately our model can make predictions on `rating`, and there are no serious restrictions against any specific errors, such as false negatives that matter more in a disease classification setting and false positives that matter more in a law setting.

At time of prediction, we are able to use all of our data aside from the `rating` column in our cleaned dataframe to perform analysis, since `rating` is our response variable.


# Baseline Model

Before we train our model, we need to first define a training set and a testing set. We used `train_test_split` to randomly split our cleaned dataset into seen data (`X_train` and `y_train`) and unseen data (`X_test` and `y_test`) with a `test_size` of 0.25. In later practices of GridSearchCV, part of the seen data will be used as validation data. Though, after we've obtained our training and test dataset, they will remain the same throughout the project so that we can fairly compare the performance of our baseline and final model with no disturbance from different seen and unseen data.

Again, since we are dealing with a classification problem, we will mainly be using `DecisionTreeClassifier`. The two features we've engineered in this base line model are `submitted_year` and `interacted_year`. Both of them are treated as dicrete categorical features, and therefore we decided to use `OneHotEncoder` to transform these two columns. The remainder columns are numerical features, so we leave them as-is for now and passthrough them to our `DecisionTreeClassifier`.

This process is achieved via `Pipeline` module, in which it allows us to first apply necessary preprocessings to our dataframe and then apply the classifer to our target data.

### Performace and Interpretation

The result of our `DecisionTreeClassifier` showed an accuracy score on training data of about 0.9046, and an accuracy score on test data of 0.5832. This is not an ideal result, since a much higher training-accuracy and a low testing-accuracy score indicate that our model is overfitting the training data. Thus, we need to make necessary adjustments to resolve the serious overfitting issue in our final model.


# Final Model

### Selecting More Features

For the final model, we've decided to engineer more features to add upon our baseline model. Namely, we chose `n_steps`, `n_ingredients`, and `minutes` to perform feature engineering.

We have visualized their distributionn with regard to `rating`:

<iframe src="assets/Scatter_Plot_N_Steps.html" width=550 height=450 frameBorder=0></iframe>

<iframe src="assets/Scatter_Plot_N_Ingre.html" width=550 height=450 frameBorder=0></iframe>

<iframe src="assets/Scatter_Plot_Minutes.html" width=550 height=450 frameBorder=0></iframe>

From the plots above, we can see that there seems to be some association between `n_steps`, `n_ingredients`, `minutes` and `rating`, some with visually bimodal trends.

We believe these features will help our model to make better classification decisions partially due to their bimodal relationships with the ratings, and also because we believe that these features are associated with a high-level concept of 'tediousness' of a recipe, as we hypothesized that either recipes that are easy to make or recipes with a delicate preparation process will gain higher ratings. 

### Engineering features

Now that we've decided on what features to use, we wanted to process these data so that they help with better prediction results. 

For `n_steps` and `n_ingredients`, the former ranges from 0 to about 100 steps, and the latter ranges from 0 to 40 ingredients. Because of their manageable range and absence of outliers, we decided to use `KBinsDiscretizer` to transform their numerical values into discrete bins of 10 and 5, respectively. This process allows us to better handle non-linear relationships (in case these features are not linearly related to rating) and reduce overfitting as the complexity of the original data is simplified.

Though, while looking at the distribution of `minutes`, we're surprised by its huge range of data as its percentile distribution is as follows:

| Percentile       | Minutes    |
| ---------------- | ---------- |
| 25th percentile  | 20.0       |
| 50th percentile  | 35.0       |
| 75th percentile  | 60.0       |
| 100th percentile | 1051200.0  |

The huge gap between 75th percentile and the maximum level hints the existence of outliers in the `minutes` column. Thus, we carried out further explorations on the `minutes` column. Using `raw_recipes` dataset to find what recipes could take so long to finish and testify if they are outliers, we filtered for recipes that take between 2 to 3 weeks to finish, which include marinated cuisines like `kimchi` and `2 week sweet pickles`, `weekly made bread`, and liquor like `limoncello` and `plum liquor`. Among recipes that take more than 3 weeks and even months and years to finish, a huge proportion are homemade liquor and wine including `coffee flavored liqueur`, `homemade kahlua`, `homemade fruit liquers`, `word stew`, etc.

Thus, our point is that we do need to account for these seemingly outliers on the higher end of the spectrum because it's likely that people are more inclined to give high rates for recipes that take them extremely lengthy time to finish. Considering the fact that using bins or quantiles can overly simply or generalize the trend, we decided to apply `StandardScalar` to transform the `minutes` column to standardized units so as to better capture individual differences between `minutes` data points.


Now that we have our `submitted_year` and `interacted_year` through OneHotEncoder,
`n_steps` and `n_ingredients` through KBinsDiscretizer, `minutes` thorugh StandardScalar, and the other columns passthroughed, we can first apply these preprocessing steps so we can then move on to our next step: finding the optimal combination of hyperparameters.


### Choosing Appropriate Model and Hyperparameters

Our project uses `DecisionTreeClassifier` model for prediction. Its documentation can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

We chose a decision tree model over a linear regression model because we are dealing with multi-class classification in this project. Of all the different decision tree models, we picked `DecisionTreeClassifier` as our model because it is fairly simple to use, is able to take in preprocessed data we provide, and works well with `GridSearchCV` when looking for the best combination of hyperparameters (more on `GridSearchCV` below).


In order to prevent overfitting and look for best parameters suitable for our `DecisionTreeClassifier` object, we will be using `GridSearchCV` to perform a k-fold cross validation procedure. For simplicity, we chose a k value of 5.

After we input a variety of hyperparameters as inputs and fitted training data, GridSearchCV outputted the following as the best combination of hyperparameters:

```
{'decision_tree__criterion': 'entropy', 'decision_tree__max_depth': 3,'decision_tree__min_samples_split': 2}
```

Then we used the best hyperparameters and trained the final model on the whole dataset because part of the training data was used as validation data in the process of GridSearchCV. Now we calclulated our final model's accuracy performance again.

The result of our final `DecisionTreeClassifier` model produced a mean accuracy on training data of 0.7247, and a mean accuracy on testing data of 0.7209. Comparing the current accuracy on test data with 0.5832 from the baseline model, we saw a significant increase in accuracy when our model was tested on unseen data. We can visualize our result with a confusion matrix below:

<iframe src="assets/confusion_matrix.html" width=500 height=450 frameBorder=0></iframe>
<iframe src="assets/confusion_matrix.png" width=550 height=450 frameBorder=0></iframe>

|   0 |   1 |   2 |   3 |   4 |    5 |\n|----:|----:|----:|----:|----:|-----:|\n|  12 |   0 |   0 |   0 |   0 | 3869 |\n|   3 |   0 |   0 |   0 |   0 |  695 |\n|   1 |   0 |   0 |   0 |   0 |  597 |\n|   3 |   0 |   0 |   0 |   0 | 1729 |\n|   5 |   0 |   0 |   0 |   0 | 9431 |

Despite the improvement, we should also be aware of the fact that our model was only classifying `rating` of either 0 or 5. This is because in the original dataset, 72.4% of all ratings are 5, 15.9% of all ratings are 4 (so high ratings consist of over 88% of all rating scores) and 6.41% of all ratings are 0. So just guessing 0 or higher rating like 5 was not due to defects of our model but instead due to the serious class imbalance issue of the original dataset. A more in-depth analysis in future studies may be required to resolve this issue.

# Fairness Analysis

In this section, we would like to perform a permutation test for fairness analysis by binarizing our `n_steps` column into two groups: 

Group X: recipes with number of steps smaller than or equal to 15

Group Y: recipes with number of steps larger than 15

Now, we will state our hypothesis: 

Null hypothesis: Our model is fair. Its accuracy for recipes with steps smaller than 15 and recipes with larger number of steps are roughly the same, and any differences are due to random chance.

Alternative Hypothesis: Our model is unfair. Its accuracy for steps smaller than 15 is higher than that for recipes with larger number of steps.

For our test statistic, we will be using signed difference in accuracy score between recipe with `n_steps` smaller than 15 and recipes with larger n_steps. And we chose a significance level of 5%.

For our evaluation metric, we will still be using accuracy for consistency.

The result of the performing the permutation test is shown below:

<iframe src="assets/Histogram.html" width=500 height=450 frameBorder=0></iframe>

From the permutation test, we derived observed difference in mean to be roughly 0.002983.

The resulting p-value is 0.23, which means that we failed to reject our null hypothesis under 5% significance level. Though we cannot make an absolute statement, there's a chance that our final model is fair as we are unable to determine that there is a significant difference between the accuracy score performance of the two binarized groups.

