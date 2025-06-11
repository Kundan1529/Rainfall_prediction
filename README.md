# 🌧️ Rainfall Prediction using Machine Learning  
This repository presents a machine learning-based solution for predicting rainfall levels using historical weather data. The objective is to develop a regression model that can accurately estimate rainfall amounts based on multiple meteorological features such as temperature, humidity, dew point, and pressure. The project applies advanced data preprocessing, feature engineering, and model evaluation techniques to build a reliable predictive system suitable for real-world applications like agriculture and climate planning 
# 🛠️ Approach 
  * Data Collection and Precessing
      * performs various step of pandas to know about the Data and process it for further steps
  * EDA :
     * plot the histogram to know about the nature of the data
     * also plot the heatmap to know the correlation between the features
     * plot the boxplot to know about the outliers
     * then downsampled the data into equal number of predicted values
  * Model Training :
     *🔍 Hyperparameter Tuning - Random Forest Classifier
     * To enhance the performance of the Random Forest Classifier, a grid search was performed to identify the optimal combination of hyperparameters. This step helps in improving model generalization and avoiding overfittin
     * This parameter grid was used with GridSearchCV to explore various combinations:

        n_estimators: Number of trees in the forest.

        max_features: The number of features to consider when looking for the best split.

        max_depth: The maximum depth of the trees (controls model complexity).

        min_samples_split: Minimum number of samples required to split an internal node.

        min_samples_leaf: Minimum number of samples required to be at a leaf node.
#📈 Results
   *Test Accuracy: 74.47%

   Confusion Matrix: Correctly classified 35 out of 47 instances

  Precision & Recall:

  Class 0: Precision = 0.77, Recall = 0.71

  Class 1: Precision = 0.72, Recall = 0.78

  F1-Score: Balanced performance across both classes

  Overall: The model shows good generalization with balanced class-wise metrics.


# 🧠 Tools & Technologies
Python
Pandas & NumPy
Scikit-learn
Matplotlib / Seaborn







