Decision Tree Classifier Evaluation on Credit and mfeat-karhunen Datasets
Author: Aditya Sankranthi

Datasets
1. Credit Dataset
ID: 44345
Link: Credit Dataset
Description: This dataset is a subsample of the original Credit dataset (ID: 44089), which is part of the tabular data benchmark. The dataset is transformed similarly, with 2,000 instances and 11 features, focusing on binary classification. The goal is to classify data into two categories: true (1) or false (0). The original source of the dataset is the Kaggle competition: Give Me Some Credit.
2. mfeat-karhunen Dataset
ID: 1020
Link: mfeat-karhunen Dataset
Description: This is a binarized version of the original mfeat-karhunen dataset (ID: 16). The multi-class target feature has been converted to a two-class nominal target feature by relabeling the majority class as positive ('P') and all others as negative ('N'). The dataset includes 2,000 instances and 65 features.
Model Evaluation
The performance of Decision Tree classifiers using two different splitting criteria (Entropy and Gini impurity) was evaluated across both datasets. The evaluation was performed using various min_samples_leaf parameter values, ranging from 1 to 10. Below are the ROC curves and AUC values for each criterion.

ROC Curves
Visualize ROC curves here.

Table of AUC Values
Dataset	Criterion	AUC	Best Parameters
Dataset 1020	Entropy	0.953	{'min_samples_leaf': 10}
Dataset 1020	Gini	0.937	{'min_samples_leaf': 10}
Dataset 44345	Entropy	0.781	{'min_samples_leaf': 10}
Dataset 44345	Gini	0.786	{'min_samples_leaf': 8}
Conclusions
1. Model Performance
The model demonstrated good performance across both datasets, with relatively high AUC values. However, Dataset 1020 appears to be more straightforward for the model to learn from, as evidenced by its higher AUC values compared to Dataset 44345.
2. Parameter Tuning
The optimal min_samples_leaf parameter was found to be around 10 for both Entropy and Gini criteria. However, for the Gini criterion on Dataset 44345, the best min_samples_leaf value was 8, suggesting some dataset-specific nuances in model complexity.
3. Criterion Comparison
The difference in the best parameters between Entropy and Gini criteria, particularly for Dataset 44345, indicates that these criteria might prioritize different aspects of the data or have varying sensitivity to noise.
