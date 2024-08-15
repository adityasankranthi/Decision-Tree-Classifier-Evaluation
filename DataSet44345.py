import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

# Load dataset
data_44345 = datasets.fetch_openml(data_id=44345)

# Decision tree with entropy criterion
dtc_entropy = DecisionTreeClassifier(criterion='entropy')

# Decision tree with gini criterion
dtc_gini = DecisionTreeClassifier(criterion='gini')

# Define parameter grid for min_samples_leaf
param_grid = {"min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# param_grid = {"min_samples_leaf": [2, 4, 6, 8, 10]}

# GridSearchCV for entropy criterion
tuned_dtc_entropy = GridSearchCV(dtc_entropy, param_grid, scoring="roc_auc", cv=10)
tuned_dtc_entropy.fit(data_44345.data, data_44345.target)
y_scores_entropy = cross_val_predict(tuned_dtc_entropy, data_44345.data, data_44345.target, method="predict_proba", cv=10)

# GridSearchCV for gini criterion
tuned_dtc_gini = GridSearchCV(dtc_gini, param_grid, scoring="roc_auc", cv=10)
tuned_dtc_gini.fit(data_44345.data, data_44345.target)
y_scores_gini = cross_val_predict(tuned_dtc_gini, data_44345.data, data_44345.target, method="predict_proba", cv=10)

# Plot ROC curves and compute AUC values for entropy criterion
fpr_entropy, tpr_entropy, th_entropy = roc_curve(data_44345.target, y_scores_entropy[:, 1], pos_label="1")
auc_entropy = roc_auc_score(data_44345.target, y_scores_entropy[:, 1])
plt.plot(fpr_entropy, tpr_entropy, label="Entropy (AUC = {:.2f})".format(auc_entropy))

# Plot ROC curves and compute AUC values for gini criterion
fpr_gini, tpr_gini, th_gini = roc_curve(data_44345.target, y_scores_gini[:, 1], pos_label="1")
auc_gini = roc_auc_score(data_44345.target, y_scores_gini[:, 1])
plt.plot(fpr_gini, tpr_gini, label="Gini (AUC = {:.2f})".format(auc_gini))

# Plot settings
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("ROC Curve - 44345")
plt.legend()
plt.show()

print("Entropy AUC:", auc_entropy)
print("Gini AUC:", auc_gini)
print("Best parameters for entropy criterion:", tuned_dtc_entropy.best_params_)
print("Best parameters for gini criterion:", tuned_dtc_gini.best_params_)
