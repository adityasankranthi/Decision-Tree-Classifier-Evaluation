from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

# loading a dataset
dia = datasets.fetch_openml(data_id=44345)

# create a decision tree object named mytree
mytree = DecisionTreeClassifier(criterion="entropy")

# Fit the tree to the features and target
mytree.fit(dia.data, dia.target)

# You cannot use roc_auc_score unless the predictions were numerical (e.g. probabilities)
predictions = mytree.predict(dia.data)
print(predictions)
#
# # We want to plot ROC curve, so we want prediction probabilities, hence we will use cross_val_predict
# dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
# y_scores = model_selection.cross_val_predict(dtc, dia.data, dia.target, method="predict_proba", cv=10)
#
# # Plotting ROC Curve
# fpr, tpr, th = roc_curve(dia.target, y_scores[:, 1], pos_label="Y")
#
# plt.xlabel("1 - Specificity")
# plt.ylabel("Sensitivity")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.plot(fpr, tpr, label="Decision Tree")
# plt.legend()
# plt.show()
#
# print(roc_auc_score(dia.target, y_scores[:, 1]))
#
# # Parameter Tuning
# parameters = [{"min_samples_leaf": [2, 4, 6, 8, 10]}]
#
# # create an object of GridSearchCV:
# dtc_1 = tree.DecisionTreeClassifier()
# tuned_dtc = model_selection.GridSearchCV(dtc_1, parameters, scoring="roc_auc", cv=5)
#
# y_scores_1 = model_selection.cross_val_predict(tuned_dtc, dia.data, dia.target, method="predict_proba", cv=10)
#
# fpr_2, tpr_2, th_2 = roc_curve(dia.target, y_scores_1[:, 1], pos_label="Y")
#
# plt.xlabel("1 - Specificity")
# plt.ylabel("Sensitivity")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.plot(fpr_2, tpr_2, label="Decision Tree 2")
# plt.legend()
# plt.show()
#
# print(roc_auc_score(dia.target, y_scores_1[:, 1]))
