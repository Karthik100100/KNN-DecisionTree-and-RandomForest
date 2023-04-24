#Indicate the imported packages/libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Load the dataset and print the data information

df=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\dataset.csv")
print(df)

#Plot some figures to visualize the dataset

class_distrib = df['class'].value_counts() 
class_distrib.plot(kind='pie')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in the dataset')
plt.grid()
plt.show()

#Print out the number of samples for each class in the dataset

class0=df.loc[df["class"].isin([0])]
print("The number of samples in class 0 are",class0.shape)


class1=df.loc[df["class"].isin([1])]
print("The number of samples in class 1 are",class1.shape)


#For each class, print out the statistical description of features
class0=class0.drop("class",axis=1)
print("Statistical discriptors for class 0 \n",class0.describe(include='all').loc[['mean', 'std', 'min', 'max']])
class1=class1.drop("class",axis=1)
print("Statistical discriptors for class 1 \n",class1.describe(include='all').loc[['mean', 'std', 'min', 'max']])


y=df["class"]
X=df.drop("class",axis=1)

X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state = 0)


k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# K-Fold Cross-Validation
combined_score={}
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(clf,X_train,y_train,cv=5,scoring=_scoring,return_train_score=True)

    pres=results['test_precision'].mean()*100
    rec=results['test_recall'].mean()*100
    acc=results['test_accuracy'].mean()*100
    f1v=results['test_f1'].mean()*100


    combined_score[k]=(2 * pres * rec) / (pres + rec) + acc + f1v

optimal_k = max(combined_score, key=lambda k:combined_score[k])


results=[]
clft=KNeighborsClassifier(n_neighbors=optimal_k)
clft.fit(X_train,y_train)

y_pred=clft.predict(X_test)


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results.append({'k for knn': optimal_k, 'Precision for knn': precision, 'Recall for knn': recall, 'Accuracy for knn': accuracy, 'F1-score for knn': f1})
# assume y_true and y_pred are your true and predicted labels
cm = confusion_matrix(y_test, y_pred)

# create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

# add labels to the plot
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for knn")
plt.show()

kf = pd.DataFrame(results)

# Print the results table
print(kf.to_string(index=False))


#decision tree

com_score={}
for depth in range(2,20):
    decision_tree_model = DecisionTreeClassifier(criterion="gini",
                                     random_state=0,max_depth=depth)
    decision_tree_model.fit(X_train,y_train)
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    decision_tree_result = cross_validate(decision_tree_model,X_train,y_train,cv=5,scoring=_scoring,return_train_score=True)
    
    
    pres=decision_tree_result['test_precision'].mean()*100
    rec=decision_tree_result['test_recall'].mean()*100
    acc=decision_tree_result['test_accuracy'].mean()*100
    f1v=decision_tree_result['test_f1'].mean()*100


    com_score[depth]=(2 * pres * rec) / (pres + rec) + acc + f1v

optimal_depth = max(com_score, key=lambda k:com_score[k])



results1=[]
des=DecisionTreeClassifier(criterion="gini",
                                     random_state=0,max_depth=optimal_depth)
des.fit(X_train,y_train)

y_pred1=des.predict(X_test)


precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
accuracy1 = accuracy_score(y_test, y_pred1)
f11 = f1_score(y_test, y_pred1)
results1.append({'depth for decision tree': optimal_depth, 'Precision for decision tree': precision1, 'Recall for decision tree': recall1, 'Accuracy for decision tree': accuracy1, 'F1-score for decision tree': f11})
# assume y_true and y_pred are your true and predicted labels
cm1 = confusion_matrix(y_test, y_pred1)

# create a heatmap of the confusion matrix
sns.heatmap(cm1, annot=True, cmap="Blues", fmt="d", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

# add labels to the plot
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for decision tree")
plt.show()

kf1 = pd.DataFrame(results1)

# Print the results table
print(kf1.to_string(index=False))


# random forest

param_grid = {'n_estimators': [50, 100, 150, 200],
              'max_depth':list(range(2,10))}
rfc = RandomForestClassifier(random_state=42,n_jobs=-1)

grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train,y_train)


rfc1=RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],n_jobs=-1,random_state=42,max_depth=grid_search.best_params_['max_depth'])
rfc1.fit(X_train,y_train)

output=rfc1.predict(X_test)
precision2 = precision_score(y_test, output)
recall2 = recall_score(y_test, output)
accuracy2 = accuracy_score(y_test, output)
f12 = f1_score(y_test, output)
dic={'depth for random forest': grid_search.best_params_['max_depth'],'estimator for random forest':grid_search.best_params_['n_estimators'], 'Precision for random forest': precision2, 'Recall for random forest': recall2, 'Accuracy for random forest': accuracy2, 'F1-score for random forest': f12}
# assume y_true and output are your true and predicted labels
cm2 = confusion_matrix(y_test, output)

# create a heatmap of the confusion matrix
sns.heatmap(cm2, annot=True, cmap="Blues", fmt="d", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

# add labels to the plot
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for random forest")
plt.show()

out=pd.Series(dic,index=dic.keys())
print(out)