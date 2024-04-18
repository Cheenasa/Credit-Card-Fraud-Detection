# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import train_test_split

# To undersample and oversample the data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x) # To restrict the float value to 2 decimal places


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/drive/MyDrive/creditcard_2023.csv")#reading data from my drive.

df = data.copy() # making a copy of the dataset to preserve the original data.
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns.')  # f-string
df.sample(10, random_state=3)#This ensures that a consistent random sample of 10 rows is shown every time the code is run.

df.duplicated().sum()

df.isnull().sum()

df.info()

# Dropping the 'id' column because it does not add to the visualizations and model building
df.drop(['id'], axis=1, inplace=True)

df1000 = df[df['Amount']<=1000]
df1000['Amount'] = StandardScaler().fit_transform(df1000['Amount'].values.reshape(-1,1))
print(df1000.shape)
df1000['Class'].value_counts()

df5000 = df[(df['Amount']>1000) & (df['Amount']<=5000)]
df5000['Amount'] = StandardScaler().fit_transform(df5000['Amount'].values.reshape(-1,1))
print(df5000.shape)
df5000['Class'].value_counts()

df_max = df[df['Amount']>5000]
df_max['Amount'] = StandardScaler().fit_transform(df_max['Amount'].values.reshape(-1,1))
print(df_max.shape)
df_max['Class'].value_counts()

X1000 = df1000.drop(["Class"] , axis=1)
y1000 = df1000["Class"]

X5000 = df5000.drop(["Class"] , axis=1)
y5000 = df5000["Class"]

X_max = df_max.drop(["Class"] , axis=1)
y_max = df_max["Class"]

# Splitting X1000, y1000
X1000_train, X1000_test, y1000_train, y1000_test = train_test_split(X1000, y1000, test_size=.30, random_state=1, stratify=y1000)

# Splitting X5000, y5000
X5000_train, X5000_test, y5000_train, y5000_test = train_test_split(X5000, y5000, test_size=.30, random_state=1, stratify=y5000)

# Splitting X_max, y_max
X_max_train, X_max_test, y_max_train, y_max_test = train_test_split(X_max, y_max, test_size=.30, random_state=1, stratify=y_max)

# Oversampling with SMOTE
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X5000_train, y5000_train)

# Undersampling with RandomUnderSampler
rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
X_train_un, y_train_un = rus.fit_resample(X5000_train, y5000_train)

# Decision Tree Classifier
dTree = DecisionTreeClassifier()

# Bagging Classifier with Decision Tree
bagging_clf = BaggingClassifier(base_estimator=dTree)

# AdaBoost Classifier with Decision Tree
adaboost_clf = AdaBoostClassifier(base_estimator=dTree)

# Random Forest Classifier
rf_clf = RandomForestClassifier()

classifiers = {"Decision Tree": dTree, "Bagging": bagging_clf, "AdaBoost": adaboost_clf, "Random Forest": rf_clf}

datasets = {"X1000": (X1000_train, X1000_test, y1000_test),
            "X5000": (X5000_train, X5000_test, y5000_test),
            "X_max": (X_max_train, X_max_test, y_max_test),
            "X5000_over": (X_train_over, X5000_test, y5000_test),
            "X5000_un": (X_train_un, X5000_test, y5000_test)}

for dataset_name, (X_train, X_test, y_test) in datasets.items():
    print(f"Dataset: {dataset_name}")
    for name, clf in classifiers.items():
        print(f"\tTraining {name} classifier...")
        clf.fit(X1000_train, y1000_train)

        print(f"\tEvaluating {name} classifier...")
        y_pred = clf.predict(X_test)

        # Evaluate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\t{name} Accuracy:", accuracy)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="g")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title(f"{name} Confusion Matrix - {dataset_name}")
        plt.show()

        # Classification report
        print(f"\t{name} Classification Report:")
        print(classification_report(y_test, y_pred))

        # AUC-ROC Score
        print(f"\t{name} AUC-ROC Score: {roc_auc_score(y_test, y_pred)}")