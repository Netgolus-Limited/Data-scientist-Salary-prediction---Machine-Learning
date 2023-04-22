# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Importing the eda_data.csv file as a dataframe to Python
df = pd.read_csv('CS14_eda_data.csv')

# 2. Creating a Python list that includes the names of all the predictors
predictors = ['Rating', 'size_id', 'sector_id', 'revenue_id', 'state_id', 'same_state', 
              'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'desc_len']

# 3. Creating a Python list that includes the name of the dependent variable
dependent_variable = ['six_figure']

# 4. Splitting the dataset into a train set and test set
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[dependent_variable], test_size=0.3, random_state=42)

# 5. Developing a decision tree classifier and depicting the tree using the train data set
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
# To depict the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True)
plt.show()

# 6. Calculating the accuracy score of the decision tree model using the test data set
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print('Accuracy score of decision tree model:', dt_acc)

# 7. Developing a random forest model and depicting the tree using the train data set
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
# To depict the tree
estimator = rf.estimators_[5]
plt.figure(figsize=(20,10))
plot_tree(estimator, filled=True)
plt.show()

# 8. Calculating the accuracy score of the random forest model using the test data set
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print('Accuracy score of random forest model:', rf_acc)

# 9. Importing the CS14_testsample.csv file as a dataframe to Python
test_df = pd.read_csv('CS14_testsample.csv')

# 10. Making a prediction whether the job postings in the testsample data set are expected to be “six-figure” jobs
# Using the random forest model as it had a higher accuracy score
test_pred = rf.predict(test_df[predictors])
for i, prediction in enumerate(test_pred):
    if prediction == 1:
        print(f"Job {i+1}: a six-figure job ({prediction})")
    else:
        print(f"Job {i+1}: not a six-figure job ({prediction})")

# 11. Calculating the importance of each predictor and determining the most important factor when predicting salary
importances = pd.Series(rf.feature_importances_, index=predictors)
print('Feature importances:\n', importances)
print('Most important factor:', importances.idxmax())
