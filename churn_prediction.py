import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
sns.set()

data=pd.read_csv(r'C:\Users\KIIT0001\Desktop\customer_churn_project\WA_Fn-UseC_-Telco-Customer-Churn.csv')
##data collection and understanding:

print(data.head())
print(data.describe(include='all'))
print(data.info())

#the number of yes and no values and their corresponding percentages:
churn_counts = data['Churn'].value_counts()
print(churn_counts)
churn_percentages = data['Churn'].value_counts(normalize=True) * 100
print(churn_percentages)

## VISUALISE RELATIONSHIPS:
# Create a bar plot
sns.countplot(data=data, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.savefig('churning_based_on_contract.png')
plt.show()

# Create a distribution plot for 'MonthlyCharges'
sns.histplot(data=data, x='MonthlyCharges', hue='Churn', kde=True)
plt.title('Distribution of Monthly Charges for Churned vs. Non-Churned Customers')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of customers')
plt.savefig('churning_based_on_monthlyCharges.png')
plt.show()

#distribution plot for 'Tenure'
sns.histplot(data=data,x='tenure',hue='Churn',kde=True)
plt.title('Tenure period for Churned vs. Non-Churned Customers')
plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.savefig('churning_based_on_tenure.png')
plt.show()

##data preprocessing:

#checking for missing values:-
print('the missing values')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')# since it seemed suspicious because it was storing numeric values despite being an object datatype
print(data.isnull().sum()) #the column TotalCharges contains 11 missing values.

missing_total_charges=data[data['TotalCharges'].isnull()]
print('the tenure for the missing values')
print(missing_total_charges['tenure'].value_counts())
#here it was inferred that the columns with missing values in TotalCharges has a tenure of 0; therefore filling those places with zero as the customers who just bought the service won't have to pay any charges instead of filling with median which could be inappropriate.

#filling those empty cells with the value '0':
data['TotalCharges'].fillna(0, inplace=True)


#now encoding the categorical features(using one hot encoding):
# Identify all the categorical columns
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',  'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod',]
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data_encoded = data_encoded.drop('customerID', axis=1)

#now seperating the features(x) and the target variable(y):
x = data_encoded.drop('Churn', axis=1) #including all the features except for the target variable
y = data_encoded['Churn'] #the target variable

#seperating the dataset for training and testing:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('dimeansions of the train and test data')
print(x_train.shape)
print(x_test.shape)

#saving the list of all feature columns for use in the app
feature_cols = x_train.columns.tolist()
joblib.dump(feature_cols, 'feature_cols.pkl')

#scaling the data(standardizing/normalizing )
from sklearn.preprocessing import StandardScaler

#identifying the numerical columns to scale
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

#initializing the StandardScaler class
scaler = StandardScaler()

#fitting the scaler on the training data and transform both training and testing data
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])
#preventing the data leakage and ensuring the model's evaluation is not biased

## building the model now:
from sklearn.linear_model import LogisticRegression
#initialize the model
model = LogisticRegression(random_state=42)

#train the model on the training data
model.fit(x_train, y_train)

#making predictions on the test data
y_pred = model.predict(x_test)

#now evaluating the model:
from sklearn.metrics import classification_report
print('the classificaton report')
print(classification_report(y_test, y_pred))#printing the classification report:

#the precision for the "Yes" class is 0.69,meaning that when our model predicted a customer would churn it was correct 69% of the time and likewise for the customers who would not churn.
#the recall(true positive rate) for the "Yes" class is 0.60,meaning that of all the customers who actually churned our model correctly identified only 60% of them and likewise for the ones who did not churn
#F1-Score:this is the harmonic mean of precision and recall .The F1-score of 0.64 for the "Yes" class indicates a moderate balance between correctly identifying churners and avoiding false alarms.
#Support: This is the number of actual customers in each class in the test set.There were 1036 customers who did not churn and 373 who did.
#accuracy: 82% meaning our model was 82% correct while making the predictions
#macro avg- for balanced data and weighted avg- for imbalanced data

#note-high recall means our model is sensitive and catches many of the actual churners and low precision means our model is also making a lot of false positives
#to reduce the number of false positives, we need to increase our threshold function(which is 0.5 , by default)[OPTIONAL to keep a threshold variable]

#note- we used classification report instead of the confusion matrix as it gives a much more clear idea and a classification report is a summarised version of the confusion matrix.(and it also calculates the key metrices)


##now addressing the class imbalance since we saw there is a great imbalance between the churners and non churners. A class imbalance occurs in a classification dataset when the number of instances in one class is significantly lower than in other classes).
#we only balance the training data and not the testing data because the test set needs to remain a true representation of the real world.
#The sole purpose of balancing the training data is to help the model learn the patterns of the minority class more effectively. without this the model would become biased and perform poorly on the rare cases.
#Doing this via SMOTE technique-
#it gives the model more examples of the minority class to learn from, which in turn helps it to better recognize and predict those cases in the real world
#SMOTE:stands for Synthetic Minority Over-sampling Technique, a method used in machine learning to address the problem of class imbalance in a dataset.
#SMOTE's core function is to create new, synthetic data points for the minority class. It does this by:
#identifying a data point in the minority class
#inding its nearest neighbors (other data points in the same minority class)
#generating a new synthetic data point somewhere along the line connecting the original data point and one of its nearest neighbors
#and this process is repeated until the minority class has a similar number of instances to the majority class, creating a more balanced dataset

from imblearn.over_sampling import SMOTE
from collections import Counter

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Print the new class distribution to confirm it's balanced
print('original dataset shape %s' % Counter(y_train))
print('resampled dataset shape %s' % Counter(y_train_resampled))

##now hyperparameter tuning :
#Hyperparameter tuning is the process of finding the best settings for our ML model to optimize its performance. These settings are not learned from the data but are defined by us only before training the model.
#The most common way to tune hyperparameters is with Grid Search. we will provide a grid of different values for each hyperparameter we want to test. The algorithm then trains a model for every possible combination on that grid and tells which combination produced the best result, often measured by a metric like F1-score or recall.
#doing this via GridSearchCV:

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#define the hyperparameters to tune(test):
#here, C is a regularization(a technique which prevents overfitting) parameter
#regularization works by adding a penalty to the model's complexity, discouraging the model from assigning very large weights(coefficients) to any single feature

#C controls the strength of this penalty (it is the inverse of the regularization strength)
#a smaller value of C means a strong penalty, thereby forcing the model to be simpler and less likely to overfit and likewise

#solver:a specific optimization algorithm Logistic Regression uses to find the best model parameters that fits our data
#here we're using liblinear and saga as these are the best-suited algorithms for binary classification.
#some other solvers are- lbfgs, newton-cg , sag
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000] #here,as the default maximum number of iterations (max_iter) for LogisticRegression in scikit-learn is 100 so here by adding [1000, 2000] to param_grid, we've given it a higher limit to work with
}

#initializing the grid search
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5, #setting up 5-fold cross validation(dividing the resampled training data into 5 parts here)
    scoring='f1' #using the metric f1-score to find the best model (as the combination among the 10 different models(combinations), thw onw whose f1 score is highest will prevail)
)

#fitting the grid search to your resampled training data(final execution step)
#iterating through every single combination(10) of hyperparameters:
grid_search.fit(x_train_resampled, y_train_resampled)

#getting the best hyperparameters and the best model
#(extraction of the best grid search)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("best hyperparameters:", best_params)

##NOW EVALUATING OUR FINAL MODEL:
#from sklearn.metrics import classification_report

#making predictions using the best model found by GridSearchCV
y_pred_tuned = best_model.predict(x_test)

# Print the final classification report
print("Final classification report(tuned model):")
print(classification_report(y_test, y_pred_tuned))

#getting the prediction probabilities
#returning a 2d array: [P(No), P(Yes)]
final_array = best_model.predict_proba(x_test)
print(final_array)
final_array_1d = final_array[:, 1]
#the first column will show the probabilities for the customers who will not churn and 2nd column is for the ones who are likely to 
#for the probabilities of the second column:
# P > 0.80 :HIGH RISK of churning
# 0.50-P-0.80 :MEDIUM RISK of churning
# P < 0.50 :LOW RISK of churning


##adding a roc-auc curve to visualise our model's ability to distinguish between two classes(yes,no)
#ROC-AUC:receiver operating characteristic - area under the curve, is a standard metric used to evaluate the performance of a classification model
#this curve will show us how our model will perform at all thresholds(the cutoff points we will set to turn our model's probability output into a final 'yes' or 'no prediction)
#roc curve:The ROC curve is a graph that visualizes a model's performance at all possible classification thresholds.Tt plots two key metrics :
#true positive rate (TPR):(recall,basically)The proportion of actual positive cases(churners) that were correctly identified.
#false positive rate (FPR):The proportion of actual negative cases (non-churners) that were incorrectly identified as positive.
#The curve basically shows the trade off between these two rates
#auc score:The AUC score is a single number that summarizes the entire ROC curve.
#the score ranges from 0.5 to 1.0.
#an AUC of 1.0 indicates a perfect model that correctly separates the two classes without any errors while an AUC of 0.5 indicates a model that is no better than a random guess.So basically a higher AUC score means your model is better at distinguishing between the two classes.
from sklearn.metrics import roc_curve, roc_auc_score

# Get the probability of the positive class (churn = 1)
#y_prob = best_model.predict_proba(x_test)[:, 1]

# Calculate the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, final_array_1d) #calculating the data points for the ROC curve threby comparing the true labels (y_test) to the predicted probabilities (y_prob)
auc_score = roc_auc_score(y_test, final_array_1d) #calculating the final auc score

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC-AUC curve')
plt.legend(loc='lower right')
plt.savefig('roc_auc_curve.png')
plt.show()


##BUILDING A SIMPLE WEB APP FOR DEPLOYMENT:

#saving the model to a file
#joblib.dump(best_model, 'churn_model.pkl')

#saving the scaler to a file
#joblib.dump(scaler, 'scaler.pkl')
#This will save our trained model and the scaler as binary files (.pkl) that our app can load later

output_folder = r'C:\Users\KIIT0001\Desktop\customer_churn_project'

# Save the model
joblib.dump(best_model, output_folder + r'\churn_model.pkl')

# Save the scaler
joblib.dump(scaler, output_folder + r'\scaler.pkl')

# Save the feature columns
joblib.dump(feature_cols, output_folder + r'\feature_cols.pkl')