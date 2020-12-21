import statsmodels.api as sm
import pandas as pd
import ast
import statistics

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

import warnings 
warnings.filterwarnings('ignore')
import joblib
from dataprep import drop_columns, balancing, unpack_string, add_lemmatized_unpacked_column

# Read in preprocessed data
ads = pd.read_csv("data/ads_preprocessed.csv")

# Prepare lemmatized words
ads = add_lemmatized_unpacked_column(ads)

# Assigning y and X
y = ads["jobad_appealingness_dis"]
X = ads.drop("jobad_appealingness_dis", axis=1)


# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing Data
train = X_train.join(y_train)

train = balancing(train)

print("Data are balanced")
print(train["jobad_appealingness_dis"].value_counts())

# Assigning y and X
y_train = train["jobad_appealingness_dis"]
X_train = train.drop("jobad_appealingness_dis", axis=1)

# 1. Logistic Regression Model on extracted text features
unnecessary_columns = ['jobad_appealingness_ind', 'discipline_label',
       'industry_label', 'gender_dominance_ind', 'gender_dominance_dis',
       'ad_cleaned', 'ad_lemmatized', 'ad_postag', 'ad_entities', 'item_id', 'title', 'description', 'discipline_id', 'industry_id',
       'female_ratio_for_discipline', 'female_ratio_for_industry', 'relative_female_ratio_discipline',
       'female_ratio_of_applicants', 'relative_female_ratio_industry', 'ad_tokens', 'workingtime_words', 'homeoffice_words',
       'salary_words', 'family_words', 'safety_words', 'health_words','number', 'euro', 'language', 'words_f', 'words_m','travel_words', 'lemmatized_unpacked']

feature_column = ['ad_length', 'words_f_count',
       'words_m_count', 'workingtime_info', 'homeoffice_opportunity', 'salary',
       'family_benefits', 'safety_statements', 'health_benefits', 'traveling']


X_train_logreg = drop_columns(X_train, unnecessary_columns)
X_test_logreg = drop_columns(X_test, unnecessary_columns)

# fit model
lr = LogisticRegression()
lr.fit(X_train_logreg, y_train)
# Make predictions on X_test
y_pred_logreg = lr.predict(X_test_logreg)
y_pred_train_logreg = lr.predict(X_train_logreg)

print ("Confusion Matrix_lr: \n", 
    confusion_matrix(y_test, y_pred_logreg)) 

print ("Accuracy_lr: \n", 
    accuracy_score(y_test, y_pred_logreg)*100)

print ("Report_lr: \n", 
        classification_report(y_test, y_pred_logreg))

print ("Accuracy_train_lr: \n", 
    accuracy_score(y_train, y_pred_train_logreg)*100)

# Predict Probabilities for ensemble model
pred_proba_logreg=lr.predict_proba(X_test_logreg)
pred_proba_train_logreg=lr.predict_proba(X_train_logreg)

# Saving LogReg model in Models-folder
print("Saving model in the model folder")
filename = 'models/lr.pkl'
joblib.dump(lr, filename)

#____________________________________________________
# 2. XGBoost Classifier on extracted text features
unnecessary_columns = ['jobad_appealingness_ind', 'discipline_label',
       'industry_label', 'gender_dominance_ind', 'gender_dominance_dis',
       'ad_cleaned', 'ad_lemmatized', 'ad_postag', 'ad_entities', 'item_id', 'title', 'description', 'discipline_id', 'industry_id',
       'female_ratio_for_discipline', 'female_ratio_for_industry', 'relative_female_ratio_discipline',
       'female_ratio_of_applicants', 'relative_female_ratio_industry', 'ad_tokens', 'workingtime_words', 'homeoffice_words',
       'salary_words', 'family_words', 'safety_words', 'health_words','number', 'euro', 'language', 'words_f', 'words_m','travel_words', 'lemmatized_unpacked']

feature_column = ['ad_length', 'words_f_count',
       'words_m_count', 'workingtime_info', 'homeoffice_opportunity', 'salary',
       'family_benefits', 'safety_statements', 'health_benefits', 'traveling']


X_train_xg = drop_columns(X_train, unnecessary_columns)
X_test_xg = drop_columns(X_test, unnecessary_columns)

# Fit XGBoost Classifier with best parameters found in RandomGridSearch
xgboost = XGBClassifier(min_child_weight= 7, max_depth = 3, learning_rate = 0.1, gamma = 0.4, colsample_bytree = 0.4)
xgboost.fit(X_train_xg, y_train)

# Make predictions on X_test
y_pred_xg = xgboost.predict(X_test_xg)
y_pred_train_xg = xgboost.predict(X_train_xg)

print ("Confusion Matrix_xg: \n", 
    confusion_matrix(y_test, y_pred_xg)) 

print ("Accuracy_xg: \n", 
    accuracy_score(y_test, y_pred_xg)*100)

print ("Report_xg: \n", 
        classification_report(y_test, y_pred_xg))

print ("Accuracy_train_xg: \n", 
    accuracy_score(y_train, y_pred_train_xg)*100)

# Predict Probabilities for ensemble model
pred_proba_xg=xgboost.predict_proba(X_test_xg)
pred_proba_train_xg=xgboost.predict_proba(X_train_xg)

# Saving XGBoost model in Models-folder
print("Saving model in the model folder")
filename = 'models/xgboost.pkl'
joblib.dump(xgboost, filename)
#________________________________________________
# 3. NLP Model: Random Forest Classifier

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 3
vect_ltf = TfidfVectorizer(min_df=3).fit(X_train['lemmatized_unpacked'])
# Saving Vectorizer in Models-folder
print("Saving model in the model folder")
filename = 'models/vect_ltf.pkl'
joblib.dump(vect_ltf, filename)

# transform the documents in the training data to a document-term matrix
X_train_vectorized_ltf = vect_ltf.transform(X_train['ad_cleaned'])
# Fit Random Forest Classifier with best parameters found in RandomGridSearch
'''rforest = RandomForestClassifier(bootstrap=False, n_estimators=1400, 
                                min_samples_leaf=2,
                                max_features = 'sqrt',
                                min_samples_split=5)

rforest.fit(X_train_vectorized_ltf, y_train)
# Make predictions on X_test_lt
# Predict the transformed test documents
y_pred_rf = rforest.predict(vect_ltf.transform(X_test['ad_cleaned']))
y_pred_train_rf = rforest.predict(X_train_vectorized_ltf)

print("Confusion Matrix_rf: \n", 
    confusion_matrix(y_test, y_pred_rf)) 

print ("Accuracy_rf: \n", 
    accuracy_score(y_test,y_pred_rf)*100) 

print("Report_rf: \n", 
    classification_report(y_test, y_pred_rf))

print ("Accuracy_train_rf: \n", 
    accuracy_score(y_train,y_pred_train_rf)*100) 

# Predict Probabilities for ensemble model
pred_proba_rf = rforest.predict_proba(vect_ltf.transform(X_test['ad_cleaned']))
pred_proba_train_rf = rforest.predict_proba(X_train_vectorized_ltf)'''

# Saving LogReg model in Models-folder
#print("Saving model in the model folder")
#filename = 'models/rforest.pkl'
#joblib.dump(rforest, filename)

#________________________________________________
# 4. NLP Model: Logistic Regression
# Fit Logistic Regression model
logreg_NLP = LogisticRegression(max_iter=1500)
logreg_NLP.fit(X_train_vectorized_ltf, y_train)

# Make predictions on X_test_lt
# Predict the transformed test documents
y_pred_logreg_NLP = logreg_NLP.predict(vect_ltf.transform(X_test['ad_cleaned']))
y_pred_train_logreg_NLP = logreg_NLP.predict(X_train_vectorized_ltf)

print("Confusion Matrix_logregnlp: \n", 
    confusion_matrix(y_test, y_pred_logreg_NLP)) 

print ("Accuracy_logregnlp: \n", 
    accuracy_score(y_test,y_pred_logreg_NLP)*100) 

print("Report_logregnlp: \n", 
    classification_report(y_test, y_pred_logreg_NLP))

print ("Accuracy_train_logregnlp: \n", 
    accuracy_score(y_train,y_pred_train_logreg_NLP)*100) 

# Predict Probabilities for ensemble model
pred_proba_logreg_NLP = logreg_NLP.predict_proba(vect_ltf.transform(X_test['ad_cleaned']))
pred_proba_train_logreg_NLP = logreg_NLP.predict_proba(X_train_vectorized_ltf)

# Saving LogReg model in Models-folder
print("Saving model in the model folder")
filename = 'models/logreg_NLP.pkl'
joblib.dump(logreg_NLP, filename)
#________________________________________________

# Final model with all four single models
X_test_copy = X_test_xg.copy()
X_test_copy["LogReg"] = pred_proba_logreg[:, 1]
X_test_copy["XGB"] = pred_proba_xg[:, 1]
#X_test_copy["RF"] = pred_proba_rf[:, 1]
X_test_copy["LogReg_NLP"] = pred_proba_logreg_NLP[:, 1]
X_test_agg = X_test_copy[["LogReg", "XGB", "LogReg_NLP"]]

X_train_copy = X_train_xg.copy()
X_train_copy["LogReg"] = pred_proba_train_logreg[:, 1]
X_train_copy["XGB"] = pred_proba_train_xg[:, 1]
#X_train_copy["RF"] = pred_proba_train_rf[:, 1]
X_train_copy["LogReg_NLP"] = pred_proba_train_logreg_NLP[:, 1]
X_train_agg = X_train_copy[["LogReg", "XGB", "LogReg_NLP"]]

#fit model
final_model = LogisticRegression()
final_model.fit(X_train_agg, y_train)
#Make predictions on X_test
y_pred_train_final = final_model.predict(X_train_agg)
y_pred_test_final= final_model.predict(X_test_agg)

print("Confusion Matrix_final: \n", 
    confusion_matrix(y_test, y_pred_test_final))

print ("Accuracy_final: \n", 
    accuracy_score(y_test, y_pred_test_final)*100) 

print("Report_final: \n", 
    classification_report(y_test, y_pred_test_final))

print ("Accuracy_train_final: \n", 
    accuracy_score(y_train, y_pred_train_final)*100) 

# Saving LogReg model in Models-folder
print("Saving model in the model folder")
filename = 'models/final_model.pkl'
joblib.dump(final_model, filename)