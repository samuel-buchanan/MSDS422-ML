

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")

# columns

# Pclass: looks like only 1st, 2nd or 3rd, numeric, make dummies
# Sex: only male or female, convert from text to binary
train = train.replace({"Sex" : {"male" : 1, "female" : 2}})
train.loc[:, "Sex"] = train.loc[:, "Sex"].fillna(0)
# Age: numerical
# SibSp: binary, don't know what it means
# Parch: from 0 to 6, mostly 0 and 2
# Ticket: partially numeric partially text, maybe not keep
# Fare: numeric
# Cabin: only 147/891 unique entries, most NaN, maybe not keep
# Embarked: only three categories, S C Q, some NA, make dummy variables

# setting up response variable
y = train.Survived

# dropping Ticket, Cabin, PassengerID, Name, Survived
train.drop("PassengerId", axis = 1, inplace = True)
train.drop("Cabin", axis = 1, inplace = True)
train.drop("Ticket", axis = 1, inplace = True)
train.drop("Name", axis = 1, inplace = True)
train.drop("Survived", axis = 1, inplace = True)

# use train.dtypes to find types of each column
categorical = train.select_dtypes(include = "object").columns
train_cat = train[categorical]
numerical = train.select_dtypes(exclude = "object").columns
train_num = train[numerical]

# Filling na's in numerical columns with median values
train_num = train_num.fillna(train_num.median())

# making dummy variables for categorical columns
train_cat = pd.get_dummies(train_cat)

# put it all together and what do you get
train = pd.concat([train_num, train_cat], axis = 1)

# no 'object' type columns means that we can just StandardScaler everything
scaler = StandardScaler()
train[:] = scaler.fit_transform(train[:])


# splitting into 70/30 train / test sets
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)

# RMSE calculation
def rmse(pred_val, true_val):
    rmse = np.sqrt(mean_squared_error(true_val, pred_val))
    return(rmse)



# LogisticRegression
logr = LogisticRegression()
logr.fit(X_train, y_train)
logr_predict_train = logr.predict(X_train)
rmse(logr_predict_train, y_train) # 0.44793...

logr_predict_test = logr.predict(X_test)
rmse(logr_predict_test, y_test) # 0.45301...

# GaussianNB, naive bayes
gaus = GaussianNB()
gaus.fit(X_train, y_train)
gaus_predict_train = gaus.predict(X_train)
rmse(gaus_predict_train, y_train) # 0.47573...

gaus_predict_test = gaus.predict(X_test)
rmse(gaus_predict_test, y_test) # 0.49248...

# BernoulliNB
bern = BernoulliNB()
bern.fit(X_train, y_train)
bern_predict_train = bern.predict(X_train)
rmse(bern_predict_train, y_train) # 0.47573...

bern_predict_test = bern.predict(X_test)
rmse(bern_predict_test, y_test) # 0.47708...



# now to apply the same formatting to the test.csv for Kaggle submissions
test = pd.read_csv("test.csv")

id_for_submission = test.PassengerId

test = test.replace({"Sex" : {"male" : 1, "female" : 2}})
test.loc[:, "Sex"] = test.loc[:, "Sex"].fillna(0)

# dropping Ticket, Cabin, PassengerID, Name
test.drop("PassengerId", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)

# use test.dtypes to find types of each column
categorical = test.select_dtypes(include = "object").columns
test_cat = test[categorical]
numerical = test.select_dtypes(exclude = "object").columns
test_num = test[numerical]

# Filling na's in numerical columns with median values
test_num = test_num.fillna(train_num.median())

# making dummy variables for categorical columns
test_cat = pd.get_dummies(test_cat)

# put it all together and what do you get
test = pd.concat([test_num, test_cat], axis = 1)

# no 'object' type columns means that we can just StandardScaler everything
scaler = StandardScaler()
test[:] = scaler.fit_transform(test[:])

logr_predict_test = logr.predict(test)
gaus_predict_test = gaus.predict(test)
bern_predict_test = bern.predict(test)


sb_logr_predict = pd.DataFrame(logr_predict_test)
sb_logr_predict["PassengerId"] = id_for_submission
sb_logr_predict.columns = ["Survived", "PassengerId"]
sb_logr_predict = sb_logr_predict[["PassengerId", "Survived"]]
sb_logr_predict.to_csv("sb_logistic_predictions.csv")
# need to drop first column in sb_logr_predict before exporting to csv

sb_gaus_predict = pd.DataFrame(gaus_predict_test)
sb_gaus_predict["PassengerId"] = id_for_submission
sb_gaus_predict.columns = ["Survived", "PassengerId"]
sb_gaus_predict = sb_gaus_predict[["PassengerId", "Survived"]]
sb_gaus_predict.to_csv("sb_GaussianNB_predictions.csv")

sb_bern_predict = pd.DataFrame(bern_predict_test)
sb_bern_predict["PassengerId"] = id_for_submission
sb_bern_predict.columns = ["Survived", "PassengerId"]
sb_bern_predict = sb_bern_predict[["PassengerId", "Survived"]]
sb_bern_predict.to_csv("sb_BernoulliNB_predictions.csv")

# AUC scores
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

logr_probs = logr.predict_proba(X_train)
logr_probs = logr_probs[:, 1] 
# have to do this since otherwise logr_probs is the wrong shape for roc_auc_score
roc_auc_score(y_train, logr_probs) # 0.85912...

gaus_probs = gaus.predict_proba(X_train)
gaus_probs = gaus_probs[:, 1]
roc_auc_score(y_train, gaus_probs) # 0.82782...

bern_probs = bern.predict_proba(X_train)
bern_probs = bern_probs[:, 1] 
roc_auc_score(y_train, bern_probs) # 0.81981...

# plotting ROC curves
logr_fpr, logr_tpr, _ =roc_curve(y_train, logr_probs)
gaus_fpr, gaus_tpr, _ =roc_curve(y_train, gaus_probs)
bern_fpr, bern_tpr, _ =roc_curve(y_train, bern_probs)

pyplot.plot(logr_fpr, logr_tpr, marker = ".", label = "Logistic Regression")
pyplot.plot(gaus_fpr, gaus_tpr, marker = ".", label = "Gaussian NB")
pyplot.plot(bern_fpr, bern_tpr, marker = ".", label = "Bernoulli NB")

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

pyplot.legend()

pyplot.show()


