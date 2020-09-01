import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA



train = pd.read_csv("train.csv")

label = train.label

train.drop("label", axis = 1, inplace = True)


rndfrst = RandomForestClassifier(n_estimators=100, max_features="sqrt", bootstrap = True, n_jobs=-1)
start = datetime.now()
rndfrst.fit(train, label)
end = datetime.now()
print(end-start) # 6.544455 seconds

# export and submit to kaggle
test = pd.read_csv("test.csv")
# temp_for_Id_values = pd.read_csv("test.csv")

rf_predicted_values = pd.DataFrame(rndfrst.predict(test))
rf_predicted_values.columns = ["Label"]
rf_predicted_values.to_csv("random_forest_predictions.csv")

# joining together train and test data
tt_data = pd.concat([train, test], axis = 0)

pca = PCA(n_components=0.95)
start = datetime.now()
overall_features = pca.fit_transform(tt_data)
end = datetime.now()
print(end-start) # 8.403654 seconds


# truncating overall_features to be the same shape as label
overall_features_trunc = overall_features[:42000]
overall_features_pred = overall_features[42000:70000]

# use principal components from above to train another RF 
# time and submit to kaggle
start = datetime.now()
rndfrst.fit(overall_features_trunc, label)
end = datetime.now()
print(end-start) # 15.223349 sec


pca_rf_predicted_values = pd.DataFrame(rndfrst.predict(overall_features_pred))
pca_rf_predicted_values.columns = ["Label"]
pca_rf_predicted_values.to_csv("pca_random_forest_predictions.csv")

# identify design flaw in this experiment, fix it
## no cross-validation? fitting on both train / test data?
# re-run experiment with fixes in place, submit to kaggle.com

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statistics

x_train, x_test, y_train, y_test = train_test_split(train, label, test_size = 0.2)

rndfrst = RandomForestClassifier(n_estimators=100, max_features="sqrt", bootstrap = True, n_jobs=-1)
start = datetime.now()
rndfrst.fit(x_train, y_train)
end = datetime.now()
print(end-start) # 6.782156 seconds


rnd_probs = rndfrst.predict_proba(x_train)
scores_train = cross_val_score(rndfrst, x_train, y_train)
print(statistics.mean(scores_train)) # 0.9590
rnd_probs = rndfrst.predict_proba(x_test)
scores_test = cross_val_score(rndfrst, x_test, y_test)
print(statistics.mean(scores_test)) # 0.9379, maybe slightly overfit

pca = PCA(n_components=0.95)
start = datetime.now()
overall_features = pca.fit_transform(x_train)
end = datetime.now()
print(end-start) # 3.936565 seconds

start = datetime.now()
rndfrst.fit(overall_features, y_train)
end = datetime.now()
print(end-start) # 11.894471 seconds

test2 = pca.transform(test)
pca_rf_predicted_values = pd.DataFrame(rndfrst.predict(test2))
pca_rf_predicted_values.columns = ["Label"]
pca_rf_predicted_values.to_csv("pca_tts_random_forest_predictions.csv")