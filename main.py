import inline as inline
import matplotlib
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error

pd.set_option('display.max_rows', None)

#Pre-Covid 2018 Crime Dataset Pre-processing
data = pd.read_csv(r'C:\Users\alexm\OneDrive\Documents\Datasets\During-Covid\crime-incident-reports-2022.csv')
data = data.drop(labels = ["INCIDENT_NUMBER", "DISTRICT", "OFFENSE_CODE_GROUP",
                           "OCCURRED_ON_DATE", "OFFENSE_DESCRIPTION", "REPORTING_AREA",
                           "SHOOTING", "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR",
                           "STREET", "UCR_PART", "Location"], axis=1)
data = data.loc[data['OFFENSE_CODE'].isin([301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                                          311, 312, 313, 314, 315, 316, 317, 318, 319, 320,
                                          321, 322, 323, 324, 333, 334, 335, 336, 337, 338,
                                          339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
                                          349, 350, 351, 352, 353, 354, 355, 356, 357, 358,
                                          359, 360, 361, 362, 363, 364, 371, 373, 374, 375,
                                          376, 377, 378, 379, 380, 381, 520, 522, 540, 542,
                                          560, 561, 562, 611, 612, 613, 614, 615, 616, 617,
                                          618, 619, 621, 622, 623, 624, 625, 626, 627, 628,
                                          629, 631, 632, 633, 634, 635, 636, 637, 638, 639,
                                          649, 1300, 1302, 1304])]

#data.to_csv(r'C:\Users\alexm\OneDrive\Documents\Datasets\During-Covid\2022 crime.csv')

#Merged Dataset
df = pd.read_csv(r'C:\Users\alexm\OneDrive\Documents\Datasets\Pre-Covid\peepo4.csv')
dftest = pd.read_csv(r'C:\Users\alexm\OneDrive\Documents\Datasets\During-Covid\2022 crime rates.csv')

#Training the Random Forests Model
X = df.iloc[:, 0:20].values
y = df.iloc[:, 20].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# rfmodel = RandomForestRegressor(n_estimators=200, random_state=0)
# rfmodel.fit(X_train, y_train)
# y_pred = rfmodel.predict(X_test)

#Training the XGBoost Model

xgmodel = xg.XGBRegressor(n_estimators=1000, max_depth=4, eta=0.1, subsample=1.0, colsample_bytree=1.0)
xgmodel.fit(X_train, y_train)
y_pred = xgmodel.predict(X_test)

#Evaluating the Models

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', xgmodel.score(X_test, y_test))

# plt.figure(figsize=(12, 12))
# heatmap = sns.heatmap(df.corr())
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
# plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()
# print(prediction_error(rfmodel, X_train, y_train, X_test, y_test))
# print(residuals_plot(rfmodel, X_train, y_train, X_test, y_test))


#np.savetxt(r'C:\Users\alexm\OneDrive\Documents\Datasets\awdasd.csv',y_pred,delimiter=',')




