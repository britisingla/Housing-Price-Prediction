import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('ParisHousing.csv')
data.info()

data.shape

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


models = {
    # 'Linear Regression': LinearRegression(),
    # 'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42),
    # 'SVR': SVR(),
    # 'Decision Tree': DecisionTreeRegressor(random_state=42),
    # 'KNN': KNeighborsRegressor(),
    # 'Bayesian Ridge': BayesianRidge(),
    # 'Gradient Boosting Regression': GradientBoostingRegressor(),
    # 'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    # 'Elastic Net': ElasticNet(),
    # 'AdaBoost Regression': AdaBoostRegressor(),
    # 'Extra Trees Regression': ExtraTreesRegressor(),
    # 'XGBoost Regression': XGBRegressor()
}

# Train and evaluate each model
model = None
best_mse = float('inf')

for name, model in models.items():
    # Evaluate the model using cross-validation with Mean Squared Error
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mse = -scores.mean()
    print(f"{name} - Mean Squared Error: {mse}")

    if mse < best_mse:
        model = model
        best_mse = mse

print(f"\nBest Model based on MSE: {model}")

model.fit(X, y)

pickle.dump(model,open('model.pkl', 'wb'))

# vals = [
#     2000,
#     5,
#     1,
#     1,
#     2,
#     9397,
#     7,
#     1,
#     2000,
#     0,
#     0,
#     1,
#     1,
#     1,
#     1,
#     1
# ]

# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([vals]))