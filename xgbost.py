import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Load data
train = pd.read_csv('publics_train.csv')
test = pd.read_csv('publics_test.csv')

uuid_test = test['uuid']


# Simple feature engineering
def add_features(df):
    df = df.copy()
    df['dish_name_len'] = df['dish_name'].fillna('').str.len()
    df['dish_desc_len'] = df['dish_desc'].fillna('').str.len()
    df['restaurant_name_len'] = df['restaurant_name'].fillna('').str.len()
    return df

train = add_features(train)
test = add_features(test)
# Prepare data
cols_to_one = ['restaurant_district', 'restaurant_city','restaurant_type', 'dish_name', 'dish_desc','restaurant_name']

data_all = pd.concat([train.drop(columns='num_purchases'), test], axis=0)
data_all = pd.get_dummies(data_all, columns=cols_to_one, drop_first=True)
data_all = data_all.drop(columns='uuid')
data_all = data_all.fillna(0)



# Fix column names for XGBoost
data_all.columns = data_all.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')

X_train = data_all.iloc[:len(train), :]
X_test = data_all.iloc[len(train):, :]
y_train = train['num_purchases']

# Train-test split for validation
x_train, x_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2406)


params = dict(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,           
    colsample_bytree=0.8,
    tree_method="hist",
    predictor="cpu_predictor",
    n_jobs=1
)


xgb_final = xgb.XGBRegressor(random_state=2406, **params)
xgb_final.fit(X_train, y_train)
y_predict = xgb_final.predict(X_test)

#sub
output = pd.DataFrame({'uuid': uuid_test, 'num_purchases_pred': y_predict})
output.to_csv('submission.csv', index=False)