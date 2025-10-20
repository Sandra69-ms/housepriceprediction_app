import accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats.mstats  import winsorize
from scipy.stats import skew
import pickle

df = pd.read_csv('House_Pricing.csv')

num_data = df.select_dtypes(include=['int64', 'float64'])
cat_data =df.select_dtypes(include=['object'])


#handling missing value
df = df.dropna(subset=['Sale Price'])
df.loc[:,"No of Bathrooms"]=df['No of Bathrooms'].fillna(df['No of Bathrooms'].median())
df.loc[:,"Flat Area (in Sqft)"]=df['Flat Area (in Sqft)'].fillna((df['Area of the House from Basement (in Sqft)'])+(df['Basement Area (in Sqft)']))
df.loc[:,"Lot Area (in Sqft)"] = df['Lot Area (in Sqft)'].fillna(df["Lot Area after Renovation (in Sqft)"])
df = df.drop("No of Times Visited", axis =1)
df =df.dropna(subset= ["Zipcode"])
df = df.drop(["Latitude","Longitude"],axis =1)
df = df.dropna(subset=['Living Area after Renovation (in Sqft)'])
df.loc[:,"Area of the House from Basement (in Sqft)"]=df['Area of the House from Basement (in Sqft)'].fillna((df['Flat Area (in Sqft)'])-(df['Basement Area (in Sqft)']))
df= df.drop(["Basement Area (in Sqft)","Area of the House from Basement (in Sqft)"],axis=1)
df = df.drop(["Renovated Year","Lot Area (in Sqft)"],axis=1)

#outlier

q1 = np.quantile(df['No of Bedrooms'],0.25)
q3 = np.quantile(df['No of Bedrooms'],0.75)
iqr = q3 - q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(upper_bound)
print(lower_bound)
print(iqr)
bedroom_outliers = []
for i in df['No of Bedrooms']:
  if i > upper_bound or i < lower_bound:
    bedroom_outliers.append(i)
df['No of Bedrooms'] = df['No of Bedrooms'].clip(lower = lower_bound,upper= upper_bound)

q1 = np.quantile(df['No of Bathrooms'],0.25)
q3 = np.quantile(df['No of Bathrooms'],0.75)
iqr = q3 - q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(upper_bound)
print(lower_bound)
print(iqr)
bathroom_outliers = []
for i in df['No of Bathrooms']:
  if i > upper_bound or i < lower_bound:
    bathroom_outliers.append(i)
df['No of Bathrooms'] = df['No of Bathrooms'].clip(lower = lower_bound,upper= upper_bound)


q1 = np.quantile(df['Overall Grade'],0.25)
q3 = np.quantile(df['Overall Grade'],0.75)
iqr = q3 - q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(upper_bound)
print(lower_bound)
print(iqr)
overall_outliers = []
for i in df['Overall Grade']:
  if i > upper_bound or i < lower_bound:
    overall_outliers.append(i)
df['Overall Grade'] = df['Overall Grade'].clip(lower = lower_bound,upper= upper_bound)


df['Living Area after Renovation (in Sqft)'] = winsorize(df['Living Area after Renovation (in Sqft)'], limits=[0.05, 0.05])

skewness = df['Living Area after Renovation (in Sqft)'].skew()
print(f"Skewness after winsorizing: {skewness}")

df['Lot Area after Renovation (in Sqft)']=np.log(df['Lot Area after Renovation (in Sqft)'])

df['Lot Area after Renovation (in Sqft)']=winsorize(df['Lot Area after Renovation (in Sqft)'],limits=(.05,.15))

#encoding

#ordinal encoding
ordinal_enc=OrdinalEncoder(categories=[['Bad','Okay','Fair','Good','Excellent']])
df['Condition of the House']=ordinal_enc.fit_transform(df[['Condition of the House']])

ordinal_encoder=OrdinalEncoder(categories=[['No','Yes']])
df['Waterfront View']=ordinal_encoder.fit_transform(df[['Waterfront View']])

df =df.drop(['Zipcode'],axis=1)

#scaling
minmax_scale =MinMaxScaler()
df['Age of House (in Years)']=minmax_scale.fit_transform(df[['Age of House (in Years)']])
df['No of Bathrooms']=df['No of Bathrooms'].astype('int64')
df['Living Area after Renovation (in Sqft)']=minmax_scale.fit_transform(df[['Living Area after Renovation (in Sqft)']])
df['Lot Area after Renovation (in Sqft)']=minmax_scale.fit_transform(df[['Lot Area after Renovation (in Sqft)']])


df = df.drop('Date House was Sold', axis=1)
house_data = df.drop(["ID","Flat Area (in Sqft)"], axis=1)

y=df['Sale Price']
x=df.drop(columns='Sale Price')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

# Select top k features using f_regression
selector = SelectKBest(score_func=f_regression, k='all')
# You can adjust k as needed
selector.fit(x_train, y_train)

# Get the scores for each feature
feature_scores = pd.DataFrame({'Feature': x_train.columns, 'Score': selector.scores_})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Select the top features based on the scores (e.g., top 5)
selected_features = feature_scores['Feature'][:5].tolist()
print("Selected features:", selected_features)

# Create new dataframes with only the selected features
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Initialize and train the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(x_train_selected, y_train)

# Make predictions on the test set
y_pred_gb = gb_model.predict(x_test_selected)

# Evaluate the Gradient Boosting model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print("\nGradient Boosting Regressor Evaluation:")
print(f"Mean Squared Error: {mse_gb}")
print(f"Root Mean Squared Error: {rmse_gb}")
print(f"R-squared: {r2_gb}")
print(f"Mean Absolute Error: {mae_gb}")

with open('model_final.pkl', 'wb')as f:
    pickle.dump(gb_model,f)





