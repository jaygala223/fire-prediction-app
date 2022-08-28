import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('./forestfires.csv')

df.head()
df.rename(columns={'area': 'fire_occured'},
          inplace=True, errors='raise')
df.head()

df.info()

df.describe()

df.drop(['X','Y','FFMC','DMC','DC'], axis=1, inplace=True)
print(df.head())


df.drop(['ISI','day','month'], axis=1, inplace=True)

#for setting values of non-negative areas to 1
def impute_fire_occurence(col):
    fire=col[0]
    if(fire > 0.0):
        return 1
    else: return fire

df['fire_occured'] = df[['fire_occured']].apply(impute_fire_occurence,axis=1)

y = df['fire_occured'].astype('int')
print(y.head(500))

X = df.drop('fire_occured', inplace=True, axis=1)

X = df
X.drop('rain', inplace=True, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.10, random_state=42)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train,y_train)
#LR.predict(X_test,y_test) #Return the predictions

#print(LR.score(X_train, y_train)) #Return the mean accuracy on the given test data and labels

#ypred = LR.predict(X_test)

#r2_score(y_test, ypred), mean_absolute_error(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred))

pickle_out = open(('model.pkl','wb'))
pickle.dump(LR, open('model.pkl','wb'))
pickle_out.close()