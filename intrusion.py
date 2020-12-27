import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing the training data
data=pd.read_csv("Train_Data.csv")
data.info()

#applying label encoding for object type features
le=LabelEncoder()
for i in data.columns:
    if data[i].dtype=='O':
        data[i]=le.fit_transform(data[i])
data.info()

data["class"].value_counts() #normal=1,anomaly=0

#splitting into dependent and independent features
x=data.drop(["class"],axis=1)
x
y=data["class"]
y

#standardization of data

scaler=StandardScaler()
scale_x=scaler.fit_transform(x)
scaled_x=pd.DataFrame(scale_x,columns=x.columns)

#generating coorelation matrix and droping corelated data feature

scaled_x_corr_df=scaled_x.corr()
col_corr=set()
for i in range(len(scaled_x_corr_df.columns)):
    for j in range(i):
        if (scaled_x_corr_df.iloc[i,j]>=0.8) and (scaled_x_corr_df.columns[j] not in col_corr):
            colname=scaled_x_corr_df.columns[i]
            col_corr.add(colname)
col_corr=list(col_corr)

new_x=scaled_x.drop(col_corr,axis=1)

#spliting data into test-train case

X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.25, random_state=42)

#training thru model

model=LogisticRegression(random_state=45).fit(X_train, y_train)
y_pred=model.predict(X_test)

#accuracy score
print(accuracy_score(y_test,y_pred))

