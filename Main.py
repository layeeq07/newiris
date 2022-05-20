import pandas as pd
df = pd.read_csv('iris.data')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = df[['sl','sw','pl','pw']]
y = df['class']

lr = LogisticRegression()

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
lr.fit(xtrain,ytrain)



from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def example(a:float,b:float,c:float,d:float):
    pred = lr.predict([[a,b,c,d]])
    return str(pred)
    