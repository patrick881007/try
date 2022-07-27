import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow

data=["gongguan_underfitting.csv","gongguan_best.csv","gongguan_overfitting.csv"]
for i in data:
    with mlflow.start_run(run_name=i):
        bike=pd.read_csv(i)
        x=bike.drop(["lent"],axis=1) #自變數
        y=bike["lent"]               #應變數

        train_x, valid_x, train_y, valid_y = train_test_split(x,y,test_size=0.3)

        lm=LinearRegression()
        lm.fit(train_x,train_y) #用訓練資料建構回歸模型
        rsquare=lm.score(train_x,train_y)

        predicted_y=lm.predict(valid_x) #代入驗證資料集的自變數，求得預測值
        rss=((predicted_y - valid_y)**2).mean()
        tss=((valid_y.mean()-valid_y)**2).mean()
        verror=1-(rss/tss)

        mlflow.log_param("num",len(x.columns))
        mlflow.log_metric("rsquare",rsquare) #訓練誤差
        mlflow.log_metric("verror",verror) #驗證誤差
        mlflow.sklearn.log_model(lm,"model")