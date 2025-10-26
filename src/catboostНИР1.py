import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_breast_cancer, load_diabetes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import catboost
from matplotlib import colors

# Set random seed for reproducibility
np.random.seed(42)

# get the file  data
df = pd.read_csv('Article1.csv')

X = df.iloc[:, 1:-8]
Y = df.iloc[:, [16]]




'''X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.1, 
    random_state=34,
    shuffle=True
   
)'''

a=8 #Test sample index
test_indices = [a]  
train_indices = [i for i in range(9) if i != a]  

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = Y.iloc[train_indices] 
y_test = Y.iloc[test_indices]



model_tuned = CatBoostRegressor(
    iterations=100000,
    learning_rate=0.04,
    depth=3,
    loss_function='RMSE',
   
    feature_weights=[0.85 ,2.1 ,1.2 ,0.1 ,0.5 ,2.5 ,0.9 ,1.7],
    random_state=42,
    l2_leaf_reg=4,
   
    early_stopping_rounds=300,
     verbose=100
)

model_tuned.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    cat_features=[],
    use_best_model=False, 
    plot=True
)
y_pred_tuned = model_tuned.predict(X_test)
y_pred_all_tuned = model_tuned.predict(X)
y_pred_train_tuned = model_tuned.predict(X_train)
print(y_test.iloc[:,0] -y_pred_tuned )
#print(Y.iloc[:,:] -y_pred_all_tuned )
print(y_test.index[0])
ns=y_test.index[0]
differences = Y.copy()
avrg=0

train_error = mean_squared_error(y_train, y_pred_train_tuned)
train_error=train_error**0.5
print('train_error')
print(train_error)


if ns==0:
 model_tuned.save_model('catboost_DDist90%_0.cbm')
elif ns==1:
 model_tuned.save_model('catboost_DDist90%_1.cbm')
elif ns==2:
 model_tuned.save_model('catboost_DDist90%_2.cbm')
elif ns==3:
 model_tuned.save_model('catboost_DDist90%_3.cbm')
elif ns==4:
 model_tuned.save_model('catboost_DDist90%_4.cbm')
elif ns==5:
 model_tuned.save_model('catboost_DDist90%_5.cbm')
elif ns==6:
 model_tuned.save_model('catboost_DDist90%_6.cbm')
elif ns==7:
 model_tuned.save_model('catboost_DDist90%_7.cbm')
elif ns==8:
 model_tuned.save_model('catboost_DDist90%_8.cbm')

print(f"Models number: {ns} saved successfully")
print(f"إصدار CatBoost: {catboost.__version__}")

plt.figure(figsize=(12, 6))


plt.plot(y_pred_all_tuned, 'o-', alpha=0.7, label='التنبؤات', color='blue', markersize=4)


plt.plot(Y.values, 's-', alpha=0.7, label='Value', color='red', markersize=4)

'''plt.title('مقارنة بين القيم الحقيقية وتنبؤات النموذج')
plt.xlabel('رقم العينة')
plt.ylabel('قيمة الهدف')'''
plt.xlabel('Number of Sambel')
plt.ylabel('Value of samble of Drift_Distance@90%(m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.close()


