import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn.feature_selection import RFEC

train_df = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')

# utils
def Convert(string):
    list1=[]
    list1[:0]=string
    return list1

def letter_to_int(letters):
    temp_=[]
    for i in letters:
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        temp_.append(alphabet.index(i.lower()) + 1)
    return

# Splitting the word and converting it to their corresponding number
train_df[['f_270','f_271','f_272','f_273','f_274','f_275','f_276','f_277','f_278','f_279']]=train_df.apply(lambda x: letter_to_int(Convert(x['f_27'])),axis=1,result_type="expand")
# Number of unique values
train_df["f_27"]=train_df["f_27"].apply(lambda x: len(set(x)))

# dropping unwanted columns
X = train_df.iloc[:, :].drop(['target', 'id'],axis=1)
y = train_df.iloc[:,32]

# we don't need to scale this as we're going with lightbgm
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=69)

# Declaring Model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Parameters
params = {
        'task': 'prediction',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.08,1
        'max_depth': -1,
        'num_leaves': 42, 
        'min_data_in_leaf': 22,
        'bagging_fraction': 1,
        'bagging_freq': 0,
        'feature_fraction': 1,
        'lambda_l1': 0.3,
        'lambda_l2': 0.3,
        'num_iterations': 20000,
        'verbosity': -1,
        'device':'gpu'
}


model_lgb = lgb.train(
    params,
    train_set=lgb_train,
    valid_sets=lgb_test,
    early_stopping_rounds=300,
    verbose_eval=100
)

lgb_predicted = model_lgb.predict(X_test)

# Got like 99% AUC score and 99.28 on submitted result
# The plotted diagram will be in image format
print(roc_auc_score(y_test, lgb_predicted))

fpr, tpr, _ = metrics.roc_curve(y_test, lgb_predicted)
auc = metrics.roc_auc_score(y_test, lgb_predicted)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Another approach which yielded good results (99%) is RFE where the Most important features are Forward selected
# Sample code is given below
#Note: i didn't experiment with it and you can also RFECV which will take more time to train and select best features

rfe = RFE(estimator=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metric= 'auc',
                                     learning_rate=0.01,num_leaves=40), n_features_to_select=10, verbose = 2)
rfe.fit(X, y)

X_rfe = rfe.transform(X)

# Hyperparameter tuning to try out
 # boosting type like goss and rf can be used.. make sure to try out dart to improve accuracy
 # num_leaves and min_data_in_leaf can be altered but make sure not to voerfit the data .... tune the elarning rate and regularisation paramters along with it
