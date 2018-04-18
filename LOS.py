
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import scipy.stats
from sklearn.metrics import explained_variance_score, r2_score
######################################################################################
'''
# connect to database SQL and import data

connection = pyodbc.connect(r'Driver={SQL Server};Server=DESKTOP-LI36RK3;'
                            r'Database=eHospital_NDGiaDinh;UID=sa;PWD=123456;Trusted_Connection=yes;')

# import data kham benh - toa thuoc
sql = "SELECT BenhAn_Id, SoNgayDieuTri, ICD_BenhChinh, ICD_BenhPhu FROM dbo.BenhAn "
# NgayVaoVien, NgayRaVien
dat = pd.read_sql(sql, connection)

dat.head()
'''

######################################################################################
dat = pd.read_csv("data/SoNgayDieuTri_QN.txt", sep='\t')
dat.head()

dat["Tuoi"] = datetime.date.today().year - dat["NamSinh"]
#dat.drop(["NamSinh"], axis=1)
#dat.head()

dat = dat[(dat.SoNgayDieuTri.isnull()== False) & (dat["SoNgayDieuTri"]<30) & (dat["SoNgayDieuTri"]>0)]

dat.shape
#Check the missing values
dat.isnull().sum()

# convert variable
dat["ICD_BenhChinh"] = np.nan_to_num(dat['ICD_BenhChinh']).astype(int)
#dat["ICD_VaoKhoa"] = np.nan_to_num(dat['ICD_VaoKhoa']).astype(int)
dat["KhoaVao_Id"] = np.nan_to_num(dat['KhoaVao_Id']).astype(int)
dat["LyDoNhapVien_Id"] = np.nan_to_num(dat['LyDoNhapVien_Id']).astype(int)
dat["nbXN"] = np.nan_to_num(dat['nbXN']).astype(int)

#dat["ICD_BenhChinh"] = dat["ICD_BenhChinh"].apply(lambda x: int(x) if np.isnan(x) == False else x)


#correlation
#dat["SoNgayDieuTri"].corr(dat["Tuoi"])

#for namevar in ["KhoaRa_Id","KhoaVao_Id","ICD_BenhChinh","ICD_BenhPhu","DoiTuong_Id"] :
#    dat[namevar] = dat[namevar].astype('category')

#### so xet nghiem CLS
#Counter(dat["DoiTuong_Id"])
#Counter(dat["KhoaRa_Id"])
sns.countplot(x="DoiTuong_Id",data=dat)
#plt.show()

######################################################################################
# data preparation
dat["GioiTinh"] = dat["GioiTinh"].apply(lambda x: 1 if x=="T" else 0)
dat.corr()
dat.columns
X = dat[["GioiTinh","Tuoi","KhoaVao_Id","ICD_BenhChinh","DoiTuong_Id","LyDoNhapVien_Id","LoaiBenhAn_Id","nbXN"]]
y = dat["SoNgayDieuTri"]

X2 = pd.get_dummies(X,columns = ["KhoaVao_Id","ICD_BenhChinh","DoiTuong_Id","LyDoNhapVien_Id","LoaiBenhAn_Id"])
X2 = X2.values
y = y.values
dat["GioiTinh"].value_counts()


X_train,X_test,y_train,y_test = train_test_split(X2, y, test_size=0.33, random_state = 5)
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train =  scaler.transform(X_train)
X_test =  scaler.transform(X_test)
'''

######################################################################################
# model
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
ytrain_predict = lm.predict(X_train)
ytest_predict = lm.predict(X_test)

#######
modelrf = RandomForestRegressor(n_estimators = 100, random_state = 42) # oob_score=True,

modelrf.fit(X_train, y_train)
ytest_predict = modelrf.predict(X_test)



#plt.plot(y_test)
sum(abs(y_test - ytest_predict))/len(y_test)
sum(abs(y_test - ytest_predict)<1)/len(y_test)
sum(abs(y_test - ytest_predict)<2)/len(y_test)
sum(abs(y_test - ytest_predict)>3)/len(y_test)

test = pd.DataFrame ({'y_test': y_test,'ytest_predict': ytest_predict})
print(test)

print (scipy.stats.spearmanr(y_test,ytest_predict))

print ('Pearson correlation ' , scipy.stats.pearsonr(y_test,ytest_predict))

r2_score(y_test,ytest_predict)


#from sklearn.metrics import accuracy_score
#accuracy_score (y_test,ytest_predict)
#print ('RMSE / average' , np.sqrt(((ytest_predict - y_test) ** 2).mean())/np.mean(y_test))

#print ('RMSE ' , np.sqrt(((ytest_predict - y_test) ** 2).mean()))
