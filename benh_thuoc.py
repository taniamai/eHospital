import pyodbc
import pandas as pd
import collections
import itertools
import random
import matplotlib.pyplot as plt

######################################################################################
# connect to database SQL and import data

connection = pyodbc.connect(r'Driver={SQL Server};Server=DESKTOP-LI36RK3;'
                            r'Database=eHospital_DongNai_A;UID=sa;PWD=123456;Trusted_Connection=yes;')

# import data kham benh - toa thuoc
sql = "SELECT K.KhamBenh_Id, K.BenhNhan_Id, K.ChanDoanICD_Id, K.ChanDoanPhuICD_Id ,T.ToaThuoc_Id,T.Duoc_Id, T.SoLuong, T.SoNgay FROM dbo.KhamBenh K INNER JOIN dbo.ToaThuoc T ON K.KhamBenh_Id = T.KhamBenh_Id WHERE K.ChanDoanICD_Id is not null "

dat = pd.read_sql(sql, connection)
#dat.shape
#dat.head()
print(dat.head())
# import danh muc Duoc va danh muc ma benh (ICD)
connection2 = pyodbc.connect(r'Driver={SQL Server};Server=DESKTOP-LI36RK3;'
                             r'Database=eHospital_DongNai_A_Dictionary;'
                             r'UID=sa;PWD=123456;Trusted_Connection=yes;')


df_ICD = pd.read_sql("SELECT ICD_Id, MaICD, TenICD FROM dbo.DM_ICD", connection2)

print(df_ICD.head())

df_Duoc = pd.read_sql("SELECT Duoc_Id, MaDuoc, TenDuocDayDu FROM dbo.DM_Duoc", connection2)

#print(df_Duoc.head())

#df_benhnhan = pd.read_sql("SELECT BenhNhan_Id, NamSinh FROM dbo.DM_BenhNhan",connection2)
########################################################################################

dat["ChanDoanICD_Id"] =[int(i) for i in dat["ChanDoanICD_Id"]]

df = pd.merge (dat, df_Duoc, on = "Duoc_Id")
df = pd.merge (df, df_ICD, left_on='ChanDoanICD_Id', right_on='ICD_Id')
#df = pd.merge (df,df_benhnhan, on="BenhNhan_Id")

df.to_csv("BenhNhan_ToaThuoc_DNA.csv",sep=',',encoding='utf8')

#df["Tuoi"] = 2017 - df["NamSinh"]
#df1 = df[df.ICD_Id == 4079]
#len(df1)

#pd.DataFrame(df1.groupby(["Duoc_Id", "Tuoi", "SoLuong"]).size().reset_index())
#df2 = df1[df1.Duoc_Id == 5752]
#plt.plot(df2.Tuoi,df2.SoLuong,'ro')
#plt.ylabel('SoLuong')
#lt.xlabel('Tuoi')


###### list cac thuoc hay duoc ke dua tren chan doan ICD chinh
# list cac benh da tung gap it nhat 10 lan
list_ICD = collections.Counter(df["ChanDoanICD_Id"])

for k in list(list_ICD):
    if list_ICD[k] < 10:
        del list_ICD[k]
list_ICD.most_common()


# list cac thuoc hay duoc ke nhat voi moi ma benh
for b in list_ICD.most_common(3):
    print(df_ICD[df_ICD["ICD_Id"] == b[0]]["TenICD"])
    thuoc = collections.Counter(df[df["ChanDoanICD_Id"] == b[0]]["TenDuocDayDu"])

########################################################################################
### nhap ma benh
while True:
    maicd = input("Nhập mã ICD (Enter to quit):  ")
    icdid = df_ICD[df_ICD.MaICD == maicd]['ICD_Id']
    if not icdid:
        break
    if int(icdid) not in list(df["ChanDoanICD_Id"]) :
        print('Mã bệnh không tồn tại')
        print("Ví dụ mã bệnh : %s" %random.sample(list(dat["ChanDoanICD_Id"]),5))
    else :
        df_benh = df[df["ChanDoanICD_Id"] == int(icdid)]
        print('Các thuốc hay được kê với bệnh %s là :' %(df_ICD[df_ICD["ICD_Id"] == int(icdid)]["TenICD"].item()) )
        thuoc = collections.Counter(df_benh["MaDuoc"])
        for k in list(thuoc):
            if thuoc[k] < 3:
                del thuoc[k]
        if len(thuoc) >5 :
            print(list(thuoc.most_common(5)))
        else :
            print(list(thuoc.most_common(len(thuoc))))



### cac thuoc hay duoc ke cung nhau nhat
        donthuoc = pd.DataFrame(df_benh.groupby(['KhamBenh_Id'])['TenDuocDayDu'].apply(list).reset_index())
        print("Các thuốc hay đươc kê cùng nhau nhất là :")
        d = collections.Counter()
        for collab in donthuoc['TenDuocDayDu']:
            if len(donthuoc) < 2:
                continue
            collab.sort()
            for comb in itertools.combinations(collab, 2):
                d[comb] += 1
        if len(d) > 5:
            print(list(d.most_common(5)))



########################################################################################
# nhap ten benh
while True:
    tenbenh = input("Nhập tên bệnh (Enter to quit):  ")
    if not tenbenh:
        break
    if not any(tenbenh in s for s in list(df_ICD["TenICD"])):
        print("Tên bệnh không tồn tại: ")
        print(df_ICD["TenICD"].head(10))
    else:
        while tenbenh not in list(df_ICD["TenICD"]):
            print ("Danh sách bệnh : ")
            print([s for s in list(df_ICD["TenICD"]) if tenbenh in s])
            tenbenh = input("Tên bệnh đầy đủ (Enter to quit): ")
            if not tenbenh:
                break
        else:
            icdid = df_ICD[df_ICD["TenICD"] == tenbenh]["ICD_Id"].item()
            print ("Danh sách thuốc : ")
            thuoc = collections.Counter(df[df["ChanDoanICD_Id"] == icdid]["TenDuocDayDu"])
            print(list(thuoc.most_common(5)))




##################################################################################


icdid = df_ICD[df_ICD.MaICD == maicd]['ICD_Id']
df_benh = df[df["ChanDoanICD_Id"] == int(icdid) & df["ChanDoanPhuICD_Id"] ==]

print("Don thuoc mau :")

donthuoc1 = pd.DataFrame(df_benh.groupby(['KhamBenh_Id'])['MaDuoc'].apply(list).reset_index())
d1 = collections.Counter()
for collab in donthuoc1["MaDuoc"] :
    d1[str(collab)] += 1
print(list(d1.most_common(5)))

mt1 = d1.most_common(1)

listkb = donthuoc[donthuoc.MaDuoc == mt1[0][0]]
