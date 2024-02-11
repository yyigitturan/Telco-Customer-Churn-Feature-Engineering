import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x:  '%.3f' % x) 

df = pd.read_csv("Telco-Customer-Churn.csv")
df.shape
df.info()
df.head()

# total charges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x : 1 if x== 'Yes' else 0)
df.Churn.head()

# # KEŞİFCİ VERİ ANALİZİ

def check_df(dataframe):
    print('#############SHAPE#################')
    print(dataframe.shape)
    print('#############TYPES#################')
    print(dataframe.dtypes)
    print('################HEAD#################')
    print(dataframe.head())
    print('###############TAIL###################')
    print(dataframe.tail())
    print('###############NA######################')
    print(dataframe.isnull().sum())
    print('################QUANTILES################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)


# # numerik ve kategorik değişkenlerin yakalanması
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

num_cols, cat_cols, cat_but_car = grab_col_names(df)


cat_but_car

num_cols = [col for col in num_cols if "CUSTOMERID" not in col]

cat_but_car = "CUSTOMERID"

cat_cols = [col for col in cat_cols if "CUSTOMERID" not in col]

cat_cols

df.columns = df.columns.str.upper()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# KATEGORİK DEĞİŞKENLERİN ANALİZİ

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#############################################################')
    if plt:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, num_cols, plot=True):
        for col in num_cols:
            print("{} için temel istatistikler:".format(col)) 
            
            quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
            print(dataframe[col].describe(quantiles).T)

            if plot:
                dataframe[col].hist(bins=20)
                plt.xlabel(col)
                plt.title(col)
                plt.show(block=True) 

num_summary(df, num_cols)

# ## Numeric değişkenlerin target göre analizi

def target_summary_with_num(dataframe, target, col):
    print(dataframe.groupby(target).agg({col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "CHURN", col)


# KATEGORİK DEĞİŞKENLERİN TARGRET GÖRE ANALİZİ
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "COUNT": dataframe[categorical_col].value_counts(),
                        "RATIO": 100* dataframe[categorical_col].value_counts() / len(dataframe)}),end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "CHURN", col)

# Korelasyon,
df[num_cols].corr()

#Korelasyon matrisi
f, ax =plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# total charges ın tenure ve mothly charges  ile ilişkili olduğu görülmekte

df.corrwith(df["CHURN"]).sort_values(ascending=False)

#Feature Engineering

#Eksik değer analizi
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

from scipy.stats import ttest_ind


def Diagnose_MV_Numerical(df, str_att_name, BM_MV):
    MV_labels = {True: 'With Missing Values', False: 'Without Missing Values'}

    labels = []
    box_sr = pd.Series('', index=BM_MV.unique())
    for poss in BM_MV.unique():
        BM = BM_MV == poss
        box_sr[poss] = df[BM][str_att_name].dropna()
        labels.append(MV_labels[poss])

    plt.boxplot(box_sr, vert=False)
    plt.yticks([1, 2], labels)
    plt.xlabel(str_att_name)
    plt.show()

    plt.figure(figsize=(10, 4))

    att_range = (df[str_att_name].min(), df[str_att_name].max())

    for i, poss in enumerate(BM_MV.unique()):
        plt.subplot(1, 2, i + 1)
        BM = BM_MV == poss
        df[BM][str_att_name].hist()
        plt.xlim = att_range
        plt.xlabel(str_att_name)
        plt.title(MV_labels[poss])

    plt.show()

    group_1_data = df[BM_MV][str_att_name].dropna()
    group_2_data = df[~BM_MV][str_att_name].dropna()

    p_value = ttest_ind(group_1_data, group_2_data).pvalue

    print('p-value of t-test: {}'.format(p_value))


from scipy.stats import chi2_contingency


def Diagnose_MV_Categorical(df, str_att_name, BM_MV):
    MV_labels = {True: 'With Missing Values', False: 'Without Missing Values'}

    plt.figure(figsize=(10, 4))
    for i, poss in enumerate(BM_MV.unique()):
        plt.subplot(1, 2, i + 1)
        BM = BM_MV == poss
        df[BM][str_att_name].value_counts().plot.bar()
        plt.title(MV_labels[poss])
    plt.show()

    contigency_table = pd.crosstab(BM_MV, df[str_att_name])
    p_value = chi2_contingency(contigency_table)[1]

    print('p-value of Chi_squared test: {}'.format(p_value))


from scipy.stats import ttest_ind


def Diagnose_MV_Numerical(df, str_att_name, BM_MV):
    MV_labels = {True: 'With Missing Values', False: 'Without Missing Values'}

    labels = []
    box_sr = pd.Series('', index=BM_MV.unique())
    for poss in BM_MV.unique():
        BM = BM_MV == poss
        box_sr[poss] = df[BM][str_att_name].dropna()
        labels.append(MV_labels[poss])

    plt.boxplot(box_sr, vert=False)
    plt.yticks([1, 2], labels)
    plt.xlabel(str_att_name)
    plt.show()

    plt.figure(figsize=(10, 4))

    att_range = (df[str_att_name].min(), df[str_att_name].max())

    for i, poss in enumerate(BM_MV.unique()):
        plt.subplot(1, 2, i + 1)
        BM = BM_MV == poss
        df[BM][str_att_name].hist()
        plt.xlim = att_range
        plt.xlabel(str_att_name)
        plt.title(MV_labels[poss])

    plt.show()

    group_1_data = df[BM_MV][str_att_name].dropna()
    group_2_data = df[~BM_MV][str_att_name].dropna()

    p_value = ttest_ind(group_1_data, group_2_data).pvalue

    print('p-value of t-test: {}'.format(p_value))

BM_MV = df["TOTALCHARGES"].isna()
Diagnose_MV_Numerical(df, 'TENURE', BM_MV) # 1
Diagnose_MV_Numerical(df,'MONTHLYCHARGES', BM_MV) # p testi 0.01

df["TOTALCHARGES"].fillna(df["MONTHLYCHARGES"], inplace=True)
df.isnull().sum()

# Base Model Kurulumu
cat_cols = [col for col in cat_cols if col not in ['CHURN']]
df[cat_cols].head()

dff = df.copy()
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)
dff.head()

y = dff["CHURN"]
X = dff.drop(["CHURN","CUSTOMERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train,y_train)
y_pred = catboost_model.predict(X_test)

print('Accuracy: {}'.format(round(accuracy_score(y_pred, y_test), 4)))
print('Recall: {}'.format(round(recall_score(y_pred, y_test), 4)))
print('Precision: {}'.format(round(precision_score(y_pred, y_test), 4)))
print('f1: {}'.format(round(f1_score(y_pred, y_test), 4)))
print('Auc: {}'.format(round(roc_auc_score(y_pred,y_test), 4)))

# Aykırı değer analizi

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    IQR = quartile3-quartile1
    up_limit = 1.5* IQR + quartile3
    low_limit = quartile1 - 1.5*IQR
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return  False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# aykırı değr baskılama işlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Feature Extraction
# TENURE değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["TENURE"] >= 0) & (df["TENURE"] <= 12), "NEW_TENURE_YEAR"] = "0-1 YEAR"
df.loc[(df["TENURE"] > 12) & (df["TENURE"] <= 24), "NEW_TENURE_YEAR"] = "1-2 YEAR"
df.loc[(df["TENURE"] > 24) & (df["TENURE"] <= 36), "NEW_TENURE_YEAR"] = "2-3 YEAR"
df.loc[(df["TENURE"] > 36) & (df["TENURE"] <= 48), "NEW_TENURE_YEAR"] = "3-4 YEAR"
df.loc[(df["TENURE"] > 48) & (df["TENURE"] <= 60), "NEW_TENURE_YEAR"] = "4-5 YEAR"
df.loc[(df["TENURE"] > 60) & (df["TENURE"] <= 72), "NEW_TENURE_YEAR"] = "5-6 YEAR"

# KONTRATI 1 VEYA 2 YILLIK MÜŞTERİLERİ ENGAGED OLARAK BELİRTME
df["NEW_ENGAGED"] = df['CONTRACT'].apply(lambda x: 1 if x in ['One year', 'Two year'] else 0)

# Destek, yedek koruma almayan kişiler
df["NEW_NO_PROT"] = df.apply(lambda x: 1 if (x["ONLINEBACKUP"] != "Yes") or
                                            (x["TECHSUPPORT"] != "Yes") or
                                            (x["DEVICEPROTECTION"] != "Yes") else 0, axis=1)

# Kişinin aldığı toplam servis sayısı
df["NEW_TOTAL_SERVICES"] = (df[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                                'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES']] == "Yes").sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
# "STREAMİINGTV" yerine "STREAMINGTV" olmalı.
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["STREAMINGTV"]=="Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu
# "Credit card (automatic" yerine "Credit card (automatic)" olmalı, parantez eksik.
df["NEW_FLAG_AUTO_PAYMENT"] = df["PAYMENTMETHOD"].apply(lambda  x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# Ortalama aylık ödeme
# TENURE 0 olduğunda division by zero hatası oluşmaması için düzenleme yapıldı.
df["NEW_AVG_CHARGES"] = df.apply(lambda x: x["TOTALCHARGES"] / (x["TENURE"] + 1) if x["TENURE"] > 0 else x["MONTHLYCHARGES"], axis=1)

# Güncel Fiyatın ortalama fiyata göre artışı
# NEW_INCREASE hesaplaması için formülün yanlış olduğunu düşünüyorum. Doğrusu aşağıdaki gibi olmalı.
df["NEW_INCREASE"] = df["MONTHLYCHARGES"] / df["NEW_AVG_CHARGES"]

# Servis başına ücret
df["NEW_AVG_SERVICE_FEE"] = df["MONTHLYCHARGES"] / (df["NEW_TOTAL_SERVICES"] + 1)

# ENCODİNG
# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() ==2]
binary_cols

for col in binary_cols:
    df= label_encoder(df, col)

# ONE-HOT ENCODİNG İŞLEMİ
# CAT-COLS LİSTESİNİN GÜNCELLEME İŞLEMİ
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["CHURN", "NEW_TOTAL_SERVICES"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

# MODELLEME
y = df["CHURN"]
X = df.drop(["CHURN","CUSTOMERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

df = pd.get_dummies(df, columns=['NEW_TENURE_YEAR'])

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print('Accuracy: {}'.format(round(accuracy_score(y_pred, y_test), 2)))
print('Recall: {}'.format(round(recall_score(y_pred, y_test), 2)))
print('Precision: {}'.format(round(precision_score(y_pred, y_test), 2)))
print('f1: {}'.format(round(f1_score(y_pred, y_test), 2)))
print('Auc: {}'.format(round(roc_auc_score(y_pred,y_test), 2)))

def plot_feature_importance(importance, names,model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names':feature_names, 'feature_importance':feature_importance}
    f1_df = pd.DataFrame(data)
    f1_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    plt.figure(figsize=(15,10))
    sns.barplot(x=f1_df['feature_importance'],y=f1_df['feature_names'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

print(plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST'))