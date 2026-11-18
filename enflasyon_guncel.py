import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# --- Grafik kayit katmani (cikti katmani; model mantigina dokunmaz) ---
# Agg backend pencere acmadigi icin plt.show() normalde hicbir sey yapmaz.
# Bu katman her plt.show() cagrisinda acik grafikleri ./grafikler dizinine
# PNG olarak kaydeder; ayni grafik tekrar show edilirse dosyasi guncellenir.
import os
GRAFIK_DIZIN=os.path.join(os.path.dirname(os.path.abspath(__file__)),"grafikler")
os.makedirs(GRAFIK_DIZIN,exist_ok=True)
_orijinal_show=plt.show
_bilinen_grafikler=set()
_grafik_dosyalari={}
def _grfigleri_kaydet(*args,**kwargs):
    for num in plt.get_fignums():
        fig=plt.figure(num)
        try:
            ad=fig.axes[0].get_title()
        except Exception:
            ad=""
        ad="".join(c if c not in '\\/:*?"<>|' else "_" for c in ad).strip()[:60]
        if not ad:
            ad="grafik"
        dosya_adi=f"grafik_{num:02d}_{ad}.png"
        if num in _grafik_dosyalari and _grafik_dosyalari[num]!=dosya_adi:
            try:
                os.remove(os.path.join(GRAFIK_DIZIN,_grafik_dosyalari[num]))
            except OSError:
                pass
        fig.savefig(os.path.join(GRAFIK_DIZIN,dosya_adi),dpi=120,bbox_inches="tight")
        _grafik_dosyalari[num]=dosya_adi
        if num not in _bilinen_grafikler:
            _bilinen_grafikler.add(num)
            print("GRAFIK KAYDEDILDI:",dosya_adi)
    return _orijinal_show(*args,**kwargs)
plt.show=_grfigleri_kaydet
from datetime import date
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import linear_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid
import numpy as np
import seaborn as sns
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import itertools
import pmdarima as pm
from sklearn.pipeline import make_pipeline
from pmdarima.arima import auto_arima
from sklearn.kernel_ridge import KernelRidge		
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pmdarima.model_selection import train_test_split
import requests
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

EVDS_KEY = 'A0wX7Cuh6F'

def evds_get(seriler,startdate,enddate,frequency=5):
    """EVDS API sorgusu (evds3 uc noktasi): DataFrame dondurur (Tarih + deger kolonlari)."""
    url=('https://evds3.tcmb.gov.tr/igmevdsms-dis/series=' + '&series='.join(seriler)
         +'&startDate='+startdate+'&endDate='+enddate
         +'&type=json&frequency='+str(frequency))
    r=requests.get(url,headers={'key':EVDS_KEY},timeout=120)
    j=r.json()
    cikti=pd.DataFrame(j.get('items',[]))
    for kolon in seriler:
        cikti[kolon.replace('.','_')]=pd.to_numeric(cikti[kolon.replace('.','_')],errors='coerce')
    return cikti[['Tarih']+[s.replace('.','_') for s in seriler]]

def ay_indeksi(tarihler):
    """EVDS 'yyyy-m' tarihlerinden ay sonu indeksi uretir."""
    return pd.PeriodIndex(pd.to_datetime(pd.Series(tarihler).astype(str),format='%Y-%m'),freq='M').to_timestamp('M')

def ay_seri(dfc,deger_kolonu):
    """EVDS aylik ciktisini PeriodIndex'li sayisal Series'e cevirir."""
    s=pd.Series(pd.to_numeric(dfc[deger_kolonu],errors='coerce').values,
                index=pd.PeriodIndex(pd.to_datetime(dfc['Tarih'].astype(str),format='%Y-%m'),freq='M'))
    return s.dropna()

def zincirle(eski,yeni):
    """Iki seriyi kesisim donemindeki ortalama oranla tek seri hale getirir."""
    ortusme=eski.index.intersection(yeni.index)
    oran=float((eski.loc[ortusme]/yeni.loc[ortusme]).mean())
    return pd.concat([eski[eski.index<yeni.index.min()],yeni*oran]).sort_index()
# EVDS API istek basina 1000 satir dondurdugu icin gunluk seri 3 yillik parcalarda cekiliyor
usd_parcalar=[]
for y1 in range(2006,date.today().year+1,3):
    baslangic=("01-02-" if y1==2006 else "01-01-")+str(y1)
    bitis="31-12-"+str(min(y1+2,date.today().year))
    usd_parcalar.append(evds_get(["TP.DK.USD.S.YTL"],startdate=baslangic,enddate=bitis,frequency=2))
usd=pd.concat(usd_parcalar,ignore_index=True).drop_duplicates(subset="Tarih").sort_values("Tarih").reset_index(drop=True)
usd.dropna(inplace=True)
usd['Tarih'] = pd.to_datetime(usd['Tarih'], format='%d-%m-%Y')

# Tarihi indeks olarak ayarla
usd.set_index('Tarih', inplace=True)

# Aylık ortalama değerleri hesapla
usd = usd.resample('M').mean()

# Sonucu görüntüle
usd
enf=evds_get(["TP.FE.OKTG01"],startdate="01-02-2006",enddate=date.today().strftime("%d-%m-%Y"))
enf_yeni=evds_get(["TP.TUKFIY2025.GENEL"],startdate="01-01-2023",enddate=date.today().strftime("%d-%m-%Y"))
tufe_zincir=zincirle(ay_seri(enf,"TP_FE_OKTG01"),ay_seri(enf_yeni,"TP_TUKFIY2025_GENEL"))
inflation=pd.DataFrame({"Tarih":tufe_zincir.index.astype(str),"TÜFE":tufe_zincir.values})
inflation
dates=tufe_zincir.index.to_timestamp("M")
df = pd.DataFrame()
df["CPI"] = inflation["TÜFE"]
df = df.set_index(dates)
plt.figure(figsize=(8,5))
plt.title("Turkey CPI")
plt.plot(df)
plt.show()
df6=pd.DataFrame()
df6["USD/TL"]=usd["TP_DK_USD_S_YTL"]

df6["USD/TL"]=df6["USD/TL"].astype(float)
df6
money_eski=evds_get(["TP.PR.ARZ13"],startdate="28-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
money_yeni=evds_get(["TP.PBD.H09"],startdate="01-01-2024",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
para_zincir=zincirle(ay_seri(money_eski,"TP_PR_ARZ13")/1e+08,ay_seri(money_yeni,"TP_PBD_H09")/1e+08)
money=pd.DataFrame(para_zincir)
money.columns=["Para Arzi"]
money=money.set_index(para_zincir.index.to_timestamp("M"))
money
m3_eski=evds_get(["TP.PR.ARZ22"],startdate="28-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
m3_yeni=evds_get(["TP.PBD.H17"],startdate="01-01-2024",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
m3_zincir=zincirle(ay_seri(m3_eski,"TP_PR_ARZ22")/1e+08,ay_seri(m3_yeni,"TP_PBD_H17")/1e+08)
m3=pd.DataFrame(m3_zincir)
m3.columns=["M3 Para Arzi"]
m3=m3.set_index(money.index)
m3
issizlik=evds_get(["TP.TIG08"],startdate="28-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
issizlik.columns=["Tarih","İşsizlik"]
issizlik["İşsizlik"]=pd.to_numeric(issizlik["İşsizlik"],errors="coerce")
issizlik=issizlik.dropna(axis=0)
issizlik=issizlik.set_index(ay_indeksi(issizlik["Tarih"]))
del issizlik["Tarih"]
issizlik
import yfinance as yf
df14=yf.download("BZ=F")["Close"]
df14=pd.DataFrame(df14)
df14.columns=["Brent"]
df14=df14.resample('M').mean()
df14
kredi_eski=evds_get(["TP.KREDI.L001"],startdate="01-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
kredi_yeni=evds_get(["TP.KREHACBS.A1"],startdate="01-01-2024",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
kredi_zincir=zincirle(ay_seri(kredi_eski,"TP_KREDI_L001")/1e+08,ay_seri(kredi_yeni,"TP_KREHACBS_A1")/1e+08)
kredi=pd.DataFrame({"Tarih":kredi_zincir.index.astype(str),"Kredi":kredi_zincir.values})
kredi
df3=pd.DataFrame()
df3["Kredi"]=kredi["Kredi"]
df3["TIME"]=kredi_zincir.index.to_timestamp("M")
df3 = df3.set_index(['TIME'])
df3
pfaiz=evds_get(["TP.APIFON4"],startdate="01-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
pfaiz.columns=["Tarih","Faiz"]
pfaiz.iloc[:11,1]=17.5
pfaiz.iloc[11:23,1]=21.5
pfaiz.iloc[23:35,1]=19.25
pfaiz.iloc[35:47,1]=10.25
pfaiz.iloc[47:59,1]=8
pfaiz["Faiz"]=pfaiz["Faiz"].astype(float)
#pfaiz.iloc[212,1]=35
pfaiz
pfaiz=pd.DataFrame(pfaiz)
pfaiz.columns=["Tarih","Politika Faizi"]
pfaiz=pfaiz.set_index(ay_indeksi(pfaiz["Tarih"]))
del pfaiz["Tarih"]
pfaiz
kfaiz=evds_get(["TP.KTFTUK"],startdate="01-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
kfaiz.columns=["Tarih","Kredi Faizi"]
kfaiz=kfaiz.set_index(ay_indeksi(kfaiz["Tarih"]))
del kfaiz["Tarih"]
kfaiz=kfaiz.astype(float)
kfaiz
dosya_yolu = r"C:\Users\Administrator\Documents\InflationForecast-main\ticaret satis hacim endeksi 2021=100.xls"
perakende = pd.read_excel(dosya_yolu,header=None)
#perakende=pd.read_excel("perakendesatishacimendeksivedegisimoranlari.xls")
def _blok_oku(ham,sutun):
    blok=ham.iloc[7:,[0,1,sutun]].copy()
    blok.columns=["Yil","Ay","Deger"]
    blok["Yil"]=blok["Yil"].ffill()
    blok=blok.dropna(subset=["Ay"])
    blok=blok[pd.to_numeric(blok["Deger"],errors="coerce").notna()]
    indeks=pd.PeriodIndex(year=blok["Yil"].astype(int),month=blok["Ay"].astype(int),freq="M").to_timestamp("M")
    return pd.Series(blok["Deger"].astype(float).values,index=indeks)
perakende1=pd.DataFrame()
perakende1["Arındırılmamış"]=_blok_oku(perakende,26)
perakende1["Takvimden Arındırılmış"]=_blok_oku(perakende,28)
perakende1["Mevsimsellikten ve Takvimden Arındırılmış"]=_blok_oku(perakende,31)
perakende1
motorin=pd.read_excel("./motorin_tarihleri_tamamlanmis.xlsx")
motorin.columns=["Tarih","Motorin"]
motorin=motorin.set_index(pd.date_range(start="2003-01-01",periods=len(motorin),freq="1M"))
del motorin["Tarih"]
motorin=motorin.astype(float)
motorin
sanayi_ham=pd.read_excel("./sanayi uretim endeksi 2021=100.xls",header=None)
blok=sanayi_ham.iloc[8:,[0,1,2]].copy()
blok.columns=["Yil","Ay","Deger"]
blok["Yil"]=blok["Yil"].ffill()
blok=blok.dropna(subset=["Ay"])
blok=blok[pd.to_numeric(blok["Deger"],errors="coerce").notna()]
sanayi=pd.DataFrame(blok["Deger"].astype(float).values,
                    index=pd.PeriodIndex(year=blok["Yil"].astype(int),month=blok["Ay"].astype(int),freq="M").to_timestamp("M"),
                    columns=["Sanayi"])
sanayi
mfaiz=evds_get(["TP.TRY.MT02"],startdate="01-02-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
mfaiz["TP_TRY_MT02"]=pd.to_numeric(mfaiz["TP_TRY_MT02"],errors="coerce")
mfaiz=mfaiz.set_index(ay_indeksi(mfaiz["Tarih"]))
del mfaiz["Tarih"]
mfaiz.columns=["Mevduat Faizi"]
mfaiz
from datetime import date
enf_belirsizlik=evds_get(["TP.BEK.S01.E.S"],startdate="30-06-2013",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
enf_belirsizlik["TP_BEK_S01_E_S"]=pd.to_numeric(enf_belirsizlik["TP_BEK_S01_E_S"],errors="coerce")
enf_belirsizlik=enf_belirsizlik.set_index(ay_indeksi(enf_belirsizlik["Tarih"]))
del enf_belirsizlik["Tarih"]
enf_belirsizlik.columns=["Enflasyon Belirsizliği"]
enf_belirsizlik
borc=evds_get(["TP.KB.A09"],startdate="01-01-2006",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
borc["TP_KB_A09"]=pd.to_numeric(borc["TP_KB_A09"],errors="coerce")
borc=borc.set_index(ay_indeksi(borc["Tarih"]))
del borc["Tarih"]
borc.columns=["Borç"]
borc
redk=evds_get(["TP.RK.U01.Y"],startdate="31-03-2011",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
redk["TP_RK_U01_Y"]=pd.to_numeric(redk["TP_RK_U01_Y"],errors="coerce")
redk=redk.set_index(ay_indeksi(redk["Tarih"]))
del redk["Tarih"]
redk.columns=["REDK"]
redk
redkt=evds_get(["TP.RK.T1.Y"],startdate="31-03-2011",enddate=date.today().strftime("%d-%m-%Y"),frequency=5)
redkt["TP_RK_T1_Y"]=pd.to_numeric(redkt["TP_RK_T1_Y"],errors="coerce")
redkt=redkt.set_index(ay_indeksi(redkt["Tarih"]))
del redkt["Tarih"]
redkt.columns=["REDKT"]
redkt
data=pd.DataFrame()
data["USD/TL"]=df6["USD/TL"][:-1].rolling(3).mean()
data["ParaArzi"]=money["Para Arzi"].shift(1)
data["Motorin"]=motorin["Motorin"]
data["Perakende Satış"]=perakende1["Mevsimsellikten ve Takvimden Arındırılmış"]
data["Sanayi"]=sanayi["Sanayi"]
data["Kredi"]=df3["Kredi"].rolling(2).mean()
data["M3"]=m3["M3 Para Arzi"].shift(1)
data["Borç"]=(borc["Borç"]/1e+08).shift(1)
data["TÜFE"]=df["CPI"]
data=data.pct_change(periods=1)*100
data=data.dropna(axis=0)
data
data["Enflasyon Belirsizliği"]=enf_belirsizlik
data["Kredi Faizi"]=kfaiz["Kredi Faizi"]
data["Mevduat Faizi"]=mfaiz["Mevduat Faizi"]
data["Zam"]=0
data["Politika Faizi"]=pfaiz["Politika Faizi"]
data["İşsizlik"]=issizlik["İşsizlik"]
data["REDK"]=redk["REDK"]
data["REDKT"]=redkt["REDKT"]
yearly_mean = data.groupby(data.index.year)['TÜFE'].mean()
data['Şok'] = (data.index.year.map(yearly_mean) < data['TÜFE']).astype(int)
yearly_mean = data.groupby(data.index.year)['TÜFE'].mean()
data=data.dropna(axis=0)
data=data[['USD/TL', 'Motorin', 'Perakende Satış', 'Kredi Faizi','Zam', 'Sanayi', 'Enflasyon Belirsizliği','ParaArzi','Borç','Kredi','Mevduat Faizi','REDK','REDKT','M3','İşsizlik','Politika Faizi',"TÜFE"]]
# Asgari ucret zamlari etiket-tabanli (orijinal iloc eslemesi: data 2014-01'de basliyordu)
# Oca2014=%11.01, Oca2015=%12.3, Oca2016=%29.32, Oca2017=%7.92, Oca2018=%14.17,
# Oca2019=%25.45, Oca2020=%15, Oca2021=%26.56, Oca2022=%50, Tem2022=%30, Oca2023=%54,
# Tem2023=%34, Oca2024=%49, Tem2024=%30, 2025 ara zam yok (Oca/Tem=0), Oca2026=%27
zam_degerleri={"2014-01-31":11.01,"2015-01-31":12.3,"2016-01-31":29.32,"2017-01-31":7.92,
               "2018-01-31":14.17,"2019-01-31":25.45,"2020-01-31":15,"2021-01-31":26.56,
               "2022-01-31":50,"2022-07-31":30,"2023-01-31":54,"2023-07-31":34,
               "2024-01-31":49,"2024-07-31":30,"2025-01-31":0,"2025-07-31":0,"2026-01-31":27}
for tarih,deger in zam_degerleri.items():
    konum=pd.Timestamp(tarih)
    if konum in data.index:
        data.loc[konum,"Zam"]=deger
data
features=["USD/TL","Motorin","Enflasyon Belirsizliği","ParaArzi","Kredi Faizi","TÜFE"]
train=data[features].iloc[:-3,:]
test=data[features].iloc[-3:,:]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
test_period = 3 
test_size = test_period   

train_data = data[:-test_size]
test_data = data[-test_size:]

# Bağımsız değişkenleri ve TÜFE'yi ayırın
X_train = train.drop(['TÜFE'], axis=1).values
y_train = train['TÜFE'].values
X_test = test.drop(['TÜFE'], axis=1).values
y_test = test['TÜFE'].values

import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
import keras
keras.utils.set_random_seed(1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, activation='elu', input_shape=(X_train_scaled.shape[1],1)))
model.add(tf.keras.layers.Dense(64,activation='selu'))
model.add(tf.keras.layers.Dense(64,activation='elu'))
model.add(tf.keras.layers.Dense(1))
rms = optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=rms, loss='mean_squared_error')


early_stopping = EarlyStopping(patience=25, restore_best_weights=True)

model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=32, validation_data=(X_test_scaled, y_test_scaled),callbacks=[early_stopping])

from sklearn.metrics import mean_squared_error
import numpy as np

# Modelinizi eğittikten sonra tahminlerinizi yapın
y_pred = model.predict(X_test_scaled)

# Tahminleri gerçek değerlerle karşılaştırarak MSE hesaplayın
mse = mean_squared_error(y_test_scaled, y_pred)
rmse = np.sqrt(mse)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

y_pred_original_scale = scaler.inverse_transform(y_pred)

# Tahmin edilen değerleri inceleyin
np.sqrt(mean_squared_error(y_test,y_pred_original_scale))
          
cor_matrix=data.corr()
cor_matrix["TÜFE"].sort_values(ascending=False)  

data.to_csv("data.csv")              

for combo in combinations(features, 5):
    selected_features = list(combo)
    selected_features.append('Zam')
    if (selected_features.count('Zam')==2):
            selected_features.remove('Zam')
    print(selected_features)                                                            
    
a=['USD/TL', 'Motorin', 'Perakende Satış', 'Zam', 'Sanayi']
a.append('Zam')
print(a)

import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'Perakende Satış', 'Sanayi', 'Borç', 'Mevduat Faizi', 'REDK', 'REDKT', 'İşsizlik', 'Politika Faizi', 'Zam']
X = data[features]
X = sm.add_constant(X)
y=data[["TÜFE"]]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
model = sm.OLS(y, X).fit()
rmse=[]
tahminler=pd.DataFrame()
tahminler["Aylık Enflasyon"]=y_test
tahminler["Linear Regression"]=model.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Linear Regression"])))

from sklearn import linear_model
from sklearn.metrics import  mean_squared_error
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'Perakende Satış', 'Sanayi', 'Mevduat Faizi', 'İşsizlik', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)

tahminler["Bayesian Regression"]=reg.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Bayesian Regression"])))

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'Perakende Satış', 'Sanayi', 'Mevduat Faizi', 'İşsizlik', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]

reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train,y_train)

tahminler["Lasso Regression"]=reg.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Lasso Regression"])))

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'Perakende Satış', 'Sanayi', 'Mevduat Faizi', 'İşsizlik', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X_train, y_train)

tahminler["SGD Regressor"]=reg.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["SGD Regressor"])))

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'ParaArzi', 'Mevduat Faizi', 'REDK', 'REDKT', 'İşsizlik', 'Politika Faizi', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
reg = linear_model.Lars(random_state=0)
reg.fit(X_train,y_train)

tahminler["Lars Regression"]=reg.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Lars Regression"])))

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
warnings.filterwarnings("ignore")
features=['USD/TL', 'Motorin', 'Perakende Satış', 'Sanayi', 'Borç', 'Mevduat Faizi', 'REDK', 'REDKT', 'İşsizlik', 'Politika Faizi', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X_train, y_train) 

tahminler["Gaussian Regression"]=gpr.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Gaussian Regression"])))

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
warnings.filterwarnings("ignore")
features=['Motorin', 'Perakende Satış', 'Sanayi', 'Borç', 'Mevduat Faizi', 'REDK', 'REDKT', 'İşsizlik', 'Politika Faizi', 'Zam']
X = data[features]
X_train=X.iloc[:-3,:]
X_test=X.iloc[-3:,:]
y_train=y.iloc[:-3,:]
y_test=y.iloc[-3:,:]
krr = KernelRidge(alpha=1.0)
krr.fit(X_train, y_train)

tahminler["Kernel Regression"]=krr.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(tahminler["Aylık Enflasyon"],tahminler["Kernel Regression"])))

tahminler["LSTM"]=y_pred_original_scale
rmse.append(np.sqrt(mean_squared_error(y_test,y_pred_original_scale)))

rmse_df=pd.DataFrame(rmse,index=[tahminler.columns[1:]])
rmse_df.columns=["RMSE"]
rmse_df=rmse_df.sort_values(by="RMSE")
rmse_df
tahminler.index.strftime("%Y-%m")

tahminler=tahminler.set_index(pd.date_range(start=data.index[-3],periods=3,freq="M"))
tahminler.index=tahminler.index.strftime("%Y-%m")

tahminler["Ortalama"]=tahminler.iloc[:,1:].mean(axis=1)
tahminler[["Aylık Enflasyon","Ortalama"]].plot(kind="bar",figsize=(8,5))

model=auto_arima(pfaiz["Politika Faizi"],seasonal=False,information_criterion="bic")
model_fit = model.fit(pfaiz["Politika Faizi"])
faiztahmin = model_fit.predict(13)
faiztahmin=pd.DataFrame(faiztahmin)
faiztahmin.columns=["Politika Faizi"]
row=pd.DataFrame({"Politika Faizi":pfaiz.iloc[-5:,0]},index=pd.date_range(start=pfaiz.index[-5],periods=5,freq="1M"))
faiztahmin=pd.concat([row,faiztahmin],axis=0)
# Politika faizi yolu (etiket-tabanli): 2026-11=%36 (guncel), kademeli -0.5/ay → 2027-12=%30, sonra sabit
faiz_yolu={"2026-11":36,"2026-12":35.5,"2027-01":35,"2027-02":34.5,"2027-03":34,"2027-04":33.5,
           "2027-05":33,"2027-06":32.5,"2027-07":32,"2027-08":31.5,"2027-09":31,"2027-10":30.5,
           "2027-11":30,"2027-12":30}
faiztahmin.iloc[4:,0]=pd.Series(faiztahmin.index[4:]).dt.strftime("%Y-%m").map(faiz_yolu).ffill().bfill().astype(float).values
faiztahmin

model=auto_arima(kfaiz["Kredi Faizi"],seasonal=True,m=9)
model_fit = model.fit(kfaiz["Kredi Faizi"])

forecast = model_fit.predict(26)
kfaiztahmin=pd.DataFrame(forecast)
kfaiztahmin.columns=["Kredi Faizi"]
new_row=pd.DataFrame({"Kredi Faizi":kfaiz.iloc[-5:,0]},index=pd.date_range(start=kfaiz.index[-5],periods=5,freq="1M"))
kfaiztahmin=pd.concat([new_row,kfaiztahmin],axis=0)
kfaiztahmin

trainexog=kfaiz[47:-2]
trainexog

testexog=kfaiztahmin[2:]
testexog

model=auto_arima(mfaiz["Mevduat Faizi"],X=pfaiz,trace=False,seasonal=False,information_criterion="aic")
mfaiztahmin=pd.DataFrame(model.predict(13,X=faiztahmin[5:]))
mfaiztahmin.columns=["Mevduat Faizi"]
row=pd.DataFrame({"Mevduat Faizi":mfaiz.iloc[-2:,0]},index=pd.date_range(start=mfaiz.index[-2],periods=2,freq="1M"))
mfaiztahmin=pd.concat([row,mfaiztahmin],axis=0)
mfaiztahmin

trainexog=pd.DataFrame()
trainexog["Dolar"]=usd.iloc[-16:-6,0].rolling(2).mean().dropna()*df14.iloc[:-6,0]
trainexog=trainexog.dropna()
trainexog=np.array(trainexog).reshape(-1,1)
testexog=pd.DataFrame()
testexog["Dolar"]=usd.iloc[-7:,0].rolling(2).mean().dropna()*df14.iloc[-6:,0]
testexog=testexog.dropna()
testexog=np.array(testexog).reshape(-1,1)

exog=pd.DataFrame()
exog["X"]=usd["TP_DK_USD_S_YTL"].rolling(2).mean().dropna()*df14["Brent"]
exog=exog.dropna()
exog
#exog=np.array(exog).reshape(-1,1)

zamtahmin=pd.DataFrame()
zamtahmin["Zam"]=[0]*29
zamtahmin.columns=["Zam"]
zamtahmin=zamtahmin.set_index(pd.date_range(start=data.index[-3],periods=len(zamtahmin),freq="1M"))
# Asgari ucret zam senaryosu (etiket-tabanli): Ocak 2027=%15, diger aylar 0
zam_senaryosu={"2027-01":15}
zamtahmin["Zam"]=pd.Series(zamtahmin.index).dt.strftime("%Y-%m").map(zam_senaryosu).fillna(0).astype(float).values
zamtahmin
faiztahmin.columns=["Faiz"]
exogtrain=pd.DataFrame()
exogtrain["Faiz"]=pfaiz["Politika Faizi"].iloc[95:-14]
exogtrain=np.array(exogtrain).reshape(-1,1)

exogtest=pd.DataFrame()
exogtest=pd.concat([pfaiz["Politika Faizi"].iloc[-14:-2],faiztahmin.iloc[2:,0]],axis=0)
exogtest=np.array(exogtest).reshape(-1,1)

# Politika faizi exog'unu issizlik serisinin tarih araligina hizala (orijinal: iloc[95:-2], issizlik 2014-01'de basliyordu)
exog=pfaiz["Politika Faizi"].reindex(issizlik.index)
exog=np.array(exog).reshape(-1,1)

exogtest=faiztahmin.iloc[2:]
exogtest=np.array(exogtest).reshape(-1,1)

model=auto_arima(issizlik,seasonal=True,m=3,X=exog)
model_fit = model.fit(issizlik,X=exog)
issizliktahmin = model_fit.predict(16,X=exogtest)
issizliktahmin=pd.DataFrame(issizliktahmin)
issizliktahmin.columns=["İşsizlik"]
issizliktahmin

from prophet import Prophet
df6.shape

sn=pd.DataFrame()
sn["ds"]=df6.index
sn["y"]=df6.values
sn["cap"]=df6["USD/TL"].iloc[-1]*1.5
prophet=Prophet(growth="linear",changepoint_prior_scale=4.913477926048151,changepoint_range=0.8995647346860385,seasonality_prior_scale=5.929453787047488,n_changepoints=24,yearly_seasonality="auto",weekly_seasonality=True)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=13,freq="M")
future_dates["cap"]=df6["USD/TL"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("USD/TL")
plt.plot(predictions["yhat"],label="Tahmin")
plt.plot(df6["USD/TL"],label="Gerçek")
plt.xticks(predictions.index[10::24],rotation=45)
plt.legend()
plt.show()

usdtahmin=(predictions["yhat"][-14:].pct_change(periods=1)*100).dropna()
usdtahmin
#df6.iloc[-1,0]=df6.iloc[-1,0]+0.03  # el duzeltmesi kaldirildi (veri guncel)
usdtahmin1 = []
result = df6.iloc[-1,0]
for i in range(len(usdtahmin)):
    result *= (1 + usdtahmin[i] / 100)
    usdtahmin1.append(result)

usdtahmin1 = np.array(usdtahmin1).reshape(-1, 1)
usdtahmin1 = pd.DataFrame(usdtahmin1)
usdtahmin1.columns = ["USD/TL"]
usdtahmin1 = pd.concat([df6.iloc[-5:],usdtahmin1],axis=0)
usdtahmin1 = usdtahmin1.set_index(pd.date_range(start=df6.index[-5],periods=len(usdtahmin1),freq="1M"))
usdtahmin1

sn=pd.DataFrame()
sn["ds"]=df14[51:].index
sn["y"]=df14[51:].values
sn["cap"]=df14["Brent"].iloc[-1]*1.5
prophet=Prophet(growth="logistic",changepoint_prior_scale=9.40940230511886,changepoint_range=0.26678385022653445)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=13,freq="M")
future_dates["cap"]=df14["Brent"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])
plt.title("Brent Petrol")
plt.plot(predictions["yhat"][-25:],label="Prophet")
plt.plot(df14["Brent"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df14)

# LSTM için veriyi hazırlama
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 12  # Örnek olarak 12 aylık bir dönemi kullanıyoruz
X, y = prepare_data(df_scaled, time_steps)
train_size=-12
# Veriyi eğitim ve test setlerine bölme

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
keras.utils.set_random_seed(7)
model = Sequential()
model.add(LSTM(units=128, activation='sigmoid', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=1)

pred=scaler.inverse_transform(model.predict(X_test))
plt.figure(figsize=(12, 6))
plt.plot(df14.index[-len(y_test):], scaler.inverse_transform(y_test), label='Gerçek Değerler')
plt.plot(df14.index[-len(y_test):], pred, label='En İyi Tahminler')
plt.title('Brent Petrol Fiyatları - Gerçek vs Tahmin (LSTM)')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df14)

# LSTM için veriyi hazırlama
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 12  # Örnek olarak 12 aylık bir dönemi kullanıyoruz
X, y = prepare_data(df_scaled, time_steps)
train_size=-12
# Veriyi eğitim ve test setlerine bölme

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# generate the input and output sequences
n_lookback = 13  # length of input sequences (lookback period)
n_forecast = 13  # length of output sequences (forecast period)
X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)
keras.utils.set_random_seed(0)
# fit the model
model = Sequential()
model.add(LSTM(units=128, activation='sigmoid', return_sequences=True, input_shape=(n_lookback, 1)))
model.add(Dense(units=16, activation='sigmoid'))  # Dense katmanı eklendi
model.add(Dense(units=1))  # Çıkış birimi sayısını 1 olarak değiştirildi

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=16, verbose=1)

# generate the forecasts
X_ = y[-n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

# Tahmin yaparken çıkış şeklini düzenle
Y_ = model.predict(X_).reshape(1, n_forecast)

# Orijinal ölçekleme işlemi
Y_ = scaler.inverse_transform(Y_)

print(Y_)
Y_=np.array(Y_).reshape(-1,1)

future=pd.DataFrame()
future["Brent"]=df14["Brent"]
Y_=pd.DataFrame(Y_)
Y_.columns=["Brent"]
Y_=Y_.set_index(pd.date_range(start=(pd.Period(df14.index[-1],freq="M")+1).to_timestamp("M"),periods=13,freq="M"))
future=pd.concat([future,Y_],axis=0)
future

plt.figure(figsize=(12, 6))
#plt.plot(pred,label='Test Tahmini',color='green')
plt.plot(future["Brent"][-24:-12], label='Mevcut Veri',color='blue')
plt.plot(future["Brent"][-13:] ,label='Gelecek Tahmin',color='red')
plt.title('Brent Petrol Fiyatları - Gelecek 12 Ay Tahmini(LSTM)')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
enf_belirsizlik
enf_belirsizlik["Enflasyon Belirsizliği"][-12:]
sn=pd.DataFrame()
sn["ds"]=enf_belirsizlik[55:].index
sn["y"]=enf_belirsizlik[55:].values
sn["cap"]=0
prophet=Prophet(changepoint_prior_scale=3.646029869236057,changepoint_range=0.9986477932636241,growth='linear',n_changepoints=45,seasonality_prior_scale=9.989607383352132,yearly_seasonality=True,weekly_seasonality=False)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=13,freq="M")
future_dates["cap"]=0
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])
plt.title("Enflasyon Belirsizliği")
plt.plot(predictions["yhat"][-25:],label="Tahmin")
plt.plot(enf_belirsizlik["Enflasyon Belirsizliği"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

model=auto_arima(enf_belirsizlik,trace=False,information_criterion='aic',seasonal=True,m=5)
enfbelirsizliktahmin1=model.predict(13)
enfbelirsizliktahmin1=pd.DataFrame(enfbelirsizliktahmin1)
enfbelirsizliktahmin1.columns=["Enflasyon Belirsizliği"]
row=pd.DataFrame({"Enflasyon Belirsizliği":enf_belirsizlik.iloc[-1:,0]},index=pd.date_range(start=enf_belirsizlik.index[-1],periods=1,freq="1M"))
enfbelirsizliktahmin1=pd.concat([row,enfbelirsizliktahmin1],axis=0)
enfbelirsizliktahmin1
sn=pd.DataFrame()
sn["ds"]=perakende1["Mevsimsellikten ve Takvimden Arındırılmış"][80:].index
sn["y"]=perakende1["Mevsimsellikten ve Takvimden Arındırılmış"][80:].values
sn["cap"]=perakende1["Mevsimsellikten ve Takvimden Arındırılmış"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=3.481321992729419,changepoint_range=0.8995930316928846,growth='linear',seasonality_prior_scale=8.977376335194005,yearly_seasonality=True,weekly_seasonality=True)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=16,freq="M")
future_dates["cap"]=perakende1["Mevsimsellikten ve Takvimden Arındırılmış"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Perakende Satışlar")
plt.plot(predictions["yhat"][-28:],label="Tahmin")
plt.plot(perakende1["Mevsimsellikten ve Takvimden Arındırılmış"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

perakendetahmin=(predictions["yhat"][-29:].pct_change(periods=1)*100).dropna()
perakendetahmin
motorintahmin=pd.DataFrame((1+(((usdtahmin1[2:].pct_change(periods=1)*100).dropna())/100)).values*(1+(((future[-16:].pct_change(periods=1)*100).dropna())/100)).values)
motorintahmin.columns=["Motorin"]
# Motorin tahmini aylari: USD tahminindeki degisim aylariyla ayni pencere (orijinalde motorin son ayiyle ayniydi)
motorintahmin=motorintahmin.set_index(usdtahmin1.index[3:])
motorintahmin.iloc[0,0]=(motorin.pct_change(periods=1)*100).iloc[-1,0]
motorintahmin[1:]=((motorintahmin[1:])-1)*100
motorintahmin

sn=pd.DataFrame()
sn["ds"]=money["Para Arzi"][110:].index
sn["y"]=money["Para Arzi"][110:].values
sn["cap"]=money["Para Arzi"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=0.17670782903839727,changepoint_range=0.8995442051379563,growth='linear',seasonality_prior_scale=9.026889153868014)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=13,freq="M")
future_dates["cap"]=money["Para Arzi"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("M2 Para Arzı")
plt.plot(predictions["yhat"][-24:],label="Tahmin")
plt.plot(money["Para Arzi"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

paraarzitahmin=(predictions["yhat"].pct_change(periods=1)*100).tail(14)
paraarzitahmin

paraarzitahmin1 = []
result = money.iloc[-2,0]
for i in range(len(paraarzitahmin)):
    result *= (1 + paraarzitahmin[i] / 100)
    paraarzitahmin1.append(result)

paraarzitahmin1 = np.array(paraarzitahmin1).reshape(-1, 1)
paraarzitahmin1 = pd.DataFrame(paraarzitahmin1)
paraarzitahmin1.columns = ["Para Arzi"]
paraarzitahmin1 = paraarzitahmin1.set_index(pd.date_range(start=money.index[-1],periods=len(paraarzitahmin1),freq="1M"))
row=pd.DataFrame({"Para Arzi":money.iloc[-4:-1,0]},index=money.index[-4:-1])
paraarzitahmin1=pd.concat([row,paraarzitahmin1])
paraarzitahmin1

sn=pd.DataFrame()
sn["ds"]=borc["Borç"][103:].index
sn["y"]=borc["Borç"][103:].values
sn["cap"]=borc["Borç"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=3.671534257218686,changepoint_range=0.8995013103592469,growth='linear',seasonality_prior_scale=4.464356522636207,n_changepoints=2)

prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=15,freq="M")
future_dates["cap"]=borc["Borç"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Kamu Borç Stoku")
plt.plot(predictions["yhat"][-27:],label="Tahmin")
plt.plot(borc["Borç"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

borctahmin=(predictions["yhat"].pct_change(periods=1)*100).tail(15)
borctahmin

borctahmin1 = []
result = borc.iloc[-1,0]
for i in range(len(borctahmin)):
    result *= (1 + borctahmin[i] / 100)
    borctahmin1.append(result)

borctahmin1 = np.array(borctahmin1).reshape(-1, 1)
borctahmin1 = pd.DataFrame(borctahmin1)
borctahmin1.columns = ["Borç"]
borctahmin1 = borctahmin1.set_index(pd.date_range(start=borc.index[-1],periods=len(borctahmin1),freq="1M"))
row=pd.DataFrame({"Borç":borc.iloc[-4:,0]},index=pd.date_range(start=borc.index[-5],periods=4,freq="1M"))
borctahmin1=pd.concat([row,borctahmin1],axis=0)
borctahmin1

df3

df3=df3.iloc[:-1]
df3

sn=pd.DataFrame()
sn["ds"]=df3["Kredi"][150:].index
sn["y"]=df3["Kredi"][150:].values
sn["cap"]=df3["Kredi"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=2.10520609698171,changepoint_range=0.6734967268169046,growth='logistic',seasonality_prior_scale=4.921223978299016,n_changepoints=6,yearly_seasonality=True,weekly_seasonality='auto')

prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=14,freq="M")
future_dates["cap"]=df3["Kredi"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Kredi Hacmi")
plt.plot(predictions["yhat"][-26:],label="Tahmin")
plt.plot(df3["Kredi"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()
kreditahmin=(predictions["yhat"].pct_change(periods=1)*100).tail(16)
kreditahmin

kreditahmin1 = []
result = df3.iloc[-3,0]
for i in range(len(kreditahmin)):
    result *= (1 + kreditahmin[i] / 100)
    kreditahmin1.append(result)

kreditahmin1 = np.array(kreditahmin1).reshape(-1, 1)
kreditahmin1 = pd.DataFrame(kreditahmin1)
kreditahmin1.columns = ["Kredi"]
kreditahmin1 = kreditahmin1.set_index(pd.date_range(start=df3.index[-2],periods=len(kreditahmin1),freq="1M"))
kreditahmin1.iloc[:2,0]=df3.iloc[-2:,0]
kreditahmin1
m3=m3.iloc[:-1]
m3

sn=pd.DataFrame()
sn["ds"]=m3["M3 Para Arzi"][151:].index
sn["y"]=m3["M3 Para Arzi"][151:].values
sn["cap"]=m3["M3 Para Arzi"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=1.0239522189165728,changepoint_range=0.8631788185438879,growth='logistic',seasonality_prior_scale=8.271825948806738,n_changepoints=14,yearly_seasonality="auto",weekly_seasonality=True)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=14,freq="M")
future_dates["cap"]=m3["M3 Para Arzi"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("M3 Para Arzı")
plt.plot(predictions["yhat"][-26:],label="Tahmin")
plt.plot(m3["M3 Para Arzi"][-12:],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()
m3tahmin=predictions["yhat"].tail(17)
m3tahmin[:3]=m3.iloc[-3:,0]
m3tahmin=pd.DataFrame(m3tahmin)
m3tahmin.columns=["M3"]
m3tahmin=m3tahmin.set_index(pd.date_range(start=m3.index[-3],periods=len(m3tahmin),freq="1M"))
m3tahmin

sn=pd.DataFrame()
sn["ds"]=sanayi["Sanayi"].index
sn["y"]=sanayi["Sanayi"].values
prophet=Prophet(changepoint_prior_scale=9.619978429571834,changepoint_range=0.8872026398316396,growth='linear',seasonality_prior_scale=4.66384076344283,n_changepoints=25,yearly_seasonality="auto",weekly_seasonality=True)
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=15,freq="M")
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Sanayi Üretim Endeksi")
plt.plot(predictions["yhat"],label="Tahmin")
plt.plot(sanayi["Sanayi"],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

sanayitahmin=(predictions["yhat"].tail(16).pct_change(periods=1)*100).dropna()
sanayitahmin1 = []
result = df3.iloc[-1,0]
for i in range(len(sanayitahmin)):
    result *= (1 + sanayitahmin[i] / 100)
    sanayitahmin1.append(result)

sanayitahmin1 = np.array(sanayitahmin1).reshape(-1, 1)
sanayitahmin1 = pd.DataFrame(sanayitahmin1)
sanayitahmin1.columns = ["Sanayi"]
sanayitahmin1 = sanayitahmin1.set_index(pd.date_range(start=sanayi.index[-1],periods=len(sanayitahmin1),freq="1M"))
#row=pd.DataFrame({"Sanayi":df3.iloc[-2:,0]},index=pd.date_range(start="2023-10-30",periods=2,freq="1M"))
#sanayitahmin1=pd.concat([row,sanayitahmin1],axis=0)
sanayitahmin1

sn=pd.DataFrame()
sn["ds"]=redkt["REDKT"].index
sn["y"]=redkt["REDKT"].values
sn["cap"]=redkt["REDKT"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=4.998474885835357,
 changepoint_range=0.8719913329652468,
 growth='logistic')
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=14,freq="M")
future_dates["cap"]=redkt["REDKT"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Reel Efektif Döviz Kuru(TÜFE Bazlı)")
plt.plot(predictions["yhat"],label="Tahmin")
plt.plot(redkt["REDKT"],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

predictions["yhat"].tail(15)

redkttahmin=predictions["yhat"].tail(27)
redkttahmin=pd.DataFrame(redkttahmin)
redkttahmin.columns=["REDKT"]
redkttahmin=redkttahmin.set_index(pd.date_range(start=redkt.index[-1],periods=len(redkttahmin),freq="1M"))
redkttahmin

sn=pd.DataFrame()
sn["ds"]=redk["REDK"].index
sn["y"]=redk["REDK"].values
sn["cap"]=redk["REDK"].iloc[-1]*1.5
prophet=Prophet(changepoint_prior_scale=4.697452368413781,
 changepoint_range=0.9845493532029166,
 growth='logistic')
prophet.fit(sn)
future_dates=prophet.make_future_dataframe(periods=27,freq="M")
future_dates["cap"]=redk["REDK"].iloc[-1]*1.5
predictions=prophet.predict(future_dates)
predictions=predictions.set_index(predictions["ds"])

plt.title("Reel Efektif Döviz Kuru(ÜFE Bazlı)")
plt.plot(predictions["yhat"],label="Tahmin")
plt.plot(redk["REDK"],label="Gerçek")
plt.xticks(rotation=45)
plt.legend()
plt.show()

redktahmin=predictions["yhat"].tail(27)
redktahmin=pd.DataFrame(redktahmin)
redktahmin.columns=["REDK"]
redktahmin=redktahmin.set_index(pd.date_range(start=redk.index[-1],periods=len(redkttahmin),freq="1M"))
redktahmin
import statsmodels.api as sm
tahmindf=pd.DataFrame()
tahmindf=pd.concat([usdtahmin1["USD/TL"].rolling(3).mean(),paraarzitahmin1["Para Arzi"].shift(1),sanayitahmin1["Sanayi"],kreditahmin1.rolling(2).mean(),m3tahmin.shift(1),borctahmin1.shift(1)],axis=1)
tahmindf=tahmindf.pct_change(periods=1)*100
#tahmindf=tahmindf.dropna(axis=0)
tahmindf["Enflasyon Belirsizliği"]=enfbelirsizliktahmin1["Enflasyon Belirsizliği"]
tahmindf["Zam"]=zamtahmin["Zam"]
tahmindf["Motorin"]=motorintahmin["Motorin"]
tahmindf["Kredi Faizi"]=kfaiztahmin["Kredi Faizi"]
tahmindf["Mevduat Faizi"]=mfaiztahmin["Mevduat Faizi"]
tahmindf["İşsizlik"]=issizliktahmin
tahmindf["REDKT"]=redkttahmin["REDKT"]
tahmindf["REDK"]=redktahmin["REDK"]
#tahmindf["Politika Faizi"]=faiztahmin["Faiz"]
tahmindf["Perakende"]=perakendetahmin



tahmindf=tahmindf.iloc[:19]
tahmindf["Politika Faizi"]=pd.Series(tahmindf.index).dt.strftime("%Y-%m").map(faiz_yolu).ffill().bfill().astype(float).values
tahmindf=tahmindf[['USD/TL', 'Motorin','Perakende', 'Kredi Faizi', 'Zam','Sanayi',
       'Enflasyon Belirsizliği', 'Para Arzi', 'Borç', 'Kredi', 'Mevduat Faizi','M3','REDKT','REDK','İşsizlik','Politika Faizi']]

tahmindf.columns=["USD/TL","Motorin","Perakende Satış","Kredi Faizi","Zam","Sanayi","Enflasyon Belirsizliği",'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi','M3','REDKT','REDK','İşsizlik','Politika Faizi']
tahmindf=tahmindf.dropna(axis=0)
print("TAHMINDF UFUK:", tahmindf.index[0].date(), "->", tahmindf.index[-1].date(), "| satir:", len(tahmindf))
#tahmindf.iloc[0,1]=-4.67  # el duzeltmesi kaldirildi (veri guncel)
#tahmindf.iloc[0,0]=2.42
tahmindf

features=tahmindf.columns
df6.to_csv('df6.csv')

import keras
import keras.optimizers
prediction_list=[]
for i in range(10):
   keras.utils.set_random_seed(i)
   X = data.drop(['TÜFE'], axis=1).values
   y = data['TÜFE'].values
   X=data[features].values

   X_scaled = scaler.fit_transform(X)
   y_scaled = scaler.fit_transform(y.reshape(-1, 1))

   model = tf.keras.Sequential()
   model.add(tf.keras.layers.LSTM(128, activation='elu', input_shape=(16,1)))
   model.add(tf.keras.layers.Dense(64,activation='selu'))
   model.add(tf.keras.layers.Dense(64,activation='elu'))
   model.add(tf.keras.layers.Dense(1))
   rms = keras.optimizers.RMSprop(learning_rate=0.01)
   model.compile(optimizer=rms, loss='mean_squared_error')
   model.fit(X_scaled, y_scaled, epochs=100, batch_size=32)
   tahmin=tahmindf[features].values
   X_scaled = scaler.fit_transform(X)
   tahmin_scaled = scaler.transform(tahmin)
   pred = model.predict(tahmin_scaled)
   y_scaled = scaler.fit_transform(y.reshape(-1, 1))
   pred = scaler.inverse_transform(pred)
   prediction_list.append(pred)
columns = [f'Prediction_{i+1}' for i in range(10)]
predictions_df = pd.DataFrame(np.column_stack(prediction_list), columns=columns)
pred=predictions_df.iloc[:,:11].mean(axis=1).values
pred

corr=data.corr()["TÜFE"].sort_values(ascending=False)
np.abs(corr).sort_values(ascending=False)

tahmindf.columns

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']


X = data[features]
y=data["TÜFE"]
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=100000, tol=1e-6))
reg.fit(X, y)

tüfetahmin1 = []
result = df["CPI"].iloc[-1]
sgdtahmin=reg.predict(tahmindf[features])
for i in range(len(sgdtahmin)):
    result *= (1 + sgdtahmin[i] / 100)
    tüfetahmin1.append(result)

tüfetahmin1 = np.array(tüfetahmin1).reshape(-1, 1)
tüfetahmin1 = pd.DataFrame(tüfetahmin1)
tüfetahmin1.columns = ["SGD Regressor"]
tüfetahmin1 = tüfetahmin1.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin1

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']






X= data[features]
model=LinearRegression()
model.fit(X, y)

tüfetahmin2 = []
result = df["CPI"].iloc[-1]
lineartahmin = model.predict(np.array(tahmindf[features]))
lineartahmin=lineartahmin.ravel()
for i in range(len(lineartahmin)):
    result *= (1 + lineartahmin[i] / 100)
    tüfetahmin2.append(result)

tüfetahmin2 = np.array(tüfetahmin2).reshape(-1, 1)
tüfetahmin2 = pd.DataFrame(tüfetahmin2)
tüfetahmin2.columns = ["Linear Regression"]
tüfetahmin2 = tüfetahmin2.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin2
features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']





X = data[features]
reg = linear_model.Lasso(alpha=0.001)
reg.fit(X,y)


tüfetahmin5 = []
result = df["CPI"].iloc[-1]
lassotahmin=reg.predict(tahmindf[features])
for i in range(len(lassotahmin)):
    result *= (1 + lassotahmin[i] / 100)
    tüfetahmin5.append(result)

tüfetahmin5 = np.array(tüfetahmin5).reshape(-1, 1)
tüfetahmin5 = pd.DataFrame(tüfetahmin5)
tüfetahmin5.columns = ["Lasso Regression"]
tüfetahmin5 = tüfetahmin5.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin5

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']







X = data[features]
y=data["TÜFE"]
reg = linear_model.Lars(random_state=0)
reg.fit(X,y)



tüfetahmin6 = []
result = df["CPI"].iloc[-1]
larstahmin=reg.predict(tahmindf[features])
for i in range(len(larstahmin)):
    result *= (1 + larstahmin[i] / 100)
    tüfetahmin6.append(result)

tüfetahmin6 = np.array(tüfetahmin6).reshape(-1, 1)
tüfetahmin6 = pd.DataFrame(tüfetahmin6)
tüfetahmin6.columns = ["Lars Regression"]
tüfetahmin6 = tüfetahmin6.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin6

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']




X = data[features]
y=data["TÜFE"]
krr = KernelRidge()
krr.fit(X, y)

tüfetahmin7 = []
result = df["CPI"].iloc[-1]
krrtahmin=krr.predict(tahmindf[features])
for i in range(len(krrtahmin)):
    result *= (1 + krrtahmin[i] / 100)
    tüfetahmin7.append(result)

tüfetahmin7 = np.array(tüfetahmin7).reshape(-1, 1)
tüfetahmin7 = pd.DataFrame(tüfetahmin7)
tüfetahmin7.columns = ["Kernel Regression"]
tüfetahmin7 = tüfetahmin7.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin7

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']





X = data[features]
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel)
y=data["TÜFE"]
gpr.fit(X, y) 


tüfe = df["CPI"].iloc[-1]  
tüfetahmin8 = []
gaustahmin=gpr.predict(tahmindf[features])
gaustahmin=1+(gaustahmin/100)
for elem in gaustahmin:
    tüfe *= elem  
    tüfetahmin8.append(tüfe)
tüfetahmin8=pd.DataFrame(tüfetahmin8)
tüfetahmin8=tüfetahmin8.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin8.columns=["Gaussian Regression"]

tüfetahmin8

features=['USD/TL', 'Motorin',  'Kredi Faizi', 'Zam', 
       'Enflasyon Belirsizliği', 'ParaArzi', 'Borç', 'Kredi', 'Mevduat Faizi',
       'M3', 'REDKT']

X = data[features]
y=data["TÜFE"]
reg = linear_model.BayesianRidge()
reg.fit(X, y)


tüfe = df["CPI"].iloc[-1] 
tüfetahmin9 = []
bayestahmin=reg.predict(tahmindf[features])
bayestahmin=1+(bayestahmin/100)
for elem in bayestahmin:
    tüfe *= elem  
    tüfetahmin9.append(tüfe)
tüfetahmin9=pd.DataFrame(tüfetahmin9)
tüfetahmin9=tüfetahmin9.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin9.columns=["Bayessian Regression"]

tüfetahmin9

pred1=pd.DataFrame(pred,index=pd.date_range(start=df.index[-1],periods=len(pred),freq="1M"))
pred1=pred1[0]

tüfetahmin19 = []
result = df["CPI"].iloc[-1]
lstmtahmin=pred1
for i in range(len(lstmtahmin)):
    result *= (1 + lstmtahmin[i] / 100)
    tüfetahmin19.append(result)

tüfetahmin19 = np.array(tüfetahmin19).reshape(-1, 1)
tüfetahmin19 = pd.DataFrame(tüfetahmin19)
tüfetahmin19.columns = ["LSTM"]
tüfetahmin19 = tüfetahmin19.set_index(pd.date_range(start=df.index[-1], periods=len(tahmindf), freq="1M"))
tüfetahmin19

tüfetahmin=pd.DataFrame()
tüfetahmin["SGD Regressor"]=tüfetahmin1["SGD Regressor"]
tüfetahmin["Linear Regression"]=tüfetahmin2["Linear Regression"]
tüfetahmin["Lasso Regression"]=tüfetahmin5["Lasso Regression"]
tüfetahmin["Lars Regression"]=tüfetahmin6["Lars Regression"]
tüfetahmin["Kernel Regression"]=tüfetahmin7["Kernel Regression"]
tüfetahmin["Gaussian Regression"]=tüfetahmin8["Gaussian Regression"]
tüfetahmin["Bayessian Regression"]=tüfetahmin9["Bayessian Regression"]
tüfetahmin["LSTM"]=tüfetahmin19["LSTM"]
tüfetahmin["Ortalama"]=tüfetahmin.mean(axis=1)
tüfetahmin

dates=tufe_zincir.index.to_timestamp("M")
df=pd.DataFrame()

df["CPI"]=inflation["TÜFE"]
df=df.set_index(dates)





dfas1=pd.DataFrame()
dfas1=pd.concat([df["CPI"],tüfetahmin.iloc[:,0]],axis=0)
dfas1=pd.DataFrame(dfas1)
dfas1.columns=[tüfetahmin.columns[0]]

dfas2=pd.DataFrame()
dfas2=pd.concat([df["CPI"],tüfetahmin.iloc[:,1]],axis=0)
dfas2=pd.DataFrame(dfas2)
dfas2.columns=[tüfetahmin.columns[1]]



dfas4=pd.DataFrame()
dfas4=pd.concat([df["CPI"],tüfetahmin.iloc[:,2]],axis=0)
dfas4=pd.DataFrame(dfas4)
dfas4.columns=[tüfetahmin.columns[2]]

dfas5=pd.DataFrame()
dfas5=pd.concat([df["CPI"],tüfetahmin.iloc[:,3]],axis=0)
dfas5=pd.DataFrame(dfas5)
dfas5.columns=[tüfetahmin.columns[3]]

dfas6=pd.DataFrame()
dfas6=pd.concat([df["CPI"],tüfetahmin.iloc[1:,4]],axis=0)
dfas6=pd.DataFrame(dfas6)
dfas6.columns=[tüfetahmin.columns[4]]

dfas7=pd.DataFrame()
dfas7=pd.concat([df["CPI"],tüfetahmin.iloc[1:,5]],axis=0)
dfas7=pd.DataFrame(dfas7)
dfas7.columns=[tüfetahmin.columns[5]]

dfas8=pd.DataFrame()
dfas8=pd.concat([df["CPI"],tüfetahmin.iloc[1:,6]],axis=0)
dfas8=pd.DataFrame(dfas8)
dfas8.columns=[tüfetahmin.columns[6]]

dfas9=pd.DataFrame()
dfas9=pd.concat([df["CPI"],tüfetahmin.iloc[1:,7]],axis=0)
dfas9=pd.DataFrame(dfas9)
dfas9.columns=[tüfetahmin.columns[7]]







dfas=pd.DataFrame()
dfas=pd.concat([dfas1,dfas2,dfas4,dfas5,dfas6,dfas7,dfas8,dfas9],axis=1)
dfas=dfas.pct_change(periods=12)*100
dfas=dfas.dropna(axis=0)
dfas

df["Enflasyon"]=df["CPI"].pct_change(periods=12)*100
df=df.dropna(axis=0)

row=pd.DataFrame({"SGD Regressor":df.iloc[-1,0],"Lasso Regression":df.iloc[-1,0],"Lars Regression":df.iloc[-1,0],"Gaussian Regression":df.iloc[-1,0],"Kernel Regression":df.iloc[-1,0],"Bayessian Regression":df.iloc[-1,0],"Linear Regression":df.iloc[-1,0],"LSTM":df.iloc[-1,0],"Ortalama":df.iloc[-1,0]},index=[df.index[-2]])
tüfetahmin=pd.concat([row,tüfetahmin],axis=0)
#tüfetahmin.iloc[1,:9]=tüfetahmin.iloc[1,:9]+9  # el duzeltmesi kaldirildi (veri guncel)
#tüfetahmin.iloc[2:,:9]=tüfetahmin.iloc[2:,:9]+10  # el duzeltmesi kaldirildi (veri guncel)

tüfetahmin["Ortalama"]=tüfetahmin.mean(axis=1)
tüfetahmin

aylıktahmin=tüfetahmin.pct_change(periods=1)*100
aylıktahmin=aylıktahmin.dropna(axis=0)
aylıktahmin

aylıktahmin.to_csv('modelaylık.csv')

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin10 = []
orttahmin=aylıktahmin["Ortalama"]
orttahmin=1+(orttahmin/100)
for elem in orttahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin10.append(tüfe)
tüfetahmin10=pd.DataFrame(tüfetahmin10)
tüfetahmin10=tüfetahmin10.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin10.columns=["CPI"]

tüfetahmin11=pd.concat([df["CPI"],tüfetahmin10["CPI"]],axis=0)
tüfetahmin11=pd.DataFrame((tüfetahmin11.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin12 = []
krrtahmin=aylıktahmin["Gaussian Regression"]
krrtahmin=1+(krrtahmin/100)
for elem in krrtahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin12.append(tüfe)
tüfetahmin12=pd.DataFrame(tüfetahmin12)
tüfetahmin12=tüfetahmin12.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin12.columns=["CPI"]
tüfetahmin12=pd.concat([df["CPI"],tüfetahmin12["CPI"]],axis=0)
tüfetahmin12=pd.DataFrame((tüfetahmin12.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin13 = []
bayestahmin=aylıktahmin["SGD Regressor"]
bayestahmin=1+(bayestahmin/100)
for elem in bayestahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin13.append(tüfe)
tüfetahmin13=pd.DataFrame(tüfetahmin13)
tüfetahmin13=tüfetahmin13.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin13.columns=["CPI"]
tüfetahmin13=pd.concat([df["CPI"],tüfetahmin13["CPI"]],axis=0)
tüfetahmin13=pd.DataFrame((tüfetahmin13.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin14 = []
lassotahmin=aylıktahmin["Lasso Regression"]
lassotahmin=1+(lassotahmin/100)
for elem in lassotahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin14.append(tüfe)
tüfetahmin14=pd.DataFrame(tüfetahmin14)
tüfetahmin14=tüfetahmin14.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin14.columns=["CPI"]
tüfetahmin14=pd.concat([df["CPI"],tüfetahmin14["CPI"]],axis=0)
tüfetahmin14=pd.DataFrame((tüfetahmin14.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin15 = []
larstahmin=aylıktahmin["Lars Regression"]
larstahmin=1+(larstahmin/100)
for elem in larstahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin15.append(tüfe)
tüfetahmin15=pd.DataFrame(tüfetahmin15)
tüfetahmin15=tüfetahmin15.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin15.columns=["CPI"]
tüfetahmin15=pd.concat([df["CPI"],tüfetahmin15["CPI"]],axis=0)
tüfetahmin15=pd.DataFrame((tüfetahmin15.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin16 = []
kerneltahmin=aylıktahmin["Kernel Regression"]
kerneltahmin=1+(kerneltahmin/100)
for elem in kerneltahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin16.append(tüfe)
tüfetahmin16=pd.DataFrame(tüfetahmin16)
tüfetahmin16=tüfetahmin16.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin16.columns=["CPI"]
tüfetahmin16=pd.concat([df["CPI"],tüfetahmin16["CPI"]],axis=0)
tüfetahmin16=pd.DataFrame((tüfetahmin16.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin17 = []
bayestahmin=aylıktahmin["Bayessian Regression"]
bayestahmin=1+(bayestahmin/100)
for elem in bayestahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin17.append(tüfe)
tüfetahmin17=pd.DataFrame(tüfetahmin17)
tüfetahmin17=tüfetahmin17.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin17.columns=["CPI"]
tüfetahmin17=pd.concat([df["CPI"],tüfetahmin17["CPI"]],axis=0)
tüfetahmin17=pd.DataFrame((tüfetahmin17.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1]# İlk değeri 1691 ile başlatıyoruz
tüfetahmin18 = []
lineartahmin=aylıktahmin["Linear Regression"]
lineartahmin=1+(lineartahmin/100)
for elem in lineartahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin18.append(tüfe)
tüfetahmin18=pd.DataFrame(tüfetahmin18)
tüfetahmin18=tüfetahmin18.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin18.columns=["CPI"]
tüfetahmin18=pd.concat([df["CPI"],tüfetahmin18["CPI"]],axis=0)
tüfetahmin18=pd.DataFrame((tüfetahmin18.pct_change(periods=12)*100).tail(38))

tüfe = df["CPI"].iloc[-1] # İlk değeri 1691 ile başlatıyoruz
tüfetahmin20 = []
lstmtahmin=aylıktahmin["LSTM"]
lstmtahmin=1+(lstmtahmin/100)
for elem in lstmtahmin:
    tüfe *= elem  # Her elemanı sonuca çarp
    tüfetahmin20.append(tüfe)
tüfetahmin20=pd.DataFrame(tüfetahmin20)
tüfetahmin20=tüfetahmin20.set_index(pd.date_range(start=df.index[-1],periods=len(tahmindf),freq="1M"))
tüfetahmin20.columns=["CPI"]
tüfetahmin20=pd.concat([df["CPI"],tüfetahmin20["CPI"]],axis=0)
tüfetahmin20=pd.DataFrame((tüfetahmin20.pct_change(periods=12)*100).tail(38))
aylık=data["TÜFE"]
aylık=pd.DataFrame(aylık)
ekim=pd.DataFrame({"TÜFE":data["TÜFE"].iloc[-1]},index=[data.index[-1]])
aylık=pd.concat([aylık,ekim],axis=0)
aylık.to_csv('aylık.csv')

aylık1=aylıktahmin
aylık1=aylık1[["Ortalama"]]
aylık1.columns=["TÜFE"]
aylık1=pd.concat([aylık["TÜFE"],aylık1["TÜFE"]])
aylık1=pd.DataFrame(aylık1)
#aylık1=aylık1.set_index(pd.date_range(start="2014-01-30",end="2025-02-27",freq="1M"))
aylık1.iloc[-132:].to_csv('aylık.csv')

yıllıktahmin=pd.DataFrame()
yıllıktahmin["Ortalama"]=tüfetahmin11["CPI"]
yıllıktahmin["SGD Regressor"]=tüfetahmin13["CPI"]
yıllıktahmin["Gaussian Regression"]=tüfetahmin12["CPI"]
yıllıktahmin["Lasso Regression"]=tüfetahmin14["CPI"]
yıllıktahmin["Lars Regression"]=tüfetahmin15["CPI"]
yıllıktahmin["Kernel Regression"]=tüfetahmin16["CPI"]
yıllıktahmin["Bayessian Regression"]=tüfetahmin17["CPI"]
yıllıktahmin["Linear Regression"]=tüfetahmin18["CPI"]
yıllıktahmin["Linear Regression"]=tüfetahmin18["CPI"]
yıllıktahmin["LSTM"]=tüfetahmin20["CPI"]
plt.figure(figsize=(10,5))
plt.plot(yıllıktahmin["Ortalama"].iloc[12:24],label="Enflasyon")
plt.plot(yıllıktahmin["Gaussian Regression"].iloc[23:-1],'--',label="Gaussian Regression")
plt.plot(yıllıktahmin["SGD Regressor"].iloc[23:-1],'--',label="SGD Regressor")
plt.plot(yıllıktahmin["Lasso Regression"].iloc[23:-1],'--',label="Lasso Regression")
plt.plot(yıllıktahmin["Lars Regression"].iloc[23:-1],'--',label="Lars Regression")
plt.plot(yıllıktahmin["Kernel Regression"].iloc[23:-1],'--',label="Kernel Regression")
plt.plot(yıllıktahmin["Bayessian Regression"].iloc[23:-1],'--',label="Bayessian Regression")
plt.plot(yıllıktahmin["Linear Regression"].iloc[23:-1],'--',label="Linear Regression",color="black")
plt.plot(yıllıktahmin["LSTM"].iloc[23:-1],'--',label="LSTM",color="magenta")
plt.legend(loc=0)
plt.xticks(rotation=45)
plt.xticks(pd.date_range(start="2022-12-30",end="2025-01-27",freq="1M")[::2])
plt.title("Yıllık Enflasyon")
plt.show()

df.to_csv('df.csv')

yıllıktahmin["Maksimum"]=yıllıktahmin.iloc[:,1:10].max(axis=1)
yıllıktahmin["Minimum"]=yıllıktahmin.iloc[:,1:10].min(axis=1)
yıllıktahmin["Ortalama"]=yıllıktahmin.iloc[:,1:10].mean(axis=1)
yıllıktahmin["Medyan"]=yıllıktahmin.iloc[:,1:11].median(axis=1)
yıllıktahmin.iloc[11:-1].to_csv('C:/Users/Administrator/Documents/InflationForecast-main/yıllıktahmin.csv')
yıllıktahmin

sonu2026=str(yıllıktahmin.iloc[28,0])[:5]
tepe=str(yıllıktahmin.iloc[25,0])[:5]
sonu2027=str(yıllıktahmin.iloc[37,0])[:5]
max=str(yıllıktahmin.iloc[-2,-3])[:5]
min=str(yıllıktahmin.iloc[-2,-2])[:5]

min1=str(yıllıktahmin.iloc[-1,-2]+1)[:2]
max1=str(yıllıktahmin.iloc[-1,-3]+1)[:2]

df.iloc[-25:,0].shift(12).dropna()

cariay=evds_get(["TP.BEK.S01.A.U"],startdate=df.index[-13].strftime("01-%m-%Y"),enddate=date.today().strftime("%d-%m-%Y"))
cariay["TP_BEK_S01_A_U"]=pd.to_numeric(cariay["TP_BEK_S01_A_U"],errors="coerce")
cariay=cariay.set_index(pd.date_range(start=df.index[-13],periods=len(cariay),freq="1M"))
del cariay["Tarih"]
cariay.columns=["Cari Ay Medyan Tahmin"]
cariay["TÜFE"]=df.iloc[-14:,0].shift(1).dropna()
cariay["Cari Ay Medyan Tahmin"]=1+(cariay["Cari Ay Medyan Tahmin"]/100)
cariay["Enflasyon"]=cariay["Cari Ay Medyan Tahmin"]*cariay["TÜFE"]
cariay["TÜFE Önceki"]=df.iloc[-27:,0].shift(12).dropna()
cariay["Enflasyon"]=((cariay["Enflasyon"]/cariay["TÜFE Önceki"])-1)*100
cariay

ay1sonra=evds_get(["TP.BEK.S01.B.U"],startdate=date.today().strftime("01-%m-%Y"),enddate=date.today().strftime("%d-%m-%Y"))
ay1sonra["TP_BEK_S01_B_U"]=pd.to_numeric(ay1sonra["TP_BEK_S01_B_U"],errors="coerce")
ay1sonra["TÜFE"]=df["CPI"].iloc[-1]
ay1sonra["TÜFE Yeni"]=(1+(ay1sonra["TP_BEK_S01_B_U"]/100))*ay1sonra["TÜFE"]
ay1sonra["TÜFE Eski"]=df["CPI"].iloc[-11]
ay1sonra["Enflasyon"]=((ay1sonra["TÜFE Yeni"]/ay1sonra["TÜFE Eski"])-1)*100
ay1sonra

ay2sonra=evds_get(["TP.BEK.S01.C.U"],startdate=date.today().strftime("01-%m-%Y"),enddate=date.today().strftime("%d-%m-%Y"))
ay2sonra["TP_BEK_S01_C_U"]=pd.to_numeric(ay2sonra["TP_BEK_S01_C_U"],errors="coerce")
ay2sonra["TÜFE"]=ay1sonra["TÜFE Yeni"]
ay2sonra["TÜFE Yeni"]=(1+(ay2sonra["TP_BEK_S01_C_U"]/100))*ay2sonra["TÜFE"]
ay2sonra["TÜFE Eski"]=df["CPI"].iloc[-10]
ay2sonra["Enflasyon"]=((ay2sonra["TÜFE Yeni"]/ay2sonra["TÜFE Eski"])-1)*100
ay2sonra

ay12sonra=evds_get(["TP.BEK.S01.E.U"],startdate=date.today().strftime("01-%m-%Y"),enddate=date.today().strftime("%d-%m-%Y"))
ay12sonra["TP_BEK_S01_E_U"]=pd.to_numeric(ay12sonra["TP_BEK_S01_E_U"],errors="coerce")
ay12sonra.columns=["Tarih","12 Ay Sonrası"]
ay12sonra

import matplotlib.dates as mdates  # datetime işlevlerini içe aktarın

# ...

plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(15,8))
plt.scatter(x=df.index[-1]+pd.offsets.MonthEnd(2),y=ay1sonra["Enflasyon"],color="green",zorder=3,s=50)
plt.scatter(x=df.index[-1]+pd.offsets.MonthEnd(3),y=ay2sonra["Enflasyon"],color="green",zorder=3,s=50)
plt.scatter(x=cariay[1:].index,y=cariay["Enflasyon"][1:].values,color="green",zorder=3,label='TCMB Cari Ay Tahmini',s=50)
plt.scatter(x=df.index[-1]+pd.offsets.MonthEnd(13),y=ay12sonra["12 Ay Sonrası"].values,color="green",zorder=3,s=50)
#plt.scatter(x=yıllıktahmin.index[21:24],y=[61.94,60.84,62.13],color='black',zorder=3,label='Geçmiş Tahminler',s=50)  # bayat el degerleri kaldirildi
plt.scatter(x=yıllıktahmin.index[28],y=yıllıktahmin.iloc[28,0],color='orange',zorder=3,s=50)
plt.scatter(x=yıllıktahmin.index[25],y=yıllıktahmin.iloc[25,0],color='orange',zorder=3,s=50)
plt.scatter(x=yıllıktahmin.index[37],y=yıllıktahmin.iloc[37,0],color='orange',zorder=3,s=50)
#plt.scatter(x=yıllıktahmin.index[36],y=yıllıktahmin.iloc[36,0],color='orange',zorder=3,s=50)
plt.plot(yıllıktahmin["Ortalama"].iloc[24:38],'--',label="Ortalama Tahmin",linewidth=7)
plt.plot(yıllıktahmin["Ortalama"].iloc[13:25],label="Enflasyon",linewidth=7)
plt.fill_between(yıllıktahmin.iloc[24:38].index.strftime("%Y-%m"), yıllıktahmin["Maksimum"].iloc[24:38].shift(1), yıllıktahmin["Minimum"].iloc[24:38].shift(1), alpha=0.5, color='gray')
# Tarihi ekseni birimlerine dönüştürün
annotation_date = mdates.datestr2num('2026-12')
plt.text(mdates.datestr2num('2027-01'), float(sonu2026)-3, sonu2026, fontsize=18, color='black', horizontalalignment='center',fontweight='bold')
plt.text(mdates.datestr2num('2026-11'), float(tepe)+1, tepe, fontsize=18, color='black', horizontalalignment='right',fontweight='bold')
plt.text(mdates.datestr2num('2027-10'), float(sonu2027)+2, sonu2027, fontsize=18, color='black', horizontalalignment='right',fontweight='bold')
#plt.text(mdates.datestr2num('2025-12'), float(sonra24ay)+3, sonra24ay, fontsize=12, color='black', horizontalalignment='right',fontweight='bold')

plt.text(annotation_date, 32, f"Enflasyonun 2027 sonunda %{min1}-%{max1} aralığında olacağı tahmin edilmektedir.", fontsize=21, color='black', horizontalalignment='right',fontweight='bold')
plt.xticks(yıllıktahmin.index[10:37:2],rotation=45)
plt.yticks(np.arange(20,40,5))
plt.title("Türkiye Enflasyon(Yıllık %)")
plt.legend(loc=2)
plt.show()

import matplotlib.axes as ax
gecmis=pd.DataFrame({
    "Yıllık Enflasyon(%)":(df["CPI"].pct_change(periods=12)*100).iloc[-3:].round(2),
    "Beklenti(%)":cariay["Enflasyon"].iloc[-4:-1].round(2)
})
gecmis=gecmis.set_index(gecmis.index.strftime("%Y-%m"))
gecmis.plot(kind="bar")
plt.legend()
plt.xticks(rotation=0)
plt.show()  # son grafik de kayit katmanina yazilir (blokta show yoktu)



