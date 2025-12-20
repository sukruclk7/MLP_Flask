import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# =========================================================
# 1. VERİ SETİNİN OLUŞTURULMASI
# =========================================================
# Bu çalışmada, proje gereksinimlerine uygun olacak şekilde
# sentetik (yapay) bir futbolcu veri seti oluşturulmuştur.
# Gerçek veri yerine sentetik veri kullanılması, modelleme
# sürecinin kontrol edilebilir ve tekrarlanabilir olmasını sağlar.

data = {
    'overall_rating': np.random.randint(50, 95, 200),     # Oyuncunun genel yetenek puanı
    'age': np.random.randint(18, 38, 200),                # Oyuncu yaşı
    'goals': np.random.randint(0, 40, 200),               # Gol sayısı
    'assists': np.random.randint(0, 25, 200),             # Asist sayısı
    'position': np.random.choice(
        ['Forvet', 'Orta Saha', 'Defans', 'Kaleci'], 200
    ),                                                     # Oyuncu pozisyonu (kategorik)
    'jersey_number': np.random.randint(1, 99, 200)        # Forma numarası (bilinçli olarak gereksiz değişken)
}

# DataFrame oluşturulması
df = pd.DataFrame(data)

# Hedef değişken olan piyasa değeri (market_value) hesaplanmaktadır.
# Değer hesaplanırken oyuncu yeteneği, yaşı, gol sayısı ve pozisyonu
# dikkate alınmıştır.
df['market_value'] = 0

for index, row in df.iterrows():
    base_price = (
        (row['overall_rating'] * 1_000_000)
        - (row['age'] * 500_000)
        + (row['goals'] * 200_000)
    )

    # Forvet oyuncularının piyasa değerine pozisyon bazlı ek katkı
    if row['position'] == 'Forvet':
        base_price += 5_000_000

    # Gerçekçi dağılım için rastgele gürültü (noise) eklenmesi
    noise = np.random.randint(-2_000_000, 2_000_000)

    # Piyasa değerinin negatif olmaması sağlanmaktadır
    df.at[index, 'market_value'] = max(base_price + noise, 500_000)

# Veri seti CSV dosyası olarak kaydedilir
df.to_csv('futbolcu_verisi.csv', index=False)
print("Veri seti başarıyla oluşturuldu ve 'futbolcu_verisi.csv' dosyasına kaydedildi.\n")

# =========================================================
# 2. VERİ ÖN İŞLEME (DATA PREPROCESSING)
# =========================================================
print("=== VERİ ÖN İŞLEME AŞAMASI ===")

# 2.1 Eksik Veri Analizi
# Veri setindeki eksik değerler kontrol edilmektedir
print("Eksik veri sayıları:")
print(df.isnull().sum(), "\n")

# Eksik değerler olması durumunda, sayısal sütunlar için
# ortalama (mean) değer ile doldurma işlemi uygulanır
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2.2 Kategorik Verilerin Sayısal Hale Getirilmesi
# Makine öğrenmesi algoritmaları kategorik verilerle doğrudan
# çalışamadığı için Label Encoding yöntemi kullanılmıştır
le = LabelEncoder()
df['position_encoded'] = le.fit_transform(df['position'])

# Orijinal kategorik sütun modelleme aşamasında kullanılmayacaktır
df_model = df.drop(columns=['position'])

# =========================================================
# 3. GERİYE DOĞRU ELEME (BACKWARD ELIMINATION)
# =========================================================
# Bu aşamada, istatistiksel olarak anlamlı olmayan değişkenleri
# tespit etmek amacıyla OLS (Ordinary Least Squares) modeli kurulmuştur

print("\n=== BACKWARD ELIMINATION AŞAMASI ===")

X = df_model.drop(columns=['market_value'])  # Bağımsız değişkenler
y = df_model['market_value']                 # Bağımlı değişken

# OLS modeli için sabit terim eklenir
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()

# Model özet tablosu (p-value değerleri dahil)
print(ols_model.summary())

# p-value değeri 0.05'ten büyük olan değişkenler istatistiksel
# olarak anlamsız kabul edilir ve modelden çıkarılır
p_values = ols_model.pvalues
max_p_value = p_values.max()

if max_p_value > 0.05:
    removed_feature = p_values.idxmax()
    print(f"\nElenecek değişken: {removed_feature} (p-value = {max_p_value:.4f})")
    X = X.drop(columns=[removed_feature])
else:
    print("\nTüm değişkenler istatistiksel olarak anlamlıdır.")

# =========================================================
# 4. MODEL EĞİTİMİ VE DEĞERLENDİRME
# =========================================================
print("\n=== MODEL EĞİTİMİ VE DEĞERLENDİRME ===")

# Veri setinin eğitim ve test olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Çoklu doğrusal regresyon modelinin oluşturulması
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapılması
y_pred = reg_model.predict(X_test)

# Model performans metrikleri
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Skoru: {r2:.4f}")
print(f"MAE (Ortalama Mutlak Hata): {mae:,.0f}")
print(f"MSE (Hata Kareleri Ortalaması): {mse:,.0f}")

# =========================================================
# 5. MODELİN KAYDEDİLMESİ
# =========================================================
# Eğitilen model, Flask uygulamasında kullanılmak üzere
# pickle formatında diske kaydedilmektedir

with open('model.pkl', 'wb') as file:
    pickle.dump(reg_model, file)

print("\nModel başarıyla 'model.pkl' dosyası olarak kaydedildi.")
