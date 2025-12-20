# BLG-407 Makine Öğrenmesi - Proje 3
## ⚽ Yapay Zeka Destekli Futbolcu Değeri Tahmin Sistemi

Bu proje, **BLG-407 Makine Öğrenmesi** dersi kapsamında geliştirilmiştir. Projenin temel amacı, futbolcuların çeşitli istatistiksel verilerini (güç, yaş, gol, asist vb.) kullanarak piyasa değerlerini **Çoklu Doğrusal Regresyon (Multiple Linear Regression)** yöntemi ile tahmin etmektir. Eğitilen model, kullanıcı dostu bir **Flask** web arayüzü ile sunulmuştur.

---

### 👤 Öğrenci Bilgileri
* **Adı Soyadı:** Şükrü Çelik
* **Öğrenci Numarası:** 2212721016
* **Ders:** BLG-407 Makine Öğrenmesi

---

### 📂 Proje İçeriği ve Dosyalar
* **`app.py`**: Flask web sunucusunu başlatan ve `model.pkl` dosyasını kullanarak tahmin yapan ana uygulama dosyası.
* **`model.pkl`**: Eğitilmiş ve serileştirilmiş (pickle) Makine Öğrenmesi modeli.
* **`Futbolcu_Model_Egitimi.ipynb`**: Veri setinin oluşturulması, ön işleme, Backward Elimination ve model eğitiminin yapıldığı Jupyter Notebook dosyası.
* **`futbolcu_verisi.csv`**: Projede kullanılan (sentetik olarak üretilmiş) veri seti.
* **`templates/index.html`**: Kullanıcının veri girişi yaptığı web arayüzü tasarımı.

---

### ⚙️ 1. Veri Ön İşleme (Data Preprocessing)
Modelin başarısını artırmak ve hocanın istediği kriterleri sağlamak adına aşağıdaki işlemler uygulanmıştır:

1.  **Veri Seti:** Proje gereksinimlerine uygun, kontrol edilebilir sentetik bir veri seti oluşturulmuştur.
2.  **Öznitelik Seçimi (Feature Selection):** Modelde piyasa değerini etkileyen en kritik 5 özellik kullanılmıştır:
    * `Overall Rating` (Genel Güç)
    * `Age` (Yaş)
    * `Goals` (Gol Sayısı)
    * `Assists` (Asist Sayısı)
    * `Position` (Mevki)
3.  **Kategorik Dönüşüm (Encoding):** `Position` sütunu (Forvet, Defans vb.) sayısal olmadığı için **Label Encoding** yöntemiyle sayısal değerlere (0, 1, 2, 3) dönüştürülmüştür.
4.  **Kayıp Veri Analizi:** Veri setindeki olası boş değerler, ilgili sütunun ortalaması (`mean`) ile doldurulmuştur.

---

### 📉 2. Geriye Doğru Eleme (Backward Elimination)
İstatistiksel olarak anlamsız değişkenlerin modelden atılması işlemi kod içinde otomatikleştirilmiştir:

* **Test Değişkeni:** Veri setine bilinçli olarak `Jersey Number` (Forma Numarası) adında, fiyata etkisi olmayan rastgele bir değişken eklenmiştir.
* **OLS Analizi:** Statsmodels kütüphanesi ile OLS (Ordinary Least Squares) raporu çıkarılmış ve P-value değerleri incelenmiştir.
* **Sonuç:** `Jersey Number` değişkeninin **P-value değeri 0.05'ten büyük** çıktığı için (istatistiksel olarak anlamsız), algoritma tarafından **otomatik olarak tespit edilmiş ve veri setinden çıkarılmıştır.**

---

### 📊 3. Model Başarısı ve Metrikler
Model eğitimi sonucunda test verisi üzerinde elde edilen başarı metrikleri şöyledir:

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **R² (R-Squared)** | **0.97+** | Model, verideki değişimin %97'sini açıklayabilmektedir. (Çok Yüksek Başarı) |
| **MAE** | Düşük | Ortalama Mutlak Hata, kabul edilebilir seviyededir. |
| **MSE** | Düşük | Hata Kareler Ortalaması optimize edilmiştir. |

---

### 🚀 4. Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için adımları takip edin:

**Adım 1: Gerekli Kütüphaneleri Yükleyin**
Terminale şu kodu yapıştırın:
```bash
pip install flask pandas numpy scikit-learn statsmodels
