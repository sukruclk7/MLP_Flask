from flask import Flask, render_template, request
import pickle
import numpy as np

# Uygulamayı başlat
app = Flask(__name__)

# Modeli yükle (Daha önce eğittiğimiz model.pkl)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Formdan gelen verileri al
        overall = int(request.form['overall'])
        age = int(request.form['age'])
        goals = int(request.form['goals'])
        assists = int(request.form['assists'])
        position_text = request.form['position']

        # Kategorik veriyi (Forvet, Defans) sayıya çevirmemiz lazım.
        # Çünkü modeli eğitirken LabelEncoder kullandık ve alfabetik sıraladı.
        # Defans=0, Forvet=1, Kaleci=2, Orta Saha=3
        position_map = {
            'Defans': 0,
            'Forvet': 1,
            'Kaleci': 2,
            'Orta Saha': 3
        }
        position_encoded = position_map[position_text]

        # Modelin istediği formatta listeye çevir
        # Sıralama eğitimdekiyle AYNI olmalı: [overall, age, goals, assists, position_encoded]
        features = np.array([[overall, age, goals, assists, position_encoded]])

        # Tahmin yap
        prediction = model.predict(features)

        # Sonucu güzelleştir (TL formatına çevir)
        output = round(prediction[0], 2)
        formatted_output = "{:,.0f} €".format(output).replace(",", ".")

        return render_template('index.html', prediction_text=f'Tahmini Piyasa Değeri: {formatted_output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Hata oluştu: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)