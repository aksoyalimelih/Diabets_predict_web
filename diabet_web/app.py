from flask import Flask, request, render_template, jsonify
import joblib  # Modelinizi yüklemek için
import numpy as np

app = Flask(__name__)
model = joblib.load('diabet_model.pkl')  # Modelinizi yükleyin

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Formdan verileri alın
    hamilelik_sayisi = float(request.form['Hamilelik_Sayisi'])
    kan_sekeri = float(request.form['Kan_Sekeri'])
    tansiyon = float(request.form['Tansiyon'])
    cilt_kalinligi = float(request.form['Cilt_Kalinligi'])
    insulin = float(request.form['Insulin'])
    vucut_kitle_indeksi = float(request.form['Vucut_Kitle_Indeksi'])
    diyabet_soyagaci_fonksiyonu = float(request.form['Diyabet_Soyagaci_Fonksiyonu'])
    yas = float(request.form['Yas'])
    
    # Veriyi modelle tahmin yapacak formata dönüştürün
    features = np.array([[hamilelik_sayisi, kan_sekeri, tansiyon, cilt_kalinligi, insulin, vucut_kitle_indeksi, diyabet_soyagaci_fonksiyonu, yas]])
    
    # Tahmin yapın
    prediction = model.predict(features)
    
    # Tahmini sonuç döndürün
    result = 'Diabet Pozitif' if prediction[0] == 1 else 'Diabet Negatif'
    
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)