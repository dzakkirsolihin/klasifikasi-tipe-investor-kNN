import joblib

def validate_input(prompt, valid_values=None, input_type=int):
    while True:
        try:
            value = input_type(input(prompt))
            if valid_values and value not in valid_values:
                raise ValueError("Input not in valid options.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def predict_user_input():
    # Load model, encoder, and scaler
    knn = joblib.load('knn_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    
    print("\nEnter details to predict investor type:")
    penghasilan_bulanan = validate_input("Penghasilan bulanan (dalam rupiah): ", input_type=int)
    menikah = validate_input("Apakah menikah? (1 = Ya, 0 = Tidak): ", valid_values=[0, 1])
    tanggungan = validate_input("Jumlah tanggungan: ", input_type=int)
    tujuan_investasi = input("Tujuan investasi (tabungan, pensiun, pendidikan): ").lower()
    if tujuan_investasi == 'tabungan':
        tujuan_investasi_encoded = 2
    elif tujuan_investasi == 'pensiun':
        tujuan_investasi_encoded = 1
    elif tujuan_investasi == 'pendidikan':
        tujuan_investasi_encoded = 0
    else:
        print("Tujuan investasi tidak valid. Gunakan: tabungan, pensiun, pendidikan.")
        return
    
    jangka_investasi = validate_input("Jangka waktu investasi (dalam tahun): ", input_type=int)
    usia = validate_input("Usia: ", input_type=int)
    pernah_investasi = validate_input("Apakah pernah investasi sebelumnya? (1 = Ya, 0 = Tidak): ", valid_values=[0, 1])
    toleransi_kehilangan = validate_input("Toleransi kehilangan modal (dalam persen): ", input_type=int)
    investasi_jika_market_turun = validate_input("Tetap investasi jika pasar turun? (1 = Ya, 0 = Tidak): ", valid_values=[0, 1])
    
    # Prepare and scale input data
    user_data = [[
        penghasilan_bulanan,
        menikah,
        tanggungan,
        tujuan_investasi_encoded,
        jangka_investasi,
        usia,
        pernah_investasi,
        toleransi_kehilangan,
        investasi_jika_market_turun
    ]]
    user_data_scaled = scaler.transform(user_data)
    
    # Predict investor type
    prediction = knn.predict(user_data_scaled)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    print(f"\nPredicted Investor Type: {predicted_class}")
    
    # Additional explanation based on investor type
    explanations = {
        "konservatif": "Investor konservatif mengutamakan keamanan modal dengan menghindari risiko tinggi. "
                       "Investasi seperti deposito atau obligasi sering menjadi pilihan utama.",
        "moderat": "Investor moderat menggabungkan risiko dan imbal hasil. Mereka bersedia mengambil risiko sedang "
                   "untuk mendapatkan keuntungan lebih besar dibandingkan tipe konservatif.",
        "agresif": "Investor agresif berani mengambil risiko tinggi untuk mendapatkan potensi keuntungan besar. "
                   "Investasi seperti saham individu atau instrumen berisiko tinggi adalah fokus utama."
    }
    
    # Display the explanation
    print("\nPenjelasan Tipe Investor:")
    print(explanations.get(predicted_class, "Tidak ada penjelasan untuk tipe ini."))

if __name__ == "__main__":
    predict_user_input()
