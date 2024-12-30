import joblib

# Memuat model dan label encoder
knn = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def predict_user_input():
    print("\nMasukkan detail untuk memprediksi tipe investor:")
    penghasilan_bulanan = int(input("Penghasilan bulanan (dalam rupiah): "))
    menikah = int(input("Apakah menikah? (1 = Ya, 0 = Tidak): "))
    tanggungan = int(input("Jumlah tanggungan: "))
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
    
    jangka_investasi = int(input("Jangka waktu investasi (dalam tahun): "))
    usia = int(input("Usia: "))
    pernah_investasi = int(input("Apakah pernah investasi sebelumnya? (1 = Ya, 0 = Tidak): "))
    toleransi_kehilangan = int(input("Toleransi kehilangan modal (dalam persen): "))
    investasi_jika_market_turun = int(input("Tetap investasi jika pasar turun? (1 = Ya, 0 = Tidak): "))
    
    # Membuat array input
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
    
    # Melakukan prediksi
    prediction = knn.predict(user_data)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    print(f"\nPrediksi tipe investor Anda: {predicted_class}")
    
    # Penjelasan tambahan berdasarkan tipe investor
    if predicted_class == 'konservatif':
        print("Tipe investor konservatif adalah investor yang mengutamakan keamanan modal. "
              "Biasanya lebih memilih investasi yang stabil dengan risiko rendah.")
    elif predicted_class == 'moderat':
        print("Tipe investor moderat adalah investor yang bersedia mengambil risiko sedang "
              "untuk mendapatkan keuntungan yang lebih tinggi dibandingkan investasi konservatif.")
    elif predicted_class == 'agresif':
        print("Tipe investor agresif adalah investor yang cenderung mengambil risiko tinggi "
              "dengan harapan mendapatkan keuntungan besar dalam waktu relatif singkat.")

# Jalankan fungsi prediksi
if __name__ == "__main__":
    predict_user_input()
