import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Memuat dataset
data = pd.read_csv('tipe_investor.csv')

# Preprocessing data
label_encoder = LabelEncoder()
data_cleaned = data.drop(columns=['Nama User'])
data_cleaned['tujuan_investasi'] = label_encoder.fit_transform(data_cleaned['tujuan_investasi'])
boolean_columns = ['menikah', 'pernah_investasi', 'investasi_jika_market_turun']
for col in boolean_columns:
    data_cleaned[col] = data_cleaned[col].astype(int)

X = data_cleaned.drop(columns=['tipe_investor'])
y = label_encoder.fit_transform(data_cleaned['tipe_investor'])

# Menggabungkan fitur dan target untuk oversampling
data_balanced = pd.concat([X, pd.DataFrame(y, columns=['tipe_investor'])], axis=1)
moderat = data_balanced[data_balanced['tipe_investor'] == 2]
konservatif = data_balanced[data_balanced['tipe_investor'] == 0]
agresif = data_balanced[data_balanced['tipe_investor'] == 1]

# Oversampling
konservatif_upsampled = resample(konservatif, replace=True, n_samples=len(moderat), random_state=42)
agresif_upsampled = resample(agresif, replace=True, n_samples=len(moderat), random_state=42)
data_upsampled = pd.concat([moderat, konservatif_upsampled, agresif_upsampled])

# Membagi data
X_upsampled = data_upsampled.drop(columns=['tipe_investor'])
y_upsampled = data_upsampled['tipe_investor']
X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Melatih model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Akurasi: {accuracy}")
print("Classification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Menyimpan model dan label encoder
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model dan encoder berhasil disimpan.")