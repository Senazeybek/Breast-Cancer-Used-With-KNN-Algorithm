import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

# Veriyi yükleme
df = pd.read_csv('breast_cancer.csv')

# Sütun isimlerini belirleme
df.columns = ["id", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
              "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
              "normal_nucleoli", "mitoses", "class"]

# ID sütununu kaldırma
df = df.drop(columns=["id"])

# 'bare_nuclei' sütunundaki '?' karakterlerini NaN ile değiştirme ve sonra bu NaN değerleri sayısal veri ile doldurma
df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'], errors='coerce')
df = df.dropna()

# Hedef değişkeni ve özellikleri belirleme
X = df.drop(columns=["class"])
y = df["class"].map({2: 0, 4: 1})  # 2: benign (iyi huylu), 4: malignant (kötü huylu)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred = knn.predict(X_test)

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Karışıklık Matrisi
conf_matrix = confusion_matrix(y_test, y_pred)

# 1. Sınıf Dağılım Grafiği
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Sınıf Dağılım Grafiği')
plt.xlabel('Sınıf')
plt.ylabel('Sayı')
plt.show()

# 2. Özellik Dağılım Grafikleri
df_features = df.drop(columns=['class'])
df_features.hist(figsize=(12, 10), bins=20)
plt.suptitle('Özellik Dağılım Grafikleri')
plt.show()

# 3. Doğruluk Skoruna Göre k Değerinin Seçilmesi
neighbors = np.arange(1, 21)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

plt.figure(figsize=(10, 6))
plt.plot(neighbors, train_accuracy, label='Eğitim Seti Doğruluğu', marker='o')
plt.plot(neighbors, test_accuracy, label='Test Seti Doğruluğu', marker='o')
plt.xlabel('Komşu Sayısı')
plt.ylabel('Doğruluk')
plt.title('K-En Yakın Komşu Modelinin Doğruluk Grafiği')
plt.legend()
plt.grid(True)
plt.show()

# 4. Karışıklık Matrisini Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.title('Karışıklık Matrisi')
plt.show()

# 5. ROC Eğrisi ve AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

