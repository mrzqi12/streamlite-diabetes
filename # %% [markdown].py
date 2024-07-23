# %% [markdown]
# ### 1.Tentukan Libary Yang digunakan 

# %%
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# %% [markdown]
# ### 2.Load Dataset

# %%
diabetes_dataset = pd.read_csv('diabetes.csv')

# %%
diabetes_dataset.head()

# %%
diabetes_dataset.shape

# %%
diabetes_dataset['Outcome'].value_counts()

# %%
# Memisahkan data dan label --> untuk memisahkan label dan diagnosa

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# %%
print(X)

# %%
print(Y)

# %% [markdown]
# ### 3.Standarisasi Data

# %%
### Fungsi standarisasi data agar data bisa diseiimbangkan dengan scaller
scaler = StandardScaler()

# %%
#Kita scaller data yang X
scaler.fit(X)

# %%
#Proses transformasi
standarized_data = scaler.transform(X)

# %%
#data di print untuk cek data sudah di scaler
print(standarized_data)

# %%
#Definisikan data
X = standarized_data
Y = diabetes_dataset['Outcome']

# %%
#Print Data set, X yang sudah di scaler dan Y label
print(X)
print(Y)

# %% [markdown]
# ### 4. Memisahkan Data Training dan Data Testing

# %%
#identifikasi 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# %%
print(X.shape, X_train.shape, X_test.shape)

# %% [markdown]
# ### 5.Membuat data latih menggunakan algoritma SVM

# %%
classifier = svm.SVC(kernel= 'linear')

# %%
classifier.fit(X_train, Y_train)

# %% [markdown]
# ### 6.Membuat Model Evaluasi untuk mengukur tingkat akurasi

# %%
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
print('Akurasi Data Training adalah = ', training_data_accuracy)

# %%
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
print('Akurasi Data Testing adalah =',test_data_accuracy)

# %% [markdown]
# ### 7.Membuat Model Prediksi

# %%
input_data = (6,148,72,35,0,33.6,0.627,50)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print('Pasien tidak kena diabetes')
else :
    print('Pasien Kena diabetes')

# %% [markdown]
# ### 8.Simpan Model

# %%
import pickle

# %%
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename,'wb'))


