#!/usr/bin/env python
# coding: utf-8

# ### The aim of this script is to classify defaulting customers based on 17 observations. I made a local database connection, performed data processing, and created 5 classification models. The best model achieved 67.67% accuracy [using Naive Bayes].
# 
# 
# ### O objetivo é fazer uma classificação de clientes inadimplentes com base em 17 observações. Foi feita a conexão com o banco em que os dados estão alocados. Em seguida o tratamento de dados e a criação de modelos de classificação. O melhor modelo atingiu 67.67 % [Naive bayes].

# #### Bibliotecas utilizadas

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix
import psycopg2


# #### Caso fosse preciso usar LabelEnconder
# ###### Os dados qualitativos já estavam em formato numérico

# In[111]:


# Colunas que precisam de LabelEnconder
#colunasEnconder = [1,2,3,4,6,7,8,11,13]
#encoders = {}
# Gerando LabelEncoders
#for coluna in colunasEnconder:
#    encoders[coluna] = LabelEncoder()
#    previsores[:,coluna] = encoders[coluna].fit_transform(previsores[:,coluna])


# #### Conexão e leitura de dados do banco local PostgreSQL

# In[112]:


conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="ProjetoFinal",
    user="postgres",
    password="admin"
)


# In[113]:


cursor = conn.cursor()


# In[114]:


query = 'SELECT * FROM "CREDITO";'


# In[115]:


cursor.execute(query)


# In[116]:


results = cursor.fetchall()
dados = pd.DataFrame(results)


# In[117]:


query = """SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'CREDITO';"""


# In[118]:


cursor.execute(query)


# In[119]:


results = cursor.fetchall()
rotulos = np.array(results)


# In[120]:


cursor.close()
conn.close()


# In[121]:


dados.columns = rotulos[:,0]


# In[122]:


dados.head()


# #### Início do tratamento de dados - Remoção e NAs e valores incompatíveis, outliers, etc.

# In[123]:


dados['HistoricoCredito'].value_counts(dropna=False)


# In[124]:


dados['Proposito'].value_counts(dropna=False)


# In[125]:


dados['Investimentos'].value_counts(dropna=False)


# In[126]:


dados['Emprego'].value_counts(dropna=False)


# In[127]:


dados['TempoParcelamento'].value_counts(dropna=False)


# In[128]:


dados['EstadoCivil'].value_counts(dropna=False)


# In[129]:


dados['FiadorTerceiros'].value_counts(dropna=False)


# In[130]:


dados['ResidenciaDesde'].value_counts(dropna=False)


# In[131]:


dados['Idade'].value_counts(dropna=False)


# In[132]:


dados['OutrosFinanciamentos'].value_counts(dropna=False)


# In[133]:


dados['Habitacao'].value_counts(dropna=False)


# In[134]:


dados['EmprestimoExistente'].value_counts(dropna=False)


# In[135]:


dados['Profissao'].value_counts(dropna=False)


# In[136]:


dados['Dependentes'].value_counts(dropna=False)


# In[137]:


dados['SocioEmpresa'].value_counts(dropna=False)


# In[138]:


dados['Estrangeiro'].value_counts(dropna=False)


# In[139]:


dados['Status'].value_counts(dropna=False)


# #### Tratamento de dados em: Emprego, ResidenciaDesde, Habitacao, Profissao

# In[140]:


dados.loc[dados['Emprego'].isna()]


# In[141]:


# Substituicao NaNs de Emprego pela moda
moda = float(dados['Emprego'].mode())

dados['Emprego'].fillna(moda, inplace=True)


# In[142]:


dados['Emprego'].value_counts(dropna=False)


# In[143]:


# Substituicao NaNs de Residencia pela moda
moda = float(dados['ResidenciaDesde'].mode())

dados['ResidenciaDesde'].fillna(moda, inplace=True)

dados['ResidenciaDesde'].value_counts(dropna=False)


# In[144]:


# Substituicao NaNs de Habitacao pela moda
moda = float(dados['Habitacao'].mode())

dados['Habitacao'].fillna(moda, inplace=True)

dados['Habitacao'].value_counts(dropna=False)


# In[145]:


# Profissao 999 será setada para 4, pois é um item padrão de profissão
dados['Profissao'].replace(999, 4, inplace=True)
dados['Profissao'].value_counts(dropna=False)


# #### Pré processamento para construcao do modelo

# In[146]:


# Separacao de previsores e da classe
previsores = dados.iloc[:,0:17].values
classe = dados.iloc[:,17].values


# In[147]:


# Dividindo dados em treino e teste
xTreino, xTeste, yTreino, yTeste = train_test_split(previsores,
                                                    classe,
                                                    test_size=0.3,
                                                    random_state=0)


# #### Método: Naive Bayes

# In[148]:


# Classificacao com Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(xTreino, yTreino)


# In[149]:


previsoes = naive_bayes.predict(xTeste)


# In[150]:


confusao = confusion_matrix(yTeste, previsoes)
confusao


# In[151]:


taxa_acerto = accuracy_score(yTeste, previsoes)
print("Acurácia do modelo: {:.2f}%".format(taxa_acerto * 100))


# In[152]:


v = ConfusionMatrix(GaussianNB())
v.fit(xTreino, yTreino)
v.score(xTeste, yTeste)
v.poof()


# #### Método: Regressão Logística

# In[153]:


from sklearn.linear_model import LogisticRegression


# In[154]:


model = LogisticRegression()
model.fit(xTreino, yTreino)


# In[155]:


y_pred = model.predict(xTeste)


# In[156]:


taxa_acerto = accuracy_score(yTeste, y_pred)
print("Taxa de acerto: {:.2f}%".format(taxa_acerto * 100))


# In[157]:


confusao = confusion_matrix(yTeste, y_pred)
confusao


# In[158]:


v = ConfusionMatrix(LogisticRegression())
v.fit(xTreino, yTreino)
v.score(xTeste, yTeste)
v.poof()


# #### Método: SVM

# In[159]:


from sklearn.svm import SVC


# In[160]:


modelo = SVC()
modelo.fit(xTreino, yTreino)


# In[161]:


previsao = modelo.predict(xTeste)


# In[162]:


confusao = confusion_matrix(yTeste, previsao)
confusao


# In[163]:


v = ConfusionMatrix(SVC())
v.fit(xTreino, yTreino)
v.score(xTeste, yTeste)
v.poof()


# In[164]:


taxa_acerto = accuracy_score(yTeste, previsao)
print("Acurácia do modelo: {:.2f}%".format(taxa_acerto * 100))


# #### Método: Perceptron Multi Camada

# In[165]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[166]:


modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(xTreino.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelo.fit(xTreino, yTreino, epochs=100, batch_size=32)


# In[167]:


perda, taxa_acerto = modelo.evaluate(xTeste, yTeste)
print(f"Taxa de acerto: {100*taxa_acerto:.2f}" " %")


# In[168]:


predicao = modelo.predict(xTeste)
predicao = np.round(predicao).astype(int)


# In[169]:


conf_matrix = confusion_matrix(yTeste, predicao)
print("Matriz de Confusão:")
print(conf_matrix)


# #### Método: PCA [2 componentes principais] + Perceptron Multi Camada

# In[170]:


# Padronizacao de dados média zero e std 1
norm = StandardScaler()
xNorm = scaler.fit_transform(previsores)


# In[171]:


pca = PCA(n_components=10)
X_pca = pca.fit_transform(xNorm)


# In[172]:


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=classe, cmap='viridis', edgecolor='k')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Gráfico de Dispersão da PCA')
plt.colorbar(label='Classe')
plt.show()


# In[173]:


variance_explained = pca.explained_variance_ratio_
print("Percentual de variância explicada por cada componente principal:")
print(variance_explained)


# In[174]:


# Dividindo dados em treino e teste, novamente, agora informacoes da PCA
xTreino, xTeste, yTreino, yTeste = train_test_split(X_pca,
                                                    classe,
                                                    test_size=0.3,
                                                    random_state=0)


# In[175]:


# Rede neural perceptron multicamada
modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(xTreino.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# In[176]:


modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[177]:


modelo.fit(xTreino, yTreino, epochs=100, batch_size=32)


# In[180]:


predicao = modelo.predict(xTeste)
predicao = np.round(predicao).astype(int)


# In[181]:


conf_matrix = confusion_matrix(yTeste, predicao)
print("Matriz de Confusão:")
print(conf_matrix)


# In[182]:


perda, taxa_acerto = modelo.evaluate(xTeste, yTeste)
print(f"Taxa de acerto: {100*taxa_acerto:.2f}" " %")


# In[ ]:




