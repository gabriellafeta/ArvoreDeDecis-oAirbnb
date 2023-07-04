# Importando pacotes

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
from graphviz import Source
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import os
import openpyxl

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Carregando a base do estudo realizado anteriormente

base_final_df = pd.read_excel("base_final_ordenada.xlsx")

# Perceba que foi adicionada a coluna "Boa escolha" como uma variável binária
# baseada nos critérios de preço, distância média e distância mínima
# Agora será definida a árvore de decisão inicial de predição para a coluna "Boa escolha"

X = base_final_df[["dist_media", "dist_min"]]
y = base_final_df["Boa escolha"]

clasf_boa_escolha = DecisionTreeClassifier(max_depth=2, random_state=42)

# Utilizamos um valor de max_depth=2 para simplificar o modelo dado o número de variáveis
# Implementando a predição cruzada com o SciKit-Learn
# Primeiro avaliaremos o efeito do número de folds na precisão e revocação

lista_precisao = []
lista_revocacao = []

for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    y_prev = cross_val_predict(clasf_boa_escolha, X, y, cv=k)
    precisao = precision_score(y, y_prev)
    revocacao = recall_score(y, y_prev)
    lista_precisao.append(precisao)
    lista_revocacao.append(revocacao)


# Avaliando graficamente o comportamento da precisão e revocação para os diferentes valores de K

# plt.plot(lista_precisao, label='Precisão')
# plt.plot(lista_revocacao, label='Revocação')
# plt.legend()
# plt.show()

# Considerando o tamanho da base de dados e o contexto será utilizado k=2, dado que prioriza um alto valor de precisão
# Nesse contexto, apesar de ter um bom valor de revocação, prioriza-se precisão, pois o objetivo é direcionar a escolha
# de um único airbnb ideal para uma viagem, portanto é melhor garantir apenas os seguros

# o valor de k=2 indica que o conjunto de treino e de testes tem que conter metdade das amostras cada

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Treinar o algoritmo com o conjunto de treinamento
clasf_boa_escolha.fit(X_train, y_train)
y_prev_final = clasf_boa_escolha.predict(X_test)

precisao_final = precision_score(y_test, y_prev_final)

print(precisao_final)

# Visualizando a arvore de decisão

# export_graphviz(
#         clasf_boa_escolha,
#         out_file=os.path.join(IMAGES_PATH, "clasf_boa_escolha.dot"),
#         feature_names=X.columns,
#         class_names=["Escolha de menor valor", "Boa escolha"],
#         rounded=True,
#         filled=True
#     )
#
# Source.from_file(os.path.join(IMAGES_PATH, "clasf_boa_escolha.dot"))

# Projetando a Curva ROC e o valor da área sobre ela como parâmetro de desempenho


y_prev_k2 = cross_val_predict(clasf_boa_escolha, X, y, cv=2)
taxa_falso_positivo, taxa_verd_positivo, limiar = roc_curve(y_train, y_prev_final)

# Calcular a taxa de falso positivo, taxa de verdadeiro positivo e limiar para a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prev_final)

# Plotar a curva ROC
plt.plot(fpr, tpr)
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC')
plt.show()

area = roc_auc_score(y_test, y_prev_final)
print(area)
