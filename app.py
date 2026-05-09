import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- INICIALIZAÇÃO E TREINAMENTO (Ao ligar o servidor no Render) ---

# 1. Carregar os Dados
# O arquivo deve estar no seu repositório do GitHub junto com o app.py
df = pd.read_csv('limites_de_credito.csv')
X = df[['Idade', 'Salario', 'Score_Credito']]
y = df['Limite_Aprovado']

# 2. Normalização
normalizador_x = StandardScaler()
X_norm = normalizador_x.fit_transform(X)

normalizador_y = StandardScaler()
y_norm = normalizador_y.fit_transform(y.values.reshape(-1, 1))

# 3. Construção do Modelo (Mesma estrutura usada no Colab)
modelo_limite = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])
modelo_limite.compile(optimizer='adam', loss='mse')

# Treino rápido (verbose=0 para não encher o log do Render)
modelo_limite.fit(X_norm, y_norm, epochs=100, verbose=0) 

# --- ROTAS DO WEBHOOK ---

@app.route('/')
def index():
    return "Servidor da IA de Crédito Operacional!"

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    
    try:
        # Extraindo parâmetros do Dialogflow (nomes devem ser idênticos aos do console)
        parametros = req.get('queryResult').get('parameters')
        
        # AGORA CAPTURAMOS A IDADE TAMBÉM:
        idade = float(parametros.get('idade'))
        salario = float(parametros.get('salario'))
        score = float(parametros.get('score'))

        # Preparando os dados para a IA (Mesmo processo do Colab)
        cliente_novo = pd.DataFrame({'Idade': [idade], 'Salario': [salario], 'Score_Credito': [score]})
        cliente_norm = normalizador_x.transform(cliente_novo)

        # Predição (O resultado vem normalizado)
        previsao_norm = modelo_limite.predict(cliente_norm)

        # Desnormalização (Conversão para Reais)
        limite_real = normalizador_y.inverse_transform(previsao_norm)[0][0]

        # Tratando caso a IA retorne valor negativo (raro, mas possível em redes neurais)
        limite_final = max(0, limite_real)

        resposta = f"Análise finalizada! Com base na sua idade ({int(idade)} anos), salário e score, seu limite aprovado é de R$ {limite_final:,.2f}."

    except Exception as e:
        resposta = "Desculpe, não consegui processar sua análise. Verifique se informou apenas números."
        print(f"Erro no processamento: {e}")

    return jsonify({"fulfillmentText": resposta})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)