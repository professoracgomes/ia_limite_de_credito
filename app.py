import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar o modelo e os normalizadores UMA VEZ ao iniciar
# Isso evita o erro de TIMEOUT no Dialogflow
MODELO = tf.keras.models.load_model('modelo_ia.h5')
SCALER_X = joblib.load('scaler_x.pkl')
SCALER_Y = joblib.load('scaler_y.pkl')

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    try:
        parametros = req.get('queryResult').get('parameters')
        
        idade = float(parametros.get('idade'))
        salario = float(parametros.get('salario'))
        score = float(parametros.get('score'))

        # 1. Preparar dados
        entrada = pd.DataFrame({'Idade': [idade], 'Salario': [salario], 'Score_Credito': [score]})
        
        # 2. Normalizar
        entrada_norm = SCALER_X.transform(entrada)

        # 3. Predizer (Usa o modelo já carregado na memória)
        pred_norm = MODELO.predict(entrada_norm)

        # 4. Desnormalizar
        limite_real = SCALER_Y.inverse_transform(pred_norm)[0][0]
        limite_final = max(0, limite_real)

        resposta = f"IA Analisou: Com base no seu perfil, seu limite sugerido é R$ {limite_final:,.2f}."

    except Exception as e:
        resposta = "Erro ao processar predição da IA."
        print(f"Erro: {e}")

    return jsonify({"fulfillmentText": resposta})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
