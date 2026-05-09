import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor de Crédito Ativo", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    try:
        parametros = req.get('queryResult').get('parameters')
        idade = float(parametros.get('idade'))
        salario = float(parametros.get('salario'))
        score = float(parametros.get('score'))

        # Lógica matemática que simula a IA (sem carregar o TensorFlow)
        limite = (salario * 0.4) + (score * 5) + (idade * 10)
        
        resposta = f"Análise concluída! Com base no seu perfil, seu limite aprovado é R$ {limite:,.2f}."
    except:
        resposta = "Erro ao processar os dados. Tente novamente."

    return jsonify({"fulfillmentText": resposta})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
