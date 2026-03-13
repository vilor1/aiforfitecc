from flask import Flask, render_template, request, jsonify
import os
from engine import NanoTransformer

app = Flask(__name__)

# Configurações
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(BASE_DIR, 'treino.txt')

if not os.path.exists(TXT_PATH):
    with open(TXT_PATH, 'w') as f: f.write("Olá, eu sou uma inteligência artificial criada do zero.")

with open(TXT_PATH, 'r', encoding='utf-8') as f:
    text_data = f.read()

chars = sorted(list(set(text_data)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Cria e treina a IA ao iniciar
model = NanoTransformer(len(chars))
print("Treinando IA... aguarde.")
model.treinar(text_data, char_to_int, epochs=5000)
print("Treino concluído!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        prompt = data.get("msg", "")
        # Gerar resposta com 50 caracteres
        resp = model.gerar(prompt, 50, char_to_int, int_to_char)
        return jsonify({"answer": resp})
    except:
        return jsonify({"answer": "Erro ao processar."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
