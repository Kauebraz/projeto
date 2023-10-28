from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import os


app = Flask(__name__)
CORS(app)

dic = {0: 'Guincho Asa Delta', 1: 'Guincho Lança Pesado', 2: 'Guincho Plataforma'}

model = load_model('BackEnd/model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100, 100))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 100, 100, 3)
    probabilities = model.predict(i)
    predicted_class = probabilities.argmax()
    return dic[predicted_class]

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio"})

    if file:
        file_path = os.path.join('BackEnd/static', file.filename)
        file.save(file_path)
        p = predict_label(file_path)
        return jsonify({"prediction": p})

if __name__ == '__main__':
    app.run(debug=True)
    
