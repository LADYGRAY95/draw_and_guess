from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("draw_model_advanced.keras")
with open("categories.txt") as f:
    categories = [line.strip() for line in f]

def preprocess_image(img_data):
    img_bytes = base64.b64decode(img_data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((28,28))
    arr = np.array(img).astype("float32")/255.0
    arr = 1.0 - arr  # invert colors (black brush on white)
    arr = arr.reshape(1,28,28,1)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    arr = preprocess_image(data["image"])
    pred = model.predict(arr)
    idx = np.argmax(pred)
    return jsonify({
        "prediction": categories[idx],
        "confidence": float(pred[0][idx])
    })

if __name__ == "__main__":
    app.run(debug=True)
