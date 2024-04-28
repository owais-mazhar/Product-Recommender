import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
from flask import Flask, render_template, request, redirect, url_for
import io
import os
from werkzeug.utils import secure_filename
import base64
import tensorflow

app = Flask(__name__)

# Define a folder to store uploaded files
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        return redirect(url_for("predict", filename=filename))


@app.route("/predict/<filename>")
def predict(filename):
    feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))

    model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

    img = image.load_img(
        os.path.join(app.config["UPLOAD_FOLDER"], filename), target_size=(224, 224)
    )
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    neighbors = NearestNeighbors(n_neighbors=7, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized_result])

    images_data = []
    for file in indices[0][:6]:  # Change this line to iterate from index 0 to 6
        temp_img = cv2.imread(filenames[file])
        _, img_encoded = cv2.imencode(".png", cv2.resize(temp_img, (512, 512)))
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        images_data.append(img_base64)

    return render_template("result.html", images_data=images_data)


if __name__ == "__main__":
    app.run(debug=True)
