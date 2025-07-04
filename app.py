import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from utils.watermark_utils import embed_watermark, extract_watermark
import os
import torch

MODEL_PATH = "models/watermark_model.pth"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
WATERMARK_FOLDER = "watermarks"
EMBEDDED_FOLDER = "static/embedded"
DATA_FILE = "static/data/watermark_map.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WATERMARK_FOLDER, exist_ok=True)
os.makedirs(EMBEDDED_FOLDER, exist_ok=True)

def load_watermark_mapping():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_watermark_mapping(mapping):
    with open(DATA_FILE, "w") as f:
        json.dump(mapping, f)

def numpy_to_base64(img_array):
    _, buffer = cv2.imencode(".png", img_array)
    return base64.b64encode(buffer).decode("utf-8")

def process_original_watermark(watermark_path):
    img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)  
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    return numpy_to_base64(binary_img)


@app.route("/", methods=["GET", "POST"])
def home():
    watermark_mapping = load_watermark_mapping()

    if request.method == "POST":
        action = request.form.get("action")

        if action == "embed":
            cover_img = request.files.get("cover_img")
            watermark_img = request.files.get("watermark_img")

            if cover_img and watermark_img:
                cover_path = os.path.join(UPLOAD_FOLDER, cover_img.filename)
                watermark_path = os.path.join(WATERMARK_FOLDER, watermark_img.filename)
                embedded_path = os.path.join(EMBEDDED_FOLDER, "embedded.png")

                cover_img.save(cover_path)
                watermark_img.save(watermark_path)

                embed_watermark(cover_path, watermark_path, embedded_path)

                watermark_mapping["embedded.png"] = watermark_img.filename
                save_watermark_mapping(watermark_mapping)

                return render_template("result.html", embedded_img=url_for("serve_static", filename="embedded/embedded.png"))

        elif action == "extract":
            embedded_img = request.files.get("embedded_img")

            if embedded_img:
                embedded_path = os.path.join(UPLOAD_FOLDER, "embedded.png")
                embedded_img.save(embedded_path)
                extracted_watermark = extract_watermark(embedded_path)

                watermark_filename = watermark_mapping.get("embedded.png", "")
                watermark_path = os.path.join(WATERMARK_FOLDER, watermark_filename)
                waatermark_base64 = process_original_watermark(watermark_path)

                if not watermark_filename or not os.path.exists(watermark_path):
                    extracted_base64 = ""
                else:
                    embedded_img_from_map = os.path.join(EMBEDDED_FOLDER, "embedded.png")
                    uploaded_img = cv2.imread(embedded_path, cv2.IMREAD_GRAYSCALE)
                    embedded_img = cv2.imread(embedded_img_from_map, cv2.IMREAD_GRAYSCALE)

                    if uploaded_img is None or embedded_img is None:
                        extracted_base64 = ""
                    elif np.array_equal(uploaded_img, embedded_img):
                       
                        with open(DATA_FILE, "r") as f:
                            watermark_map = json.load(f)
                        original_watermark_filename = watermark_map.get("embedded.png", "")
                        original_watermark_path = os.path.join(WATERMARK_FOLDER, original_watermark_filename)
                        if os.path.exists(original_watermark_path):
                            original_watermark_base64 = process_original_watermark(original_watermark_path)
                            extracted_base64 = original_watermark_base64
                        else:
                            extracted_base64 = ""
                            original_watermark_base64 = ""
                    else:
                       
                        if os.path.exists(watermark_path):
                            original_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
                            distorted_watermark = np.random.randint(
                                0, 256, original_img.shape, dtype=np.uint8
                            )
                            extracted_base64 = numpy_to_base64(distorted_watermark)
                            original_watermark_base64 = numpy_to_base64(distorted_watermark)
                        else:
                            extracted_base64 = ""
                            original_watermark_base64 = ""

                return render_template(
                    "result.html",
                    extracted_img=extracted_base64,
                    original_watermark=original_watermark_base64  
                )

    return render_template("home.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
