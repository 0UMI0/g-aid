import os
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model import resnet50
from utils import preprocess_image, predict_image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "npr_model_augs.pth")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model, then load saved state_dict
model = resnet50(num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    image_url = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part in request."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            image_tensor = preprocess_image(save_path, device)
            result, probability = predict_image(model, image_tensor)
            image_url = f"/static/uploads/{filename}"
        else:
            error = "Unsupported file type. Use png, jpg, jpeg, or webp."

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        image_url=image_url,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)