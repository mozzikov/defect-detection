"""
app.py — Flask веб-приложение для детекции дефектов.
"""
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = Flask(__name__, template_folder="../templates")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/defect_model.pth")
CLASSES = ["Нет дефекта (Negative)", "Дефект обнаружен (Positive)"]

print(f"Загрузка модели из {MODEL_PATH}...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Модель загружена!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return jsonify({
        "class": CLASSES[predicted.item()],
        "confidence": f"{confidence.item() * 100:.1f}%",
        "details": {
            "negative_prob": f"{probs[0][0].item() * 100:.1f}%",
            "positive_prob": f"{probs[0][1].item() * 100:.1f}%"
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)