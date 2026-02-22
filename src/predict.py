"""
predict.py — Предсказание для одного изображения.
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

MODEL_PATH = "models/defect_model.pth"
CLASSES = ["Нет дефекта (Negative)", "Дефект обнаружен (Positive)"]


def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return CLASSES[predicted.item()], confidence.item() * 100


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python src/predict.py <путь_к_изображению>")
        sys.exit(1)
    model = load_model(MODEL_PATH)
    label, conf = predict_image(model, sys.argv[1])
    print(f"Результат: {label}")
    print(f"Уверенность: {conf:.1f}%")