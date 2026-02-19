# Детекция дефектов строительных конструкций

ИИ-приложение для обнаружения дефектов (трещин) бетонных поверхностей
с использованием свёрточной нейронной сети (CNN).

## Технологии
- Python 3.11, PyTorch (Transfer Learning, ResNet18)
- Flask (веб-приложение)
- Docker (контейнеризация)
- Git/GitHub (контроль версий)

## Запуск

### Локально
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

### Docker
```bash
docker build -t defect-detection .
docker run -p 5000:5000 defect-detection
```

Веб-интерфейс: http://localhost:5000

## Результаты
Точность модели на тестовой выборке: ~97%