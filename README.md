# CXR-AI Assistant

A web application that uses deep learning to analyze chest X-ray images and detect potential diseases. Built as a clinical decision support tool for educational and research purposes.

## Features

- **Multi-label disease detection** -- Predicts probabilities for 20 thoracic conditions using a DenseNet121 model trained on the NIH Chest X-ray dataset
- **Grad-CAM explainability** -- Generates heatmap overlays showing which regions of the X-ray influenced the model's predictions
- **Patient correlation** -- Optionally accepts patient demographics (age, gender, smoking history, existing conditions) and correlates them with imaging findings
- **Clinical summary** -- Produces a structured report with key findings, recommended next steps, and a watchlist for borderline results

## Tech Stack

| Layer              | Technology                                  |
| ------------------ | ------------------------------------------- |
| Frontend           | React, Vite, Tailwind CSS, Framer Motion    |
| Backend            | Flask (Python)                              |
| Model              | DenseNet121 (PyTorch)                       |
| Explainability     | Grad-CAM                                    |

## Project Structure

```
app/
├── backend/            # Flask API server
│   ├── app.py          # API routes (/api/predict, /api/gradcam, /api/health)
│   └── model.py        # Model loading and inference logic
├── frontend/           # React + Vite frontend
│   └── src/
├── models/             # Trained model checkpoints (.pt files)
├── nih_pipeline/       # Training and deployment runtime utilities
├── scripts/            # Training and data preprocessing scripts
├── tests/              # Smoke tests
├── sample_images/      # Sample X-rays for testing
├── NIH_CNN.ipynb       # CNN training notebook (Google Colab)
├── NIH_DenseNet.ipynb  # DenseNet training notebook (Google Colab)
└── NIH_ResNet.ipynb    # ResNet training notebook (Google Colab)
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API server starts on `http://localhost:5001`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server starts on `http://localhost:5173` and proxies API requests to the backend.

## API Endpoints

| Method | Endpoint         | Description                          |
| ------ | ---------------- | ------------------------------------ |
| POST   | `/api/predict`   | Upload an X-ray and get predictions  |
| POST   | `/api/gradcam`   | Generate Grad-CAM heatmap            |
| GET    | `/api/health`    | Check server and model status        |

## Disclaimer

This tool is intended for **educational and research purposes only**. It is not a certified medical device and should not be used for clinical diagnosis.
