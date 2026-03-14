# 🌐 Language Identification & Translation Tool

A multimodal NLP application built for a university project that detects the language of any input — text, image, or voice — and translates it to English.

---

## ✨ Features

- **Text input** — paste or type in any supported language
- **Image input** — upload an image containing text (OCR powered by Tesseract)
- **Voice input** — record or upload audio (transcribed via Google Speech Recognition)
- **Language identification** — custom-trained Logistic Regression model (`.joblib`)
- **Translation to English** — Helsinki-NLP models loaded on-demand from Hugging Face Hub

---

## 🌍 Supported Languages

Bulgarian, Czech, Danish, Dutch, English, Finnish, French, German, Hungarian, Latvian, Polish, Portuguese, Romanian, Slovenian, Spanish

---

## 🧠 Model Details

The language identification model is a **scikit-learn pipeline** combining:
- `TfidfVectorizer` for feature extraction
- `LogisticRegression` as the classifier

Trained on a combined multilingual dataset (~3.5M samples across 15 languages).  
See the confusion matrix below for performance:

![Confusion Matrix](assets/confusion_matrix_full_model.png)

Translation is handled by **Helsinki-NLP's opus-mt** models, fetched from Hugging Face Hub at runtime and cached per session.

---

## 🚀 Run Locally

**Prerequisites:** Python 3.8+ and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/language-id-translator.git
cd language-id-translator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

The Gradio UI will open at `http://localhost:7860`

---

## 🤗 Hosted on Hugging Face Spaces

> Live demo: [Add your Spaces link here]

---

## 📁 Project Structure

```
├── app.py                              # Main Gradio application
├── language_detection_model.joblib     # Trained language ID model
├── requirements.txt
├── assets/
│   └── confusion_matrix_full_model.png
└── notebooks/
    ├── NLP_DRAFT_1.ipynb               # Model training notebook
    └── App.ipynb                       # App prototyping notebook
```

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `NLP_DRAFT_1.ipynb` | Data preprocessing, model training, evaluation |
| `App.ipynb` | Gradio app development and testing |

---

## 🛠 Tech Stack

`Python` · `Gradio` · `scikit-learn` · `Hugging Face Transformers` · `OpenCV` · `Pytesseract` · `SpeechRecognition`
