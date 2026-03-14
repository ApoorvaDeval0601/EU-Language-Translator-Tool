# Language Identification & Translation Tool

A multimodal NLP application that detects the language of text, images, or voice input and translates it to English.

Built as a university NLP project using the [Europarl corpus](https://www.statmt.org/europarl/) across 15 European languages.

---

## How it works

1. **Input** — accepts text, an image (OCR), or audio (speech-to-text)
2. **Language identification** — a custom-trained Logistic Regression classifier predicts the language
3. **Translation** — the matching Helsinki-NLP `opus-mt` model is loaded from Hugging Face and translates to English

---

## Model

- **Architecture:** TF-IDF + Logistic Regression (scikit-learn pipeline)
- **Dataset:** Europarl v7 — ~18.1M sentences across 15 languages
- **Train/test split:** 80/20
- **Test accuracy: 97.07%**

![Confusion Matrix](assets/confusion_matrix_full_model.png)

**Supported languages:** Bulgarian, Czech, Danish, Dutch, English, Finnish, French, German, Hungarian, Latvian, Polish, Portuguese, Romanian, Slovenian, Spanish

---

## Stack

- **Frontend/UI:** Gradio
- **Language ID:** scikit-learn (TF-IDF + Logistic Regression)
- **Translation:** Hugging Face Transformers — Helsinki-NLP opus-mt models
- **OCR:** Tesseract + OpenCV
- **Speech-to-text:** Google Speech Recognition API

---

## Run locally

Requires Python 3.8+ and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed.

```bash
git clone https://github.com/YOUR_USERNAME/language-id-translator.git
cd language-id-translator
pip install -r requirements.txt
python app.py
```

App runs at `http://localhost:7860`

---

## Project structure

```
├── app.py                              # Gradio app
├── language_detection_model.joblib     # Trained classifier
├── requirements.txt
├── assets/
│   └── confusion_matrix_full_model.png
└── notebooks/
    ├── NLP_DRAFT_1.ipynb               # Data preprocessing & model training
    └── App.ipynb                       # App development
```
