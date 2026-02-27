# 🖼️ Offline OCR Suite

A simple and powerful **100% offline OCR system for Windows**.

This tool extracts text from images using:
- Tesseract
- EasyOCR
- PaddleOCR

No internet connection required ✅

---

## ✨ Features

- 🔍 Extract text from images
- 🤖 Automatic engine selection
- 🖼️ Image preprocessing (better accuracy)
- ⚡ Fast and lightweight
- 💻 GUI and CLI support
- 📊 Compare results from all engines

---

## 🚀 Quick Start (Windows)

Make sure you are inside the `offline_ocr_suite` folder.

### ▶ Run GUI

```powershell
venv\Scripts\python.exe ocr_gui.py
```

Drag and drop images to extract text.

---

### ▶ Run CLI

Basic OCR:

```powershell
venv\Scripts\python.exe ocr_cli.py image.png
```

Compare all engines:

```powershell
venv\Scripts\python.exe ocr_cli.py image.png --compare
```

---

## 📁 Project Structure

```
offline_ocr_suite/
│── ocr_manager.py
│── ocr_gui.py
│── ocr_cli.py
│── config.yaml
│── requirements.txt
│── examples.py
│── venv/
│── assets/
```

---

## ⚙️ Configuration

Edit `config.yaml` to:
- Enable or disable engines
- Change languages
- Turn preprocessing on/off

---

## 🛠 Requirements

- Python 3.8+
- Windows OS
- Tesseract installed

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License
