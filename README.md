# 🖼️ Offline OCR Suite

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Engines](https://img.shields.io/badge/engines-Tesseract%20%7C%20EasyOCR%20%7C%20PaddleOCR-orange)

A powerful, 100% offline OCR system designed for Windows. It intelligently combines **Tesseract**, **EasyOCR**, and **PaddleOCR** to provide state-of-the-art text extraction without requiring an internet connection.

## ✨ Key Features

- 🧠 **Intelligent Auto-Selection** – Automatically picks the best engine for your specific image (e.g., PaddleOCR for Asian languages, EasyOCR for photos).
- ⚡ **Lazy Loading** – Engines only load when called, ensuring the GUI starts instantly and consumes minimal memory.
- 🖼️ **Advanced Preprocessing** – Built-in deskewing, denoising, and contrast enhancement for better accuracy.
- 💾 **Smart Caching** – Remembers previously processed images to avoid redundant computation.
- 📊 **Engine Comparison** – Compare results from all three engines side-by-side to find the most accurate output.

---

## 🚀 Quick Start (Windows)

No complex setup required. Run these exact commands while inside the `offline_ocr_suite` directory.

### 1. Launch the GUI
The primary way to use the suite. Drag and drop images to see instant results.
```powershell
venv\Scripts\python.exe ocr_gui.py
```

### 2. Use the CLI
For fast, command-line processing:
```powershell
# Basic OCR
venv\Scripts\python.exe ocr_cli.py document.png

# Compare all engines
venv\Scripts\python.exe ocr_cli.py scan.png --compare
```

---

## 🛠️ Project Structure

The project is organized to be lean and high-performance:

```text
offline_ocr_suite/
├── ocr_manager.py      # Core intelligence & Engine Orchestration
├── ocr_gui.py          # Professional Tkinter Interface
├── ocr_cli.py          # Powerful Command-Line Tool
├── config.yaml         # Project Configuration (Settings, Engines, Paths)
├── requirements.txt    # Python dependencies
├── examples.py         # Developer API examples
├── venv/               # Project Virtual Environment (DO NOT DELETE)
└── assets/             # Documentation visuals
```

---

## ⚙️ Configuration

You can customize the behavior in `config.yaml`:
- **Enable/Disable Engines**: Toggle Tesseract, EasyOCR, or PaddleOCR.
- **Preprocessing**: Turn on/off deskewing, denoising, and contrast enhancement.
- **Languages**: Configure which languages each engine should prioritize.

---

## 📝 License

This project is open-source and available under the **MIT License**. Use and modify freely for personal or commercial projects.
