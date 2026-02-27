import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import json
from ocr_manager import UnifiedOCR, OCREngine


class OCRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline OCR Suite - Multi-Engine")
        self.root.geometry("1200x800")
        
        self.ocr = UnifiedOCR()
        self.current_image = None
        
        self._create_ui()
        
    def _create_ui(self):
        # Top frame - Controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="📁 Open Image", 
                  command=self._load_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Engine:").pack(side=tk.LEFT, padx=(20,5))
        self.engine_var = tk.StringVar(value="auto")
        engine_combo = ttk.Combobox(control_frame, textvariable=self.engine_var,
                                   values=["auto", "tesseract", "easyocr", "paddleocr"],
                                   width=12, state="readonly")
        engine_combo.pack(side=tk.LEFT, padx=5)
        
        self.preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Preprocess", 
                       variable=self.preprocess_var).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(control_frame, text="🔍 Run OCR", 
                  command=self._run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 Compare All", 
                  command=self._compare_engines).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="💾 Save Results", 
                  command=self._save_results).pack(side=tk.RIGHT, padx=5)
        
        # Main content - Paned window
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left - Image
        left_frame = ttk.LabelFrame(paned, text="Image", padding=5)
        paned.add(left_frame, weight=1)
        
        self.image_label = ttk.Label(left_frame, text="No image loaded")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right - Results
        right_frame = ttk.LabelFrame(paned, text="OCR Results", padding=5)
        paned.add(right_frame, weight=1)
        
        # Results treeview
        columns = ('engine', 'text', 'confidence', 'time')
        self.tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=15)
        
        self.tree.heading('engine', text='Engine')
        self.tree.heading('text', text='Text')
        self.tree.heading('confidence', text='Confidence')
        self.tree.heading('time', text='Time (s)')
        
        self.tree.column('engine', width=100)
        self.tree.column('text', width=400)
        self.tree.column('confidence', width=100)
        self.tree.column('time', width=80)
        
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Text detail view
        self.detail_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=10)
        self.detail_text.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(10,0))
        
        right_frame.grid_rowconfigure(0, weight=2)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.results_data = []
        
    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tiff *.bmp *.gif *.pdf"),
                      ("All files", "*.*")]
        )
        if path:
            self._display_image(path)
            self.status_var.set(f"Loaded: {path}")
            
    def _display_image(self, path):
        try:
            img = Image.open(path)
            # Resize for display while maintaining aspect ratio
            display_size = (500, 600)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            self.current_image = path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def _run_ocr(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Processing...")
        self.root.update()
        
        # Run in background thread
        thread = threading.Thread(target=self._ocr_worker)
        thread.start()
    
    def _ocr_worker(self):
        try:
            engine_map = {
                "tesseract": OCREngine.TESSERACT,
                "easyocr": OCREngine.EASYOCR,
                "paddleocr": OCREngine.PADDLEOCR,
                "auto": OCREngine.AUTO
            }
            
            results = self.ocr.recognize(
                self.current_image,
                engine=engine_map[self.engine_var.get()],
                preprocess=self.preprocess_var.get()
            )
            
            self.root.after(0, self._update_results, results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
    
    def _update_results(self, results):
        # Clear old results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.results_data = results
        self.detail_text.delete(1.0, tk.END)
        
        # Add new results
        for r in results:
            self.tree.insert('', tk.END, values=(
                r.engine,
                r.text[:50] + "..." if len(r.text) > 50 else r.text,
                f"{r.confidence:.2%}",
                f"{r.processing_time:.3f}"
            ))
        
        # Show best result in text area
        if results:
            best = max(results, key=lambda x: x.confidence)
            self.detail_text.insert(tk.END, f"BEST RESULT ({best.engine}, {best.confidence:.2%}):\n")
            self.detail_text.insert(tk.END, "="*50 + "\n")
            self.detail_text.insert(tk.END, best.text)
            
            # Also show all unique texts
            self.detail_text.insert(tk.END, "\n\n" + "="*50 + "\n")
            self.detail_text.insert(tk.END, "ALL RESULTS:\n")
            self.detail_text.insert(tk.END, "="*50 + "\n")
            
            seen = set()
            for r in results:
                text_key = r.text.strip().lower()
                if text_key not in seen:
                    seen.add(text_key)
                    self.detail_text.insert(tk.END, f"\n[{r.engine} | {r.confidence:.2%}]\n{r.text}\n")
        
        self.status_var.set(f"Found {len(results)} text regions")
    
    def _compare_engines(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Comparing all engines...")
        self.root.update()
        
        def compare_worker():
            try:
                comparison = self.ocr.compare_engines(self.current_image, 
                                                     preprocess=self.preprocess_var.get())
                self.root.after(0, self._show_comparison, comparison)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=compare_worker).start()
    
    def _show_comparison(self, comparison):
        win = tk.Toplevel(self.root)
        win.title("Engine Comparison")
        win.geometry("800x600")
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for engine_name, results in comparison.items():
            text.insert(tk.END, f"\n{'='*60}\n")
            text.insert(tk.END, f"{engine_name.upper()}\n")
            text.insert(tk.END, f"{'='*60}\n")
            text.insert(tk.END, f"Results count: {len(results)}\n")
            if results:
                avg_conf = sum(r.confidence for r in results) / len(results)
                text.insert(tk.END, f"Average confidence: {avg_conf:.2%}\n")
                text.insert(tk.END, f"Processing time: {results[0].processing_time:.3f}s\n\n")
                for i, r in enumerate(results[:5]):
                    text.insert(tk.END, f"{i+1}. [{r.confidence:.2%}] {r.text[:100]}\n")
            else:
                text.insert(tk.END, "No results\n")
        
        self.status_var.set("Comparison complete")
    
    def _save_results(self):
        if not self.results_data:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Text", "*.txt")]
        )
        
        if path:
            if path.endswith('.json'):
                data = [r.to_dict() for r in self.results_data]
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    for r in self.results_data:
                        f.write(f"[{r.engine}] (conf: {r.confidence:.2%})\n")
                        f.write(f"{r.text}\n")
                        f.write("-" * 50 + "\n")
            
            self.status_var.set(f"Saved to {path}")


def main():
    root = tk.Tk()
    app = OCRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
