#!/usr/bin/env python3
"""
GUI wrapper for extract_photos.py.
Asks for input/output folders and shows a progress bar.
Build as .exe with: pyinstaller --onefile --windowed photo_scan_gui.py
"""

import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path

import cv2

from extract_photos import detect_photos, process_single, IMAGE_EXTENSIONS


class PhotoScanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SplitPhotoScan - Extracteur de photos")
        self.root.resizable(False, False)

        frame = ttk.Frame(root, padding=20)
        frame.grid()

        # Input folder
        ttk.Label(frame, text="Dossier des scans :").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.input_var, width=50).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(frame, text="Parcourir…", command=self.browse_input).grid(row=0, column=2, padx=(5, 0))

        # Output folder
        ttk.Label(frame, text="Dossier de sortie :").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.output_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.output_var, width=50).grid(row=1, column=1, padx=(5, 0), pady=(10, 0))
        ttk.Button(frame, text="Parcourir…", command=self.browse_output).grid(row=1, column=2, padx=(5, 0), pady=(10, 0))

        # Format
        ttk.Label(frame, text="Format :").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.format_var = tk.StringVar(value="jpg")
        fmt_frame = ttk.Frame(frame)
        fmt_frame.grid(row=2, column=1, sticky="w", padx=(5, 0), pady=(10, 0))
        ttk.Radiobutton(fmt_frame, text="JPG", variable=self.format_var, value="jpg").pack(side="left")
        ttk.Radiobutton(fmt_frame, text="PNG", variable=self.format_var, value="png").pack(side="left", padx=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(frame, length=400, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=3, pady=(20, 0))

        # Status label
        self.status_var = tk.StringVar(value="En attente…")
        ttk.Label(frame, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, pady=(5, 0))

        # Start button
        self.start_btn = ttk.Button(frame, text="Lancer l'extraction", command=self.start)
        self.start_btn.grid(row=5, column=0, columnspan=3, pady=(15, 0))

    def browse_input(self):
        path = filedialog.askdirectory(title="Sélectionner le dossier des scans")
        if path:
            self.input_var.set(path)
            # Pre-fill output as a subfolder
            if not self.output_var.get():
                self.output_var.set(str(Path(path) / "extracted"))

    def browse_output(self):
        path = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if path:
            self.output_var.set(path)

    def start(self):
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()

        if not input_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de scans.")
            return
        if not output_dir:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier de sortie.")
            return

        input_path = Path(input_dir)
        if not input_path.is_dir():
            messagebox.showerror("Erreur", f"Le dossier n'existe pas :\n{input_dir}")
            return

        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            messagebox.showwarning("Aucun fichier", "Aucune image trouvée dans ce dossier.")
            return

        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = len(files)

        thread = threading.Thread(
            target=self.run_extraction,
            args=(files, Path(output_dir), self.format_var.get()),
            daemon=True,
        )
        thread.start()

    def run_extraction(self, files, output_dir, fmt):
        total_photos = 0

        for idx, f in enumerate(files):
            self.root.after(0, self.update_status, f"Traitement de {f.name}… ({idx + 1}/{len(files)})")

            count = process_single(f, output_dir, fmt, min_area_ratio=0.01, debug=False)
            total_photos += count

            self.root.after(0, self.update_progress, idx + 1)

        self.root.after(0, self.done, total_photos, len(files))

    def update_status(self, text):
        self.status_var.set(text)

    def update_progress(self, value):
        self.progress["value"] = value

    def done(self, total_photos, total_files):
        self.status_var.set(f"Terminé ! {total_photos} photo(s) extraites de {total_files} scan(s).")
        self.start_btn.config(state="normal")
        messagebox.showinfo(
            "Extraction terminée",
            f"{total_photos} photo(s) extraites de {total_files} scan(s).\n\n"
            f"Dossier de sortie :\n{self.output_var.get()}"
        )


def main():
    root = tk.Tk()
    PhotoScanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
