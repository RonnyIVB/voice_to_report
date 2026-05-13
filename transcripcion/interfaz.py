import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
from transcriptor import transcribe

class TranscriptorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcriptor de Audio OGG (Whisper)")
        self.root.geometry("700x500")

        # Variable para la ruta del archivo
        self.file_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # --- Frame Superior: Selección de archivo ---
        frame_top = tk.LabelFrame(self.root, text="Archivo de Audio", padx=10, pady=10)
        frame_top.pack(fill=tk.X, padx=15, pady=10)

        tk.Button(frame_top, text="Buscar archivo .ogg", command=self.select_file, bg="#e1e1e1").pack(side=tk.LEFT, padx=5)
        tk.Entry(frame_top, textvariable=self.file_path, state='readonly', width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # --- Botón de Acción ---
        self.btn_transcribe = tk.Button(
            self.root, 
            text="Iniciar Transcripción", 
            command=self.start_transcription_thread, 
            state=tk.DISABLED,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            pady=5
        )
        self.btn_transcribe.pack(pady=10)

        # --- Estado ---
        self.lbl_status = tk.Label(self.root, text="Por favor, selecciona un archivo .ogg", fg="gray")
        self.lbl_status.pack()

        # --- Área de texto (Transcripción) ---
        frame_text = tk.LabelFrame(self.root, text="Resultado de la Transcripción", padx=10, pady=10)
        frame_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.text_area = scrolledtext.ScrolledText(frame_text, wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=(("Archivos OGG", "*.ogg"), ("Todos los archivos", "*.*"))
        )
        if filename:
            self.file_path.set(filename)
            self.btn_transcribe.config(state=tk.NORMAL)
            self.lbl_status.config(text=f"Archivo cargado: {os.path.basename(filename)}", fg="black")
            self.text_area.delete(1.0, tk.END)

    def start_transcription_thread(self):
        path = self.file_path.get()
        if not path:
            return

        # Bloqueamos la interfaz mientras procesa
        self.btn_transcribe.config(state=tk.DISABLED)
        self.text_area.delete(1.0, tk.END)
        self.lbl_status.config(text="Procesando audio con Whisper... (Esto puede tardar unos segundos)", fg="blue")
        
        # Ejecutar en un hilo separado para no congelar la ventana
        thread = threading.Thread(target=self.run_transcription, args=(path,), daemon=True)
        thread.start()

    def run_transcription(self, path):
        try:
            # Llamada al método del archivo transcriptor.py
            texto = transcribe(path)
            
            # Actualizar la UI desde el hilo principal usando .after()
            self.root.after(0, self.update_ui_success, texto)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_ui_success(self, texto):
        self.text_area.insert(tk.END, texto)
        self.lbl_status.config(text="¡Transcripción completada con éxito!", fg="green")
        self.btn_transcribe.config(state=tk.NORMAL)

    def show_error(self, error_msg):
        messagebox.showerror("Error de Transcripción", f"Ocurrió un error:\n{error_msg}")
        self.lbl_status.config(text="Error en el proceso.", fg="red")
        self.btn_transcribe.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptorApp(root)
    root.mainloop()
