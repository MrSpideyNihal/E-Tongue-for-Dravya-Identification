# Advanced MVP: E-Tongue for Dravya Identification with PDF Report
# Author: Nihal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -----------------------------
# 1. Training Data (Simulated)
# -----------------------------
X_train = np.array([
    [8, 1, 2],  # Sweet Herb
    [7, 2, 1],
    [1, 8, 2],  # Sour Herb
    [2, 7, 3],
    [2, 1, 9],  # Bitter Herb
    [1, 2, 8]
])

y_train = np.array(["Sweet Herb", "Sweet Herb",
                    "Sour Herb", "Sour Herb",
                    "Bitter Herb", "Bitter Herb"])

# Train simple classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)


# -----------------------------
# 2. GUI Application
# -----------------------------
class ETongueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("E-Tongue for Dravya Identification (MVP)")
        self.root.geometry("550x550")
        self.root.configure(bg="#f4f6f8")

        title = tk.Label(root, text="🌿 E-Tongue MVP", font=("Arial", 18, "bold"), bg="#f4f6f8", fg="#2e7d32")
        title.pack(pady=10)

        subtitle = tk.Label(root, text="Simulated Taste Analysis using ML", font=("Arial", 12), bg="#f4f6f8", fg="#555")
        subtitle.pack()

        # Frame for sliders
        frame = tk.Frame(root, bg="#f4f6f8")
        frame.pack(pady=20)

        self.sliders = {}
        for taste in ["Sweetness", "Sourness", "Bitterness"]:
            row = tk.Frame(frame, bg="#f4f6f8")
            row.pack(pady=10, fill="x")

            label = tk.Label(row, text=f"{taste} (0-10):", font=("Arial", 11), bg="#f4f6f8")
            label.pack(side="left", padx=5)

            slider = ttk.Scale(row, from_=0, to=10, orient="horizontal", length=250)
            slider.set(5)
            slider.pack(side="left", padx=10)
            self.sliders[taste] = slider

        # Analyze Button
        analyze_btn = tk.Button(root, text="🔍 Analyze Sample", command=self.predict_taste,
                                font=("Arial", 12, "bold"), bg="#388e3c", fg="white", relief="flat", padx=10, pady=5)
        analyze_btn.pack(pady=15)

        # Result Display
        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f4f6f8", fg="#0d47a1")
        self.result_label.pack(pady=10)

        # Probability Bars
        self.prob_frame = tk.Frame(root, bg="#f4f6f8")
        self.prob_frame.pack(pady=10, fill="x")

        self.prob_bars = {}
        self.prob_values = {}
        for label in model.classes_:
            lbl = tk.Label(self.prob_frame, text=label, font=("Arial", 11), bg="#f4f6f8")
            lbl.pack(anchor="w", padx=20)
            bar = ttk.Progressbar(self.prob_frame, length=400, maximum=100)
            bar.pack(pady=3)
            self.prob_bars[label] = bar
            self.prob_values[label] = 0.0

        # Download Report Button
        self.report_btn = tk.Button(root, text="📥 Download Report", command=self.download_report,
                                    font=("Arial", 12, "bold"), bg="#1976d2", fg="white", relief="flat", padx=10, pady=5)
        self.report_btn.pack(pady=15)
        self.report_btn.config(state="disabled")  # disabled until analysis done

    def predict_taste(self):
        # Collect slider values
        sweet = self.sliders["Sweetness"].get()
        sour = self.sliders["Sourness"].get()
        bitter = self.sliders["Bitterness"].get()

        sample = np.array([[sweet, sour, bitter]])
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]

        # Update result
        self.result_label.config(text=f"Predicted Herb: {prediction}")

        # Update probability bars & store values
        for label, prob in zip(model.classes_, probabilities):
            self.prob_bars[label].config(value=prob * 100)
            self.prob_values[label] = prob * 100

        self.prediction = prediction
        self.sensor_values = {"Sweetness": sweet, "Sourness": sour, "Bitterness": bitter}
        self.report_btn.config(state="normal")  # enable report download

    def download_report(self):
        # Ask user for file save location
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                 filetypes=[("PDF Files", "*.pdf")],
                                                 title="Save Report As")
        if not file_path:
            return

        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4

            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(200, height - 50, "E-Tongue Report")

            # Predicted result
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 100, f"Predicted Herb: {self.prediction}")

            # Sensor values
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 140, "Sensor Readings:")
            y = height - 160
            for taste, value in self.sensor_values.items():
                c.drawString(70, y, f"{taste}: {value:.2f}")
                y -= 20

            # Probabilities
            c.drawString(50, y - 10, "Confidence Levels:")
            y -= 30
            for label, prob in self.prob_values.items():
                c.drawString(70, y, f"{label}: {prob:.1f}%")
                y -= 20

            # Footer
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(50, 50, "Made by Nihal Rodge")

            c.save()
            messagebox.showinfo("Success", f"Report saved successfully!\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")


# Run the app
root = tk.Tk()
app = ETongueApp(root)
root.mainloop()
