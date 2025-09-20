    # Enhanced E-Tongue for Dravya Identification System
    # Addressing the Ministry of Ayush Problem Statement
    # Author: Enhanced by Claude for Straw-Hats Team

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    import datetime
    import json

    # ----------------------------- 
    # 1. Enhanced Training Data with 6 Rasa (Taste) Categories
    # -----------------------------

    class ETongueDataGenerator:
        """Generate comprehensive training data for Ayurvedic herbs based on 6 Rasa"""
        
        @staticmethod
        def generate_herb_data():
            """Generate realistic sensor data for various Ayurvedic herbs"""
            
            # Comprehensive herb database with 6 taste parameters + phytochemical indicators
            herbs_data = {
                # Madhura (Sweet) dominant herbs
                "Ashwagandha": {"sweet": 8.2, "sour": 1.1, "salty": 0.8, "pungent": 1.5, "bitter": 2.3, "astringent": 2.8, "alkaloids": 7.5, "glycosides": 8.1, "phenolic": 4.2},
                "Shatavari": {"sweet": 7.8, "sour": 1.5, "salty": 0.5, "pungent": 0.8, "bitter": 1.9, "astringent": 3.2, "alkaloids": 6.2, "glycosides": 7.8, "phenolic": 3.8},
                "Yashtimadhu": {"sweet": 9.2, "sour": 0.8, "salty": 0.3, "pungent": 0.5, "bitter": 1.2, "astringent": 1.8, "alkaloids": 8.5, "glycosides": 9.1, "phenolic": 5.5},
                
                # Amla (Sour) dominant herbs  
                "Amalaki": {"sweet": 3.2, "sour": 8.5, "salty": 1.1, "pungent": 2.1, "bitter": 2.8, "astringent": 7.2, "alkaloids": 4.5, "glycosides": 6.2, "phenolic": 8.9},
                "Kokam": {"sweet": 1.8, "sour": 9.1, "salty": 0.9, "pungent": 1.2, "bitter": 1.5, "astringent": 4.8, "alkaloids": 3.2, "glycosides": 4.8, "phenolic": 7.5},
                
                # Lavana (Salty) herbs
                "Saindhava": {"sweet": 0.5, "sour": 1.2, "salty": 9.5, "pungent": 0.8, "bitter": 0.3, "astringent": 0.5, "alkaloids": 1.1, "glycosides": 0.8, "phenolic": 0.5},
                "Yavakshara": {"sweet": 1.1, "sour": 2.1, "salty": 8.8, "pungent": 1.5, "bitter": 1.2, "astringent": 1.8, "alkaloids": 2.5, "glycosides": 1.8, "phenolic": 1.2},
                
                # Katu (Pungent) dominant herbs
                "Maricha": {"sweet": 1.2, "sour": 1.8, "salty": 0.8, "pungent": 9.2, "bitter": 2.1, "astringent": 1.5, "alkaloids": 7.8, "glycosides": 3.2, "phenolic": 6.5},
                "Shunthi": {"sweet": 2.1, "sour": 1.5, "salty": 0.5, "pungent": 8.8, "bitter": 1.8, "astringent": 2.2, "alkaloids": 6.5, "glycosides": 4.1, "phenolic": 5.8},
                "Pippali": {"sweet": 1.8, "sour": 1.2, "salty": 0.8, "pungent": 8.5, "bitter": 2.5, "astringent": 1.8, "alkaloids": 8.2, "glycosides": 3.8, "phenolic": 6.2},
                
                # Tikta (Bitter) dominant herbs
                "Neem": {"sweet": 0.8, "sour": 1.2, "salty": 0.5, "pungent": 1.8, "bitter": 9.5, "astringent": 3.8, "alkaloids": 8.9, "glycosides": 5.5, "phenolic": 7.8},
                "Kalmegh": {"sweet": 0.5, "sour": 1.5, "salty": 0.3, "pungent": 2.1, "bitter": 9.8, "astringent": 2.5, "alkaloids": 9.2, "glycosides": 4.8, "phenolic": 8.5},
                "Kutki": {"sweet": 1.1, "sour": 1.8, "salty": 0.8, "pungent": 2.5, "bitter": 9.2, "astringent": 3.2, "alkaloids": 8.8, "glycosides": 5.2, "phenolic": 7.9},
                
                # Kashaya (Astringent) dominant herbs
                "Arjuna": {"sweet": 1.5, "sour": 2.1, "salty": 0.5, "pungent": 1.2, "bitter": 3.8, "astringent": 8.9, "alkaloids": 4.5, "glycosides": 6.8, "phenolic": 8.2},
                "Lodhra": {"sweet": 2.1, "sour": 1.8, "salty": 0.8, "pungent": 0.9, "bitter": 3.5, "astringent": 9.2, "alkaloids": 3.8, "glycosides": 7.2, "phenolic": 8.8},
                "Jambu": {"sweet": 1.8, "sour": 2.5, "salty": 0.5, "pungent": 1.1, "bitter": 2.8, "astringent": 8.5, "alkaloids": 4.2, "glycosides": 6.5, "phenolic": 7.8}
            }
            
            # Generate multiple samples per herb with natural variation
            X_data = []
            y_data = []
            
            for herb_name, base_values in herbs_data.items():
                for _ in range(15):  # 15 samples per herb for robustness
                    # Add natural variation (±10% noise)
                    sample = []
                    for key, value in base_values.items():
                        noise = np.random.normal(0, value * 0.1)  # 10% variation
                        sample.append(max(0, min(10, value + noise)))  # Clamp to 0-10 range
                    
                    X_data.append(sample)
                    y_data.append(herb_name)
            
            return np.array(X_data), np.array(y_data), list(herbs_data.keys()), list(base_values.keys())

    # ----------------------------- 
    # 2. Advanced ML Models
    # -----------------------------

    class HerbClassificationModel:
        def __init__(self):
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.nn_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
            self.feature_names = []
            self.herb_names = []
            
        def train_models(self):
            """Train both Random Forest and Neural Network models"""
            X, y, herb_names, feature_names = ETongueDataGenerator.generate_herb_data()
            
            # Store for later use
            self.herb_names = herb_names
            self.feature_names = feature_names
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features for neural network
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.rf_model.fit(X_train, y_train)
            self.nn_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = self.rf_model.predict(X_test)
            nn_pred = self.nn_model.predict(X_test_scaled)
            
            self.rf_accuracy = accuracy_score(y_test, rf_pred)
            self.nn_accuracy = accuracy_score(y_test, nn_pred)
            
            self.is_trained = True
            
            return {
                'rf_accuracy': self.rf_accuracy,
                'nn_accuracy': self.nn_accuracy,
                'total_samples': len(X),
                'herbs_count': len(herb_names)
            }
        
        def predict_herb(self, sensor_data, use_ensemble=True):
            """Predict herb using ensemble or individual models"""
            if not self.is_trained:
                raise ValueError("Models not trained yet!")
            
            sample = np.array([sensor_data])
            sample_scaled = self.scaler.transform(sample)
            
            # Get predictions from both models
            rf_pred = self.rf_model.predict(sample)[0]
            rf_proba = self.rf_model.predict_proba(sample)[0]
            
            nn_pred = self.nn_model.predict(sample_scaled)[0]
            nn_proba = self.nn_model.predict_proba(sample_scaled)[0]
            
            if use_ensemble:
                # Ensemble prediction (weighted average)
                ensemble_proba = (rf_proba * 0.6 + nn_proba * 0.4)  # RF weighted higher
                final_pred = self.rf_model.classes_[np.argmax(ensemble_proba)]
                confidence = np.max(ensemble_proba) * 100
            else:
                final_pred = rf_pred
                ensemble_proba = rf_proba
                confidence = np.max(rf_proba) * 100
            
            return {
                'prediction': final_pred,
                'confidence': confidence,
                'rf_prediction': rf_pred,
                'nn_prediction': nn_pred,
                'probabilities': dict(zip(self.rf_model.classes_, ensemble_proba * 100)),
                'rf_probabilities': dict(zip(self.rf_model.classes_, rf_proba * 100)),
                'nn_probabilities': dict(zip(self.nn_model.classes_, nn_proba * 100))
            }

    # ----------------------------- 
    # 3. Enhanced GUI Application
    # -----------------------------

    class EnhancedETongueApp:
        def __init__(self, root):
            self.root = root
            self.root.title("AI-Enabled E-Tongue for Dravya Identification - Ministry of Ayush")
            self.root.geometry("900x800")
            self.root.configure(bg="#f0f8f0")
            
            # Initialize ML model
            self.model = HerbClassificationModel()
            self.current_analysis = None
            
            self.setup_gui()
            self.train_models()
            
        def setup_gui(self):
            # Main title
            title = tk.Label(self.root, text="🌿 AI-Enabled E-Tongue for Dravya Identification", 
                            font=("Arial", 20, "bold"), bg="#f0f8f0", fg="#2e7d32")
            title.pack(pady=10)
            
            subtitle = tk.Label(self.root, text="Ministry of Ayush - Quality Assessment System", 
                            font=("Arial", 14), bg="#f0f8f0", fg="#1b5e20")
            subtitle.pack()
            
            # Create notebook for tabs
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Sensor Input Tab
            self.sensor_frame = ttk.Frame(notebook)
            notebook.add(self.sensor_frame, text="Sensor Inputs")
            self.setup_sensor_tab()
            
            # Analysis Results Tab
            self.results_frame = ttk.Frame(notebook)
            notebook.add(self.results_frame, text="Analysis Results")
            self.setup_results_tab()
            
            # Database Tab
            self.database_frame = ttk.Frame(notebook)
            notebook.add(self.database_frame, text="Herb Database")
            self.setup_database_tab()
            
        def setup_sensor_tab(self):
            # Sensor input section
            sensor_label = tk.Label(self.sensor_frame, text="Multi-Sensor E-Tongue Array Readings", 
                                font=("Arial", 16, "bold"), fg="#2e7d32")
            sensor_label.pack(pady=10)
            
            # Create input fields for all 9 parameters
            self.sensor_vars = {}
            self.sensor_entries = {}
            
            # Main sensor parameters (6 Rasa)
            main_frame = tk.LabelFrame(self.sensor_frame, text="Rasa (Taste) Sensors (0-10 scale)", 
                                    font=("Arial", 12, "bold"), padx=10, pady=10)
            main_frame.pack(fill="x", padx=20, pady=10)
            
            rasa_params = ["sweet", "sour", "salty", "pungent", "bitter", "astringent"]
            rasa_labels = ["Madhura (Sweet)", "Amla (Sour)", "Lavana (Salty)", 
                        "Katu (Pungent)", "Tikta (Bitter)", "Kashaya (Astringent)"]
            
            for i, (param, label) in enumerate(zip(rasa_params, rasa_labels)):
                row = tk.Frame(main_frame)
                row.pack(fill="x", pady=5)
                
                tk.Label(row, text=f"{label}:", font=("Arial", 11), width=20, anchor="w").pack(side="left")
                
                var = tk.DoubleVar(value=5.0)
                self.sensor_vars[param] = var
                
                entry = tk.Entry(row, textvariable=var, font=("Arial", 11), width=10)
                entry.pack(side="left", padx=5)
                self.sensor_entries[param] = entry
                
                scale = ttk.Scale(row, from_=0, to=10, orient="horizontal", length=200, variable=var)
                scale.pack(side="left", padx=10)
            
            # Phytochemical indicators
            phyto_frame = tk.LabelFrame(self.sensor_frame, text="Phytochemical Indicators (0-10 scale)", 
                                    font=("Arial", 12, "bold"), padx=10, pady=10)
            phyto_frame.pack(fill="x", padx=20, pady=10)
            
            phyto_params = ["alkaloids", "glycosides", "phenolic"]
            phyto_labels = ["Alkaloid Content", "Glycoside Content", "Phenolic Content"]
            
            for param, label in zip(phyto_params, phyto_labels):
                row = tk.Frame(phyto_frame)
                row.pack(fill="x", pady=5)
                
                tk.Label(row, text=f"{label}:", font=("Arial", 11), width=20, anchor="w").pack(side="left")
                
                var = tk.DoubleVar(value=5.0)
                self.sensor_vars[param] = var
                
                entry = tk.Entry(row, textvariable=var, font=("Arial", 11), width=10)
                entry.pack(side="left", padx=5)
                self.sensor_entries[param] = entry
                
                scale = ttk.Scale(row, from_=0, to=10, orient="horizontal", length=200, variable=var)
                scale.pack(side="left", padx=10)
            
            # Control buttons
            button_frame = tk.Frame(self.sensor_frame)
            button_frame.pack(pady=20)
            
            analyze_btn = tk.Button(button_frame, text="🔬 Analyze Sample", 
                                command=self.analyze_sample,
                                font=("Arial", 14, "bold"), bg="#4caf50", fg="white", 
                                relief="flat", padx=20, pady=10)
            analyze_btn.pack(side="left", padx=10)
            
            reset_btn = tk.Button(button_frame, text="🔄 Reset Values", 
                                command=self.reset_values,
                                font=("Arial", 12), bg="#ff9800", fg="white", 
                                relief="flat", padx=15, pady=8)
            reset_btn.pack(side="left", padx=10)
            
        def setup_results_tab(self):
            # Results display
            results_label = tk.Label(self.results_frame, text="Analysis Results & Quality Assessment", 
                                    font=("Arial", 16, "bold"), fg="#2e7d32")
            results_label.pack(pady=10)
            
            # Prediction result
            self.result_frame = tk.LabelFrame(self.results_frame, text="Herb Identification", 
                                            font=("Arial", 12, "bold"), padx=10, pady=10)
            self.result_frame.pack(fill="x", padx=20, pady=10)
            
            self.result_label = tk.Label(self.result_frame, text="No analysis performed yet", 
                                        font=("Arial", 16, "bold"), fg="#d32f2f")
            self.result_label.pack(pady=10)
            
            self.confidence_label = tk.Label(self.result_frame, text="", 
                                            font=("Arial", 12), fg="#1976d2")
            self.confidence_label.pack()
            
            # Model comparison
            self.model_frame = tk.LabelFrame(self.results_frame, text="Model Predictions Comparison", 
                                            font=("Arial", 12, "bold"), padx=10, pady=10)
            self.model_frame.pack(fill="x", padx=20, pady=10)
            
            # Probability distribution
            self.prob_frame = tk.LabelFrame(self.results_frame, text="Confidence Distribution", 
                                        font=("Arial", 12, "bold"), padx=10, pady=10)
            self.prob_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Report generation
            report_frame = tk.Frame(self.results_frame)
            report_frame.pack(pady=20)
            
            self.report_btn = tk.Button(report_frame, text="📊 Generate Detailed Report", 
                                    command=self.generate_report,
                                    font=("Arial", 12, "bold"), bg="#1976d2", fg="white", 
                                    relief="flat", padx=20, pady=10, state="disabled")
            self.report_btn.pack(side="left", padx=10)
            
            save_btn = tk.Button(report_frame, text="💾 Save to Database", 
                                command=self.save_to_database,
                                font=("Arial", 12), bg="#795548", fg="white", 
                                relief="flat", padx=15, pady=8, state="disabled")
            save_btn.pack(side="left", padx=10)
            
        def setup_database_tab(self):
            db_label = tk.Label(self.database_frame, text="Ayurvedic Herbs Reference Database", 
                            font=("Arial", 16, "bold"), fg="#2e7d32")
            db_label.pack(pady=10)
            
            # Create treeview for herb database
            columns = ["Herb", "Primary Rasa", "Secondary Rasa", "Therapeutic Use"]
            self.herb_tree = ttk.Treeview(self.database_frame, columns=columns, show="headings", height=15)
            
            for col in columns:
                self.herb_tree.heading(col, text=col)
                self.herb_tree.column(col, width=150)
            
            self.herb_tree.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Populate database
            self.populate_herb_database()
            
        def train_models(self):
            """Train the ML models on startup"""
            try:
                self.training_progress = tk.Toplevel(self.root)
                self.training_progress.title("Training Models...")
                self.training_progress.geometry("400x200")
                self.training_progress.configure(bg="#f0f8f0")
                
                label = tk.Label(self.training_progress, text="Training AI Models...", 
                            font=("Arial", 14), bg="#f0f8f0")
                label.pack(pady=30)
                
                progress = ttk.Progressbar(self.training_progress, length=300, mode='indeterminate')
                progress.pack(pady=20)
                progress.start()
                
                self.root.after(100, self.complete_training)
                
            except Exception as e:
                messagebox.showerror("Training Error", f"Failed to train models: {e}")
        
        def complete_training(self):
            try:
                results = self.model.train_models()
                self.training_progress.destroy()
                
                messagebox.showinfo("Training Complete", 
                                f"Models trained successfully!\n\n"
                                f"Random Forest Accuracy: {results['rf_accuracy']:.2%}\n"
                                f"Neural Network Accuracy: {results['nn_accuracy']:.2%}\n"
                                f"Total Training Samples: {results['total_samples']}\n"
                                f"Herbs in Database: {results['herbs_count']}")
            except Exception as e:
                self.training_progress.destroy()
                messagebox.showerror("Training Error", f"Failed to complete training: {e}")
        
        def analyze_sample(self):
            """Perform comprehensive herb analysis"""
            try:
                # Collect sensor data
                sensor_data = [self.sensor_vars[param].get() for param in self.model.feature_names]
                
                # Get prediction
                results = self.model.predict_herb(sensor_data)
                self.current_analysis = results
                self.current_analysis['sensor_data'] = dict(zip(self.model.feature_names, sensor_data))
                self.current_analysis['timestamp'] = datetime.datetime.now()
                
                # Update results display
                self.result_label.config(text=f"Identified Herb: {results['prediction']}", 
                                    fg="#2e7d32")
                self.confidence_label.config(text=f"Confidence: {results['confidence']:.1f}%")
                
                # Clear previous model comparison
                for widget in self.model_frame.winfo_children():
                    widget.destroy()
                
                # Show model comparison
                rf_label = tk.Label(self.model_frame, text=f"Random Forest: {results['rf_prediction']}", 
                                font=("Arial", 11))
                rf_label.pack(anchor="w")
                
                nn_label = tk.Label(self.model_frame, text=f"Neural Network: {results['nn_prediction']}", 
                                font=("Arial", 11))
                nn_label.pack(anchor="w")
                
                # Update probability bars
                self.update_probability_display(results['probabilities'])
                
                # Enable report generation
                self.report_btn.config(state="normal")
                
            except Exception as e:
                messagebox.showerror("Analysis Error", f"Failed to analyze sample: {e}")
        
        def update_probability_display(self, probabilities):
            """Update the probability distribution display"""
            # Clear previous bars
            for widget in self.prob_frame.winfo_children():
                widget.destroy()
            
            # Sort probabilities in descending order
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            for herb, prob in sorted_probs[:8]:  # Show top 8 herbs
                herb_frame = tk.Frame(self.prob_frame)
                herb_frame.pack(fill="x", padx=5, pady=2)
                
                label = tk.Label(herb_frame, text=f"{herb}:", font=("Arial", 10), width=12, anchor="w")
                label.pack(side="left")
                
                bar = ttk.Progressbar(herb_frame, length=300, maximum=100, value=prob)
                bar.pack(side="left", padx=5)
                
                value_label = tk.Label(herb_frame, text=f"{prob:.1f}%", font=("Arial", 9))
                value_label.pack(side="left", padx=5)
        
        def reset_values(self):
            """Reset all sensor values to default"""
            for var in self.sensor_vars.values():
                var.set(5.0)
        
        def populate_herb_database(self):
            """Populate the herb database treeview"""
            herb_info = [
                ("Ashwagandha", "Madhura (Sweet)", "Tikta (Bitter)", "Rasayana, Balya"),
                ("Shatavari", "Madhura (Sweet)", "Tikta (Bitter)", "Rasayana, Stanya"),
                ("Yashtimadhu", "Madhura (Sweet)", "None", "Rasayana, Kasahara"),
                ("Amalaki", "Amla (Sour)", "Kashaya (Astringent)", "Rasayana, Tridoshahara"),
                ("Kokam", "Amla (Sour)", "Madhura (Sweet)", "Pittashamaka, Digestive"),
                ("Saindhava", "Lavana (Salty)", "None", "Dipana, Pachana"),
                ("Maricha", "Katu (Pungent)", "None", "Dipana, Kaphavatahara"),
                ("Shunthi", "Katu (Pungent)", "Madhura (Sweet)", "Dipana, Shoolahara"),
                ("Pippali", "Katu (Pungent)", "Madhura (Sweet)", "Rasayana, Kaphavatahara"),
                ("Neem", "Tikta (Bitter)", "Kashaya (Astringent)", "Krimighna, Kandughna"),
                ("Kalmegh", "Tikta (Bitter)", "None", "Jwaraghna, Yakrituttejaka"),
                ("Kutki", "Tikta (Bitter)", "Katu (Pungent)", "Jwaraghna, Dipana"),
                ("Arjuna", "Kashaya (Astringent)", "Tikta (Bitter)", "Hridaya, Raktashodhaka"),
                ("Lodhra", "Kashaya (Astringent)", "Madhura (Sweet)", "Stambhana, Vranahara"),
                ("Jambu", "Kashaya (Astringent)", "Amla (Sour)", "Pramehaghna, Raktashodhaka")
            ]
            
            for herb_data in herb_info:
                self.herb_tree.insert("", "end", values=herb_data)
        
        def generate_report(self):
            """Generate comprehensive PDF report"""
            if not self.current_analysis:
                messagebox.showwarning("No Data", "Please perform an analysis first")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf")],
                title="Save Analysis Report"
            )
            
            if not file_path:
                return
            
            try:
                c = canvas.Canvas(file_path, pagesize=A4)
                width, height = A4
                y_position = height - 50
                
                # Header
                c.setFont("Helvetica-Bold", 18)
                c.drawString(50, y_position, "E-Tongue Dravya Identification Report")
                y_position -= 30
                
                c.setFont("Helvetica", 12)
                c.drawString(50, y_position, f"Ministry of Ayush - Quality Assessment System")
                y_position -= 20
                c.drawString(50, y_position, f"Generated: {self.current_analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                y_position -= 40
                
                # Analysis Results
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "ANALYSIS RESULTS")
                y_position -= 25
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, f"Identified Herb: {self.current_analysis['prediction']}")
                y_position -= 20
                
                c.setFont("Helvetica", 11)
                c.drawString(50, y_position, f"Confidence Level: {self.current_analysis['confidence']:.1f}%")
                y_position -= 15
                c.drawString(50, y_position, f"Random Forest Prediction: {self.current_analysis['rf_prediction']}")
                y_position -= 15
                c.drawString(50, y_position, f"Neural Network Prediction: {self.current_analysis['nn_prediction']}")
                y_position -= 30
                
                # Sensor Readings
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, "SENSOR READINGS")
                y_position -= 20
                
                c.setFont("Helvetica", 10)
                for param, value in self.current_analysis['sensor_data'].items():
                    param_name = param.replace('_', ' ').title()
                    c.drawString(70, y_position, f"{param_name}: {value:.2f}")
                    y_position -= 15
                
                y_position -= 20
                
                # Top Predictions
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, "TOP HERB MATCHES")
                y_position -= 20
                
                c.setFont("Helvetica", 10)
                sorted_probs = sorted(self.current_analysis['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                
                for i, (herb, prob) in enumerate(sorted_probs[:5]):
                    c.drawString(70, y_position, f"{i+1}. {herb}: {prob:.1f}%")
                    y_position -= 15
                
                y_position -= 30
                
                # Quality Assessment
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, "QUALITY ASSESSMENT")
                y_position -= 20
                
                c.setFont("Helvetica", 10)
                confidence = self.current_analysis['confidence']
                if confidence >= 80:
                    quality_status = "HIGH CONFIDENCE - Sample matches reference standard"
                elif confidence >= 60:
                    quality_status = "MODERATE CONFIDENCE - Further verification recommended"
                else:
                    quality_status = "LOW CONFIDENCE - Possible adulteration or degradation"
                
                c.drawString(70, y_position, f"Quality Status: {quality_status}")
                y_position -= 15
                
                # Model Performance
                y_position -= 30
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, "MODEL PERFORMANCE")
                y_position -= 20
                
                c.setFont("Helvetica", 10)
                c.drawString(70, y_position, f"Random Forest Accuracy: {self.model.rf_accuracy:.2%}")
                y_position -= 15
                c.drawString(70, y_position, f"Neural Network Accuracy: {self.model.nn_accuracy:.2%}")
                y_position -= 15
                
                # Footer
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(50, 50, "Generated by AI-Enabled E-Tongue System")
                c.drawString(50, 35, "Ministry of Ayush - All India Institute of Ayurveda (AIIA)")
                c.drawString(50, 20, "Developed by Straw-Hats Team")
                
                c.save()
                messagebox.showinfo("Success", f"Report saved successfully!\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {e}")
        
        def save_to_database(self):
            """Save analysis results to local database"""
            if not self.current_analysis:
                messagebox.showwarning("No Data", "Please perform an analysis first")
                return
            
            try:
                # Create database entry
                db_entry = {
                    'timestamp': self.current_analysis['timestamp'].isoformat(),
                    'prediction': self.current_analysis['prediction'],
                    'confidence': self.current_analysis['confidence'],
                    'sensor_data': self.current_analysis['sensor_data'],
                    'probabilities': self.current_analysis['probabilities']
                }
                
                # Save to JSON file (in real implementation, would use proper database)
                filename = f"etongue_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON Files", "*.json")],
                    initialname=filename,
                    title="Save Analysis Data"
                )
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(db_entry, f, indent=2, default=str)
                    
                    messagebox.showinfo("Success", f"Analysis data saved to:\n{file_path}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {e}")

    # ----------------------------- 
    # 4. Additional Features for Ministry of Ayush Requirements
    # -----------------------------

    class QualityAssessment:
        """Advanced quality assessment features"""
        
        @staticmethod
        def detect_adulteration(sensor_data, prediction_confidence):
            """Detect possible adulteration based on sensor anomalies"""
            alerts = []
            
            # Check for unusual sensor combinations
            sweet, sour, salty, pungent, bitter, astringent = sensor_data[:6]
            alkaloids, glycosides, phenolic = sensor_data[6:9]
            
            # Alert conditions
            if prediction_confidence < 50:
                alerts.append("LOW CONFIDENCE: Possible adulteration or unknown sample")
            
            if sum(sensor_data[:6]) < 10 or sum(sensor_data[:6]) > 50:
                alerts.append("SENSOR ANOMALY: Unusual taste profile detected")
            
            if alkaloids > 9 and glycosides < 2:
                alerts.append("PHYTOCHEMICAL ALERT: Unusual alkaloid/glycoside ratio")
            
            if all(val < 2 for val in sensor_data[:6]):
                alerts.append("QUALITY ALERT: Very low taste intensity - possible degradation")
            
            return alerts
        
        @staticmethod
        def standardization_check(herb_name, sensor_data):
            """Check if sample meets standardization criteria"""
            # This would be based on official Ayurvedic Pharmacopoeia standards
            # For demo purposes, using simplified criteria
            
            standards = {
                "Ashwagandha": {"alkaloids": (6.0, 9.0), "sweet": (7.0, 9.5)},
                "Neem": {"bitter": (8.5, 10.0), "alkaloids": (8.0, 10.0)},
                "Amalaki": {"sour": (7.5, 9.5), "phenolic": (8.0, 10.0)},
                # Add more standards as needed
            }
            
            if herb_name not in standards:
                return "No standardization criteria available"
            
            std = standards[herb_name]
            issues = []
            
            for param, (min_val, max_val) in std.items():
                param_idx = ["sweet", "sour", "salty", "pungent", "bitter", "astringent", 
                            "alkaloids", "glycosides", "phenolic"].index(param)
                value = sensor_data[param_idx]
                
                if value < min_val or value > max_val:
                    issues.append(f"{param.title()}: {value:.1f} (expected: {min_val}-{max_val})")
            
            if not issues:
                return "✓ Sample meets standardization criteria"
            else:
                return "⚠ Standardization issues: " + "; ".join(issues)

    # ----------------------------- 
    # 5. Main Application Launch
    # -----------------------------

    def main():
        """Launch the Enhanced E-Tongue Application"""
        root = tk.Tk()
        app = EnhancedETongueApp(root)
        
        # Add menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Sample Data", command=lambda: messagebox.showinfo("Info", "Feature coming soon"))
        file_menu.add_command(label="Export Database", command=lambda: messagebox.showinfo("Info", "Feature coming soon"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Calibrate Sensors", command=lambda: messagebox.showinfo("Info", "Sensor calibration mode"))
        tools_menu.add_command(label="Update Models", command=lambda: messagebox.showinfo("Info", "Model update feature"))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=lambda: messagebox.showinfo("Help", 
            "E-Tongue System User Guide:\n\n"
            "1. Enter sensor readings in the 'Sensor Inputs' tab\n"
            "2. Click 'Analyze Sample' to identify the herb\n"
            "3. View results in 'Analysis Results' tab\n"
            "4. Generate detailed PDF reports\n"
            "5. Save data to database for future reference\n\n"
            "For technical support, contact: support@straw-hats.ai"))
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", 
            "AI-Enabled E-Tongue for Dravya Identification\n\n"
            "Developed for Ministry of Ayush\n"
            "All India Institute of Ayurveda (AIIA)\n\n"
            "Version: 2.0 Enhanced\n"
            "Developed by: Straw-Hats Team\n\n"
            "This system uses advanced ML algorithms to identify\n"
            "Ayurvedic herbs based on electronic tongue sensors\n"
            "and phytochemical analysis."))
        
        root.mainloop()

    if __name__ == "__main__":
        main()
