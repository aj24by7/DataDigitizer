import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class CSVComparisonTool:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Comparison Tool - Linear Interpolation & Accuracy")
        self.root.geometry("900x700")
        
        # Data storage
        self.original_data = None
        self.digitized_data = None
        self.interpolated_data = None
        self.results = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="CSV Data Comparison Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # File upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Data Input", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        upload_frame.columnconfigure(0, weight=1)
        upload_frame.columnconfigure(1, weight=1)
        
        # Original data drop zone
        self.original_frame = tk.Frame(upload_frame, bg='lightblue', relief='solid', bd=2)
        self.original_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.original_frame.columnconfigure(0, weight=1)
        self.original_frame.rowconfigure(0, weight=1)
        
        self.original_label = tk.Label(self.original_frame, text="Drop Original CSV Here\n(or click to browse)", 
                                     bg='lightblue', font=('Arial', 12))
        self.original_label.grid(row=0, column=0, padx=20, pady=40)
        self.original_frame.bind("<Button-1>", lambda e: self.browse_file('original'))
        self.original_label.bind("<Button-1>", lambda e: self.browse_file('original'))
        
        # Digitized data drop zone  
        self.digitized_frame = tk.Frame(upload_frame, bg='lightgreen', relief='solid', bd=2)
        self.digitized_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.digitized_frame.columnconfigure(0, weight=1)
        self.digitized_frame.rowconfigure(0, weight=1)
        
        self.digitized_label = tk.Label(self.digitized_frame, text="Drop Digitized CSV Here\n(or click to browse)", 
                                       bg='lightgreen', font=('Arial', 12))
        self.digitized_label.grid(row=0, column=0, padx=20, pady=40)
        self.digitized_frame.bind("<Button-1>", lambda e: self.browse_file('digitized'))
        self.digitized_label.bind("<Button-1>", lambda e: self.browse_file('digitized'))
        
        # Configure drag and drop
        self.original_frame.drop_target_register(DND_FILES)
        self.original_frame.dnd_bind('<<Drop>>', lambda e: self.drop_file(e, 'original'))
        
        self.digitized_frame.drop_target_register(DND_FILES)
        self.digitized_frame.dnd_bind('<<Drop>>', lambda e: self.drop_file(e, 'digitized'))
        
        # Column selection
        column_frame = ttk.LabelFrame(main_frame, text="Column Selection", padding="10")
        column_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(column_frame, text="X Column:").grid(row=0, column=0, padx=(0, 5))
        self.x_col_var = tk.StringVar(value="x")
        self.x_col_entry = ttk.Entry(column_frame, textvariable=self.x_col_var, width=15)
        self.x_col_entry.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(column_frame, text="Y Column:").grid(row=0, column=2, padx=(0, 5))
        self.y_col_var = tk.StringVar(value="y")
        self.y_col_entry = ttk.Entry(column_frame, textvariable=self.y_col_var, width=15)
        self.y_col_entry.grid(row=0, column=3, padx=(0, 20))
        
        # Process button
        self.process_btn = ttk.Button(column_frame, text="Process & Compare", 
                                    command=self.process_data, state='disabled')
        self.process_btn.grid(row=0, column=4, padx=20)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Plot frame
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
    def browse_file(self, file_type):
        filename = filedialog.askopenfilename(
            title=f"Select {file_type} CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.load_file(filename, file_type)
    
    def drop_file(self, event, file_type):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_file(files[0], file_type)
    
    def load_file(self, filepath, file_type):
        try:
            df = pd.read_csv(filepath)
            
            if file_type == 'original':
                self.original_data = df
                self.original_label.config(text=f"Original Data Loaded\n{len(df)} rows\nColumns: {', '.join(df.columns)}")
            else:
                self.digitized_data = df  
                self.digitized_label.config(text=f"Digitized Data Loaded\n{len(df)} rows\nColumns: {', '.join(df.columns)}")
            
            # Enable process button if both files loaded
            if self.original_data is not None and self.digitized_data is not None:
                self.process_btn.config(state='normal')
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {filepath}:\n{str(e)}")
    
    def linear_interpolate(self, x_target, x_data, y_data):
        """Linear interpolation for a single x_target value"""
        # Sort data by x values
        sorted_indices = np.argsort(x_data)
        x_sorted = x_data[sorted_indices]
        y_sorted = y_data[sorted_indices]
        
        # Handle edge cases
        if x_target <= x_sorted[0]:
            return y_sorted[0]
        if x_target >= x_sorted[-1]:
            return y_sorted[-1]
            
        # Find the two points to interpolate between
        for i in range(len(x_sorted) - 1):
            if x_sorted[i] <= x_target <= x_sorted[i + 1]:
                # Linear interpolation formula: y = y1 + (y2-y1) * (x-x1)/(x2-x1)
                x1, x2 = x_sorted[i], x_sorted[i + 1]
                y1, y2 = y_sorted[i], y_sorted[i + 1]
                
                if x2 == x1:  # Avoid division by zero
                    return y1
                    
                interpolated_y = y1 + (y2 - y1) * (x_target - x1) / (x2 - x1)
                return interpolated_y
        
        return np.nan
    
    def process_data(self):
        try:
            x_col = self.x_col_var.get()
            y_col = self.y_col_var.get()
            
            # Validate columns exist
            if x_col not in self.original_data.columns or y_col not in self.original_data.columns:
                messagebox.showerror("Error", f"Columns '{x_col}' or '{y_col}' not found in original data")
                return
            
            if x_col not in self.digitized_data.columns or y_col not in self.digitized_data.columns:
                messagebox.showerror("Error", f"Columns '{x_col}' or '{y_col}' not found in digitized data")
                return
            
            # Extract data
            orig_x = self.original_data[x_col].values
            orig_y = self.original_data[y_col].values
            
            dig_x = self.digitized_data[x_col].values
            dig_y = self.digitized_data[y_col].values
            
            # Remove any NaN values
            orig_mask = ~(np.isnan(orig_x) | np.isnan(orig_y))
            orig_x, orig_y = orig_x[orig_mask], orig_y[orig_mask]
            
            dig_mask = ~(np.isnan(dig_x) | np.isnan(dig_y))
            dig_x, dig_y = dig_x[dig_mask], dig_y[dig_mask]
            
            # Interpolate digitized data to match original x values
            interpolated_y = []
            for x_target in orig_x:
                interp_y = self.linear_interpolate(x_target, dig_x, dig_y)
                interpolated_y.append(interp_y)
            
            interpolated_y = np.array(interpolated_y)
            
            # Remove any NaN results from interpolation
            valid_mask = ~np.isnan(interpolated_y)
            orig_y_clean = orig_y[valid_mask]
            interpolated_y_clean = interpolated_y[valid_mask]
            orig_x_clean = orig_x[valid_mask]
            
            if len(orig_y_clean) == 0:
                messagebox.showerror("Error", "No valid interpolation results. Check your data ranges.")
                return
            
            # Calculate accuracy metrics
            mse = np.mean((orig_y_clean - interpolated_y_clean) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(orig_y_clean - interpolated_y_clean))
            
            # R-squared
            ss_res = np.sum((orig_y_clean - interpolated_y_clean) ** 2)
            ss_tot = np.sum((orig_y_clean - np.mean(orig_y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Correlation coefficient
            correlation = np.corrcoef(orig_y_clean, interpolated_y_clean)[0, 1]
            
            # Mean absolute percentage error
            mape = np.mean(np.abs((orig_y_clean - interpolated_y_clean) / orig_y_clean)) * 100
            
            # Store results
            self.results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'correlation': correlation,
                'mape': mape,
                'n_points': len(orig_y_clean),
                'orig_x': orig_x_clean,
                'orig_y': orig_y_clean,
                'interp_y': interpolated_y_clean,
                'dig_x': dig_x,
                'dig_y': dig_y
            }
            
            self.display_results()
            self.create_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    
    def display_results(self):
        results_str = f"""ACCURACY METRICS:
========================
Mean Squared Error (MSE): {self.results['mse']:.6f}
Root Mean Squared Error (RMSE): {self.results['rmse']:.6f}
Mean Absolute Error (MAE): {self.results['mae']:.6f}
R-squared: {self.results['r_squared']:.6f}
Correlation Coefficient: {self.results['correlation']:.6f}
Mean Absolute Percentage Error (MAPE): {self.results['mape']:.2f}%

SUMMARY:
========================
Total comparison points: {self.results['n_points']}
Original data points: {len(self.original_data)}
Digitized data points: {len(self.digitized_data)}

INTERPRETATION:
========================
• Lower MSE/RMSE/MAE values indicate better accuracy
• R-squared closer to 1.0 indicates better fit
• Correlation closer to 1.0 indicates better linear relationship
• MAPE shows percentage error (lower is better)
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_str)
    
    def create_plot(self):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Original vs Interpolated comparison
        ax1.scatter(self.results['orig_x'], self.results['orig_y'], 
                   alpha=0.6, label='Original', color='blue', s=20)
        ax1.scatter(self.results['orig_x'], self.results['interp_y'], 
                   alpha=0.6, label='Interpolated', color='red', s=20)
        ax1.plot(self.results['dig_x'], self.results['dig_y'], 
                'g-', alpha=0.5, label='Digitized (raw)', linewidth=1)
        ax1.set_xlabel('X values')
        ax1.set_ylabel('Y values')
        ax1.set_title('Data Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = self.results['orig_y'] - self.results['interp_y']
        ax2.scatter(self.results['orig_x'], residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X values')
        ax2.set_ylabel('Residuals (Original - Interpolated)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add plot to tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = TkinterDnD.Tk()
    app = CSVComparisonTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()