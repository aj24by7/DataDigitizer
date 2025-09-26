# 📊 Raman Data Digitizer  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)  
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)  

This tool helps you **extract numerical data from graph images**.  
If you have a picture of a line graph (e.g., from a research paper or presentation), this program can convert it back into actual numbers that you can use in **Excel, Google Sheets, or other programs**.  

---

## 🚀 Getting Started  

1. **Run the Program**  
   - Double-click the Python file **or** run it from your command line.  
2. **Load Your Image**  
   - Go to **File → Open Image** and select your graph picture.  

---

## 📝 Step-by-Step Instructions  

### **Step 1: Load and View Your Image**  
- Use the **zoom controls** (+ and -) to adjust the view of your graph.  
- The **preview window** shows you exactly where your mouse is pointing.  

---

### **Step 2: Pick the Line Color**  
1. Click **Pick Color**.  
2. Click directly on the **data line** in your graph.  
3. Adjust the **Tolerance** value if needed:  
   - **Higher = picks up more similar colors.**  

---

### **Step 3: Set Up the Graph Boundaries**  
1. Choose calibration:  
   - **Auto Calibrate** → program detects graph area automatically.  
   - **Manual** → click the four corners of your graph area.  
2. Enter actual axis values:  
   - **X-Axis**: Define left & right edge values (e.g., 0 → 100).  
   - **Y-Axis**: Define top & bottom edge values (e.g., 0 → 50).  

---

### **Step 4: Remove Unwanted Areas (Optional but Recommended)**  
1. **Auto-Exclude Text** → hides labels and numbers.  
2. **Auto-Exclude Borders** → hides graph borders/legends.  
3. **Define Zone** → manually draw boxes around areas to ignore.  

---

### **Step 5: Detect the Lines (for complex graphs)**  
- Choose a detection method:  
  - **Simple Detection** → basic graphs.  
  - **Advanced Detection** → overlapping or multi-line graphs.  
- Use the **< > buttons** to switch between detected lines.  

---

### **Step 6: Extract the Data**  
1. Select extraction method:  
   - **Auto (Best)** – program chooses the best method.  
   - **Method 1 / Method 2** – specific detection methods.  
2. Choose data point settings:  
   - **Number of points** → specify exact number (e.g., 100).  
   - **Step between points** → spacing (e.g., every 10 pixels).  
3. Click **Extract Data**.  

---

### **Step 7: Save Your Results**  
- Click **Export to CSV**.  
- Output file contains two columns:  
  - **X_Value**  
  - **Y_Value**  

---

## 💡 Tips for Best Results  

### **For Better Accuracy**  
- Use **high-resolution, clear images**.  
- Ensure the graph line color is **distinct from the background**.  
- Remove **watermarks/logos** that overlap the graph.  

### **If the Program Has Trouble**  
- Try **Auto-Exclude Text** before extracting.  
- Increase **color tolerance** if the line isn’t detected.  
- Use **Simple Detection** first for easier graphs.  
- Double-check your **calibration points**.  

### **Common Issues**  
- ❌ *No lines detected*: Re-pick the line color or adjust tolerance.  
- ❌ *Too many points*: Lower tolerance or add exclusion zones.  
- ❌ *Missing data*: Increase tolerance or recalibrate.  

---

## ⚙️ Extra Features  
- **Statistics Panel** → Tools → Statistics Panel.  
- **Find Peak/Valley** → automatically detect highs and lows.  
- **Best Fit Line** → toggle trend line on/off.  
- **Zoom Preview** → close-up view of your graph details.  

---

## 📂 File Formats Supported  
- **Input**: PNG, JPG, JPEG, BMP, GIF, TIFF  
- **Output**: CSV (compatible with Excel, Google Sheets, etc.)  

---

## 🆘 Need Help?  
If something isn’t working:  
1. Ensure your image is **clear and high-quality**.  
2. Try **Auto-Exclude Text** before extracting data.  
3. Check that **calibration values** are correct.
4. Start with **Simple Detection** before trying advanced features.

## Additional Info 
The program works best with clean, high-contrast graphs where the data line is clearly visible and distinct from the background.

Add a link to the Discussions tab in your repo and invite users to open issues for bugs/feature requests.

This is also a great place to invite others to contribute in any ways that make sense for your project. Point people to your DEVELOPMENT and/or CONTRIBUTING guides if you have them.
