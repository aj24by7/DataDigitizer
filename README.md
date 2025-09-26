Raman Data Digitizer - User Guide

What This Program Does

This tool helps you extract numerical data from graph images. If you have a picture of a line graph (like from a research paper or presentation), this program can convert it back into actual numbers that you can use in Excel or other programs.

Getting Started

1. Run the Program: Double-click the Python file or run it from your command line
2. Load Your Image: Click "File" → "Open Image" and select your graph picture

Step-by-Step Instructions

Step 1: Load and View Your Image
- Use the zoom controls (+ and -) to get a good view of your graph
- The small preview window shows you exactly where your mouse is pointing

Step 2: Pick the Line Color
1. Click the "Pick Color" button
2. Click directly on the data line in your graph
3. Adjust the "Tolerance" number if needed (higher = picks up more similar colors)

Step 3: Set Up the Graph Boundaries
1. Click "Auto Calibrate" to let the program find the graph area automatically
   - OR click "Manual" and click the four corners of your graph area yourself
2. Enter the actual values for your graph:
   - X: What the left and right edges represent (like 0 to 100)
   - Y: What the top and bottom edges represent (like 0 to 50)

Step 4: Remove Unwanted Areas (Optional but Recommended)
1. Click "Auto-Exclude Text" to automatically hide text labels and numbers
2. Click "Auto-Exclude Borders" to hide graph borders and legends
3. If needed, click "Define Zone" to manually draw boxes around areas you want to ignore

Step 5: Detect the Lines (For Complex Graphs)
- If your graph has multiple lines, use:
  - "Simple Detection" for basic graphs
  - "Advanced Detection" for complex overlapping lines
- Use the arrow buttons (< >) to switch between detected lines

Step 6: Extract the Data
1. Choose your extraction method:
   - "Auto (Best)" - Let the program choose the best method
   - "Method 1" or "Method 2" - Use specific detection methods
2. Set how many data points you want:
   - "Number of points": Specify exactly how many points (like 100)
   - "Step between points": Set the spacing between points (like every 10 pixels)
3. Click "Extract Data"

Step 7: Save Your Results
- Click "Export to CSV" to save your data as a spreadsheet file
- The file will contain two columns: X_Value and Y_Value

Tips for Best Results

For Better Accuracy:
- Use high-resolution, clear images
- Make sure the graph line is a distinct color from the background
- Remove any watermarks or logos that overlap the data

If the Program Has Trouble:
- Try the "Auto-Exclude Text" button first before extracting data
- Increase the color tolerance if the line isn't being detected
- Use "Simple Detection" instead of "Advanced Detection" for basic graphs
- Make sure your calibration points are accurate

Common Issues:
- "No lines detected": Try picking the line color again or adjusting tolerance
- "Too many points": Lower the color tolerance or use exclusion zones
- "Missing data": Increase color tolerance or check your calibration area

Extra Features

- Statistics Panel: Click "Tools" → "Statistics Panel" to see detailed information about your data
- Find Peak/Valley: Automatically find the highest and lowest points in your data
- Best Fit Line: Toggle on/off to see a trend line through your data
- Zoom Preview: The small window shows exactly what you're looking at up close

File Formats Supported

- Input: PNG, JPG, JPEG, BMP, GIF, TIFF image files
- Output: CSV files (can be opened in Excel, Google Sheets, etc.)

Need Help?

If something isn't working:
1. Make sure your image is clear and high quality
2. Try the "Auto-Exclude Text" button before extracting data
3. Check that your calibration values are correct
4. Start with "Simple Detection" before trying advanced features

The program works best with clean, high-contrast graphs where the data line is clearly visible and distinct from the background.

Add a link to the Discussions tab in your repo and invite users to open issues for bugs/feature requests.

This is also a great place to invite others to contribute in any ways that make sense for your project. Point people to your DEVELOPMENT and/or CONTRIBUTING guides if you have them.
