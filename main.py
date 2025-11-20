import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import cv2
import pandas as pd
from scipy import stats
from scipy import ndimage
import json
import math

class LineMaskingSystem:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.detected_lines = []  # List of line masks
        self.current_line_index = 0
        self.line_colors = []  # Colors for visualization
        self.line_previews = []  # Thumbnail previews
        self.detection_parameters = {
            'min_line_length': 50,
            'max_line_gap': 10,
            'thickness_tolerance': 5,
            'clustering_eps': 15,
            'min_samples': 10
        }
        
    def detect_all_lines(self):
        """
        Advanced line detection using multiple computer vision techniques.
        Detects and separates individual spectral lines in complex multi-line plots.
        Now respects exclusion zones to avoid detecting text and labels.
        """
        if self.parent.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            # Update debug display
            self.parent.debug_label.config(text="Debug: Starting detection...")
            self.parent.root.update_idletasks()
            
            print("Starting advanced line detection with exclusion zones...")
            
            # Get parameters from UI
            edge_thresh = int(self.parent.edge_threshold_var.get())
            min_length = int(self.parent.min_length_var.get())
            cluster_dist = int(self.parent.cluster_dist_var.get())
            min_points = int(self.parent.min_points_var.get())
            
            # Update detection parameters
            self.detection_parameters.update({
                'min_line_length': min_length,
                'clustering_eps': cluster_dist,
                'min_samples': min_points
            })
            
            # Convert to working formats
            img_array = np.array(self.parent.image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # NEW: Create exclusion mask to avoid detecting text/labels
            exclusion_mask = self._create_exclusion_mask(gray.shape)
            if exclusion_mask is not None:
                self.parent.debug_label.config(text=f"Debug: Applying {len(self.parent.exclusion_zones)} exclusion zones...")
                self.parent.root.update_idletasks()
                print(f"Applying {len(self.parent.exclusion_zones)} exclusion zones to line detection")
            
            # Clear previous results
            self.detected_lines.clear()
            self.line_colors.clear()
            self.line_previews.clear()
            
            # Method 1: Edge-based line detection with Hough Transform
            self.parent.debug_label.config(text="Debug: Running Hough detection...")
            self.parent.root.update_idletasks()
            lines_hough = self._detect_lines_hough(gray, edge_thresh, exclusion_mask)
            print(f"Hough detection found {len(lines_hough)} points")
            
            # Method 2: Contour-based line detection
            self.parent.debug_label.config(text="Debug: Running contour detection...")
            self.parent.root.update_idletasks()
            lines_contour = self._detect_lines_contour(gray, exclusion_mask)
            print(f"Contour detection found {len(lines_contour)} points")
            
            # Method 3: Skeletonization approach for complex overlapping lines
            self.parent.debug_label.config(text="Debug: Running skeleton detection...")
            self.parent.root.update_idletasks()
            lines_skeleton = self._detect_lines_skeleton(gray, exclusion_mask)
            print(f"Skeleton detection found {len(lines_skeleton)} points")
            
            # Combine and separate overlapping detections
            all_line_points = lines_hough + lines_contour + lines_skeleton
            print(f"Total points before clustering: {len(all_line_points)}")
            
            if not all_line_points:
                self.parent.debug_label.config(text="Debug: No points found. Try adjusting parameters.")
                messagebox.showinfo("No Lines", "No distinct lines detected. Try:\n1. Lower edge threshold\n2. Smaller min length\n3. Use Simple Detection button\n4. First run Auto-Exclude Text if not done")
                return
            
            # Cluster points into separate lines using custom clustering
            self.parent.debug_label.config(text="Debug: Clustering points into lines...")
            self.parent.root.update_idletasks()
            separated_lines = self._separate_lines_clustering(all_line_points)
            print(f"Clustering produced {len(separated_lines)} lines")
            
            if not separated_lines:
                self.parent.debug_label.config(text="Debug: Clustering failed. Try larger cluster distance.")
                messagebox.showinfo("No Lines", "Clustering failed to separate lines. Try:\n1. Increase cluster distance\n2. Decrease min points\n3. Use Simple Detection")
                return
            
            # Create masks for each detected line
            self.parent.debug_label.config(text="Debug: Creating line masks...")
            self.parent.root.update_idletasks()
            self._create_line_masks(separated_lines, gray.shape)
            
            # Generate colors and previews for UI
            self._generate_line_visualization()
            
            self.parent.debug_label.config(text=f"Debug: Success! Found {len(self.detected_lines)} lines")
            print(f"Successfully detected {len(self.detected_lines)} distinct lines")
            messagebox.showinfo("Line Detection Complete", 
                              f"Detected {len(self.detected_lines)} distinct lines.\n"
                              f"Use the controls to cycle through and select lines for extraction.")
            
        except Exception as e:
            self.parent.debug_label.config(text=f"Debug: Error - {str(e)[:50]}...")
            messagebox.showerror("Line Detection Error", f"Failed to detect lines: {e}")
            print(f"Line detection error: {e}")
    
    def _create_exclusion_mask(self, img_shape):
        """Create a mask that excludes regions defined in exclusion_zones."""
        if not self.parent.exclusion_zones:
            return None
        
        # Create mask where excluded regions are 0, allowed regions are 255
        mask = np.ones(img_shape, dtype=np.uint8) * 255
        
        for (x0, y0, x1, y1) in self.parent.exclusion_zones:
            # Ensure coordinates are within image bounds
            x0 = max(0, min(x0, img_shape[1]-1))
            y0 = max(0, min(y0, img_shape[0]-1))
            x1 = max(0, min(x1, img_shape[1]-1))
            y1 = max(0, min(y1, img_shape[0]-1))
            
            # Set excluded region to 0
            mask[y0:y1+1, x0:x1+1] = 0
        
        return mask
    
    def _detect_lines_hough(self, gray, edge_thresh=30, exclusion_mask=None):
        """Use Hough Line Transform to detect straight line segments, respecting exclusion zones."""
        # Enhanced edge detection with adjustable threshold
        edges = cv2.Canny(gray, edge_thresh, edge_thresh * 3, apertureSize=3)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            edges = cv2.bitwise_and(edges, exclusion_mask)
        
        # Apply HoughLinesP for line segment detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=max(15, edge_thresh//2),
                               minLineLength=self.detection_parameters['min_line_length'],
                               maxLineGap=self.detection_parameters['max_line_gap'])
        
        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Sample points along each detected line
                num_points = max(5, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 3))
                x_points = np.linspace(x1, x2, num_points, dtype=int)
                y_points = np.linspace(y1, y2, num_points, dtype=int)
                
                for x, y in zip(x_points, y_points):
                    if (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0] and
                        (exclusion_mask is None or exclusion_mask[y, x] > 0)):
                        line_points.append((x, y))
        
        return line_points
    
    def _detect_lines_contour(self, gray, exclusion_mask=None):
        """Use contour detection to find curved and irregular lines, respecting exclusion zones."""
        # Adaptive thresholding to handle varying intensities
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            binary = cv2.bitwise_and(binary, exclusion_mask)
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_points = []
        for contour in contours:
            # Filter contours based on area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Look for line-like shapes (high aspect ratio or reasonable area)
            if aspect_ratio > 3 or (area > 200 and aspect_ratio > 1.5):
                # Sample points along the contour
                contour_points = contour.reshape(-1, 2)
                for point in contour_points[::2]:  # Sample every other point
                    x, y = tuple(point)
                    if (exclusion_mask is None or 
                        (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[y, x] > 0)):
                        line_points.append((x, y))
        
        return line_points
    
    def _detect_lines_skeleton(self, gray, exclusion_mask=None):
        """Use morphological skeletonization to detect line centers, respecting exclusion zones."""
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            binary = cv2.bitwise_and(binary, exclusion_mask)
        
        # Skeletonization using morphological operations
        skeleton = np.zeros_like(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        while True:
            eroded = cv2.erode(binary, kernel)
            dilated = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary, dilated))
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
        
        # Extract skeleton points
        line_points = []
        y_coords, x_coords = np.where(skeleton > 0)
        for x, y in zip(x_coords, y_coords):
            if (exclusion_mask is None or 
                (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[y, x] > 0)):
                line_points.append((x, y))
        
        return line_points
    
    def _separate_lines_clustering(self, all_points):
        """Use simple distance-based clustering to separate individual lines."""
        if not all_points:
            return []
        
        # Convert to numpy array for easier processing
        points_array = np.array(all_points)
        
        # Simple distance-based clustering
        eps = self.detection_parameters['clustering_eps']
        min_samples = self.detection_parameters['min_samples']
        
        clusters = []
        visited = np.zeros(len(points_array), dtype=bool)
        
        for i, point in enumerate(points_array):
            if visited[i]:
                continue
                
            # Find all points within eps distance
            distances = np.sqrt(np.sum((points_array - point) ** 2, axis=1))
            neighbors = np.where(distances <= eps)[0]
            
            if len(neighbors) < min_samples:
                continue  # Skip noise points
            
            # Create new cluster
            cluster = []
            queue = list(neighbors)
            
            while queue:
                current_idx = queue.pop(0)
                if visited[current_idx]:
                    continue
                    
                visited[current_idx] = True
                cluster.append(points_array[current_idx])
                
                # Find neighbors of current point
                current_point = points_array[current_idx]
                distances = np.sqrt(np.sum((points_array - current_point) ** 2, axis=1))
                current_neighbors = np.where(distances <= eps)[0]
                
                # Add unvisited neighbors to queue
                for neighbor_idx in current_neighbors:
                    if not visited[neighbor_idx]:
                        queue.append(neighbor_idx)
            
            if len(cluster) >= min_samples:
                clusters.append(np.array(cluster))
        
        # Further separate by Y-coordinate proximity (for horizontal lines)
        separated_lines = []
        for cluster_points in clusters:
            if len(cluster_points) == 0:
                continue
                
            # Sort by Y coordinate
            y_sorted = cluster_points[cluster_points[:, 1].argsort()]
            
            # Split into sub-lines if there are large Y gaps
            sub_lines = []
            current_line = [y_sorted[0]]
            
            for i in range(1, len(y_sorted)):
                if abs(y_sorted[i][1] - y_sorted[i-1][1]) < 20:  # Same line
                    current_line.append(y_sorted[i])
                else:  # New line
                    if len(current_line) > 20:  # Minimum points for a valid line
                        sub_lines.append(np.array(current_line))
                    current_line = [y_sorted[i]]
            
            if len(current_line) > 20:
                sub_lines.append(np.array(current_line))
            
            separated_lines.extend(sub_lines)
        
        return separated_lines
    
    def _create_line_masks(self, separated_lines, img_shape):
        """Create binary masks for each detected line."""
        for i, line_points in enumerate(separated_lines):
            if len(line_points) < 10:
                continue
            
            # Create binary mask
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            # Draw thick line through all points
            for point in line_points:
                x, y = point
                if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                    # Draw a small circle at each point to create thickness
                    cv2.circle(mask, (int(x), int(y)), 
                             self.detection_parameters['thickness_tolerance'], 255, -1)
            
            # Smooth the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            self.detected_lines.append(mask)
    
    def _generate_line_visualization(self):
        """Generate colors and preview thumbnails for each detected line."""
        # Generate distinct colors for each line
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, mask in enumerate(self.detected_lines):
            # Assign color
            color = colors[i % len(colors)]
            self.line_colors.append(color)
            
            # Create preview thumbnail
            try:
                preview = self._create_line_preview(mask, color)
                self.line_previews.append(preview)
            except Exception as e:
                print(f"Failed to create preview for line {i}: {e}")
                # Add a None placeholder so indices stay aligned
                self.line_previews.append(None)
    
    def _create_line_preview(self, mask, color):
        """Create a small preview image of the line for UI display."""
        # Find bounding box of the line
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return None
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Ensure we have a reasonable bounding box
        if max_x - min_x < 5 or max_y - min_y < 2:
            # Create a simple line preview if bounding box is too small
            preview_rgb = np.zeros((30, 100, 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            rgb_color = color_map.get(color, (255,255,255))
            
            # Draw a simple horizontal line
            preview_rgb[12:18, 10:90] = rgb_color
            return Image.fromarray(preview_rgb)
        
        # Extract region and resize for preview
        try:
            line_region = mask[min_y:max_y+1, min_x:max_x+1]
            
            # Resize to thumbnail size
            thumbnail_size = (100, 30)
            line_region_resized = cv2.resize(line_region, thumbnail_size)
            
            # Convert to RGB and apply color
            preview_rgb = np.zeros((*thumbnail_size[::-1], 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            
            rgb_color = color_map.get(color, (255,255,255))
            for c in range(3):
                preview_rgb[:,:,c] = (line_region_resized / 255) * rgb_color[c]
            
            return Image.fromarray(preview_rgb)
        except Exception as e:
            print(f"Error creating detailed preview: {e}")
            # Fallback to simple line preview
            preview_rgb = np.zeros((30, 100, 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            rgb_color = color_map.get(color, (255,255,255))
            preview_rgb[12:18, 10:90] = rgb_color
            return Image.fromarray(preview_rgb)
    
    def get_current_line_mask(self):
        """Get the currently selected line mask."""
        if not self.detected_lines or self.current_line_index >= len(self.detected_lines):
            return None
        return self.detected_lines[self.current_line_index]
    
    def next_line(self):
        """Switch to the next detected line."""
        if self.detected_lines:
            self.current_line_index = (self.current_line_index + 1) % len(self.detected_lines)
            return True
        return False
    
    def previous_line(self):
        """Switch to the previous detected line."""
        if self.detected_lines:
            self.current_line_index = (self.current_line_index - 1) % len(self.detected_lines)
            return True
        return False
    
    def set_line_index(self, index):
        """Set the current line index directly."""
        if 0 <= index < len(self.detected_lines):
            self.current_line_index = index
            return True
        return False

class MovableStatisticsPanel:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.window = None
        self.is_docked = False
        self.dock_zone_active = False
        self.drag_data = {"x": 0, "y": 0}
        self.dock_indicator = None
        self.dock_tab = None
        self.is_closed = True

    def create_dock_tab(self):
        """Create a small docking tab on the canvas."""
        if self.dock_tab or not hasattr(self.parent, 'canvas'):
            return
            
        try:
            canvas = self.parent.canvas
            # Create a small tab on the left edge
            tab_width, tab_height = 80, 25
            tab_x, tab_y = 10, 100
            
            # Create tab background
            self.dock_tab = canvas.create_rectangle(
                tab_x, tab_y, tab_x + tab_width, tab_y + tab_height,
                fill='#4D6BFE', outline='white', width=2, tags="dock_tab"
            )
            
            # Create tab text
            canvas.create_text(
                tab_x + tab_width//2, tab_y + tab_height//2,
                text="📊 Stats", fill='white', font=('Arial', 9, 'bold'),
                tags="dock_tab"
            )
            
            # Bind click event to show panel
            canvas.tag_bind("dock_tab", "<Button-1>", lambda e: self.show_panel())
            canvas.tag_bind("dock_tab", "<Enter>", self.on_tab_hover)
            canvas.tag_bind("dock_tab", "<Leave>", self.on_tab_leave)
        except Exception as e:
            print(f"Error creating dock tab: {e}")
    
    def remove_dock_tab(self):
        """Remove the docking tab."""
        try:
            if self.dock_tab and hasattr(self.parent, 'canvas'):
                self.parent.canvas.delete("dock_tab")
                self.dock_tab = None
        except Exception as e:
            print(f"Error removing dock tab: {e}")
    
    def on_tab_hover(self, event):
        """Handle tab hover effect."""
        try:
            if hasattr(self.parent, 'canvas'):
                self.parent.canvas.itemconfig("dock_tab", fill='#6C7BFE')
        except Exception as e:
            print(f"Error in tab hover: {e}")
    
    def on_tab_leave(self, event):
        """Handle tab leave effect."""
        try:
            if hasattr(self.parent, 'canvas'):
                self.parent.canvas.itemconfig("dock_tab", fill='#4D6BFE')
        except Exception as e:
            print(f"Error in tab leave: {e}")
        
    def show_panel(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return
        
        self.is_closed = False
        self.remove_dock_tab()  # Remove tab when panel is open
            
        # Create floating statistics window
        self.window = tk.Toplevel(self.parent.root)
        self.window.title("Statistics")
        self.window.geometry("320x450")
        self.window.configure(bg='#f0f0f0')
        self.window.attributes('-alpha', 0.95)  # Slight transparency
        
        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.hide_panel)
        
        # Make it stay on top but not always
        self.window.attributes('-topmost', False)
        
        # Custom title bar for dragging
        title_bar = tk.Frame(self.window, bg='#4D6BFE', height=30)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        
        title_label = tk.Label(title_bar, text="📊 Statistics", bg='#4D6BFE', fg='white', 
                              font=('Arial', 10, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Close button
        close_btn = tk.Button(title_bar, text="✕", bg='#DC3545', fg='white', 
                             font=('Arial', 10, 'bold'), bd=0, width=3,
                             command=self.hide_panel)
        close_btn.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Dock button
        self.dock_btn = tk.Button(title_bar, text="📌", bg='#28a745', fg='white', 
                                 font=('Arial', 10, 'bold'), bd=0, width=3,
                                 command=self.toggle_dock)
        self.dock_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        
        # Statistics content
        content_frame = tk.Frame(self.window, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(content_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(text_frame, bg='white', wrap=tk.WORD, 
                                 font=('Courier New', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind dragging events
        title_bar.bind("<Button-1>", self.start_drag)
        title_bar.bind("<B1-Motion>", self.on_drag)
        title_bar.bind("<ButtonRelease-1>", self.end_drag)
        title_label.bind("<Button-1>", self.start_drag)
        title_label.bind("<B1-Motion>", self.on_drag)
        title_label.bind("<ButtonRelease-1>", self.end_drag)
        
        # Update content
        self.update_content()
        
        # Position near parent window
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        self.window.geometry(f"320x450+{parent_x + 50}+{parent_y + 50}")
        
    def hide_panel(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.is_docked = False
            self.is_closed = True
            self.remove_dock_indicator()
            self.create_dock_tab()  # Create tab when panel is closed
            
    def start_drag(self, event):
        self.drag_data["x"] = event.x_root
        self.drag_data["y"] = event.y_root
        
    def on_drag(self, event):
        if not self.window:
            return
            
        # Calculate movement
        dx = event.x_root - self.drag_data["x"]
        dy = event.y_root - self.drag_data["y"]
        
        # Move window
        x = self.window.winfo_x() + dx
        y = self.window.winfo_y() + dy
        self.window.geometry(f"+{x}+{y}")
        
        # Check for dock zone (left side of parent window)
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        parent_width = self.parent.root.winfo_width()
        parent_height = self.parent.root.winfo_height()
        
        dock_zone_x = parent_x + 10
        dock_zone_width = 100
        
        if (dock_zone_x <= event.x_root <= dock_zone_x + dock_zone_width and
            parent_y <= event.y_root <= parent_y + parent_height):
            self.show_dock_indicator()
        else:
            self.remove_dock_indicator()
        
        self.drag_data["x"] = event.x_root
        self.drag_data["y"] = event.y_root
        
    def end_drag(self, event):
        # Check if we should dock
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        parent_height = self.parent.root.winfo_height()
        
        dock_zone_x = parent_x + 10
        dock_zone_width = 100
        
        if (dock_zone_x <= event.x_root <= dock_zone_x + dock_zone_width and
            parent_y <= event.y_root <= parent_y + parent_height):
            self.dock_to_parent()
        
        self.remove_dock_indicator()
        
    def show_dock_indicator(self):
        if not self.dock_indicator and self.parent.root.winfo_exists():
            # Create a visual indicator on the main canvas
            canvas = self.parent.canvas
            canvas_width = canvas.winfo_width()
            
            self.dock_indicator = canvas.create_rectangle(
                5, 5, 325, canvas.winfo_height() - 5,
                outline='#4D6BFE', width=3, dash=(10, 5),
                tags="dock_indicator"
            )
            
    def remove_dock_indicator(self):
        if self.dock_indicator:
            self.parent.canvas.delete("dock_indicator")
            self.dock_indicator = None
            
    def dock_to_parent(self):
        if not self.window:
            return
            
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        
        # Position at left side of parent
        self.window.geometry(f"320x450+{parent_x + 10}+{parent_y + 80}")
        self.is_docked = True
        self.dock_btn.config(text="📌", bg='#ffc107')  # Change icon when docked
        
    def toggle_dock(self):
        if self.is_docked:
            # Undock - move away from parent
            parent_x = self.parent.root.winfo_x()
            parent_y = self.parent.root.winfo_y()
            self.window.geometry(f"320x450+{parent_x + 400}+{parent_y + 100}")
            self.is_docked = False
            self.dock_btn.config(text="📌", bg='#28a745')
        else:
            self.dock_to_parent()
            
    def update_content(self):
        if not self.window or not self.window.winfo_exists() or not hasattr(self, 'stats_text'):
            return
            
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        if not self.parent.statistics:
            self.stats_text.insert(tk.END, "No data available for statistics.\n\nExtract data first to see:\n• Point counts\n• Statistical measures\n• Best fit analysis\n• Extrema detection")
            self.stats_text.config(state=tk.DISABLED)
            return
            
        stats_str = "--- Line Masking Results ---\n"
        if hasattr(self.parent, 'line_masking') and self.parent.line_masking.detected_lines:
            stats_str += f"Lines Detected: {len(self.parent.line_masking.detected_lines)}\n"
            stats_str += f"Current Line: {self.parent.line_masking.current_line_index + 1}\n\n"
        
        stats_str += "--- Method Results ---\n"
        stats_str += f"Method 1: {len(self.parent.method1_points)} points\n"
        stats_str += f"Method 2: {len(self.parent.method2_points)} points\n"
        stats_str += f"Current: {self.parent.method_var.get()}\n\n"
        
        stats_str += "--- General ---\n"
        stats_str += f"{'Points Found:':<15} {self.parent.statistics['Points']}\n"
        stats_str += f"{'Correlation:':<15} {self.parent.statistics.get('Correlation', 0):.4f}\n"
        stats_str += f"{'Area Under Curve:':<15} {self.parent.statistics.get('Area (Trapezoid)', 0):.4f}\n\n"
        
        if 'Peak' in self.parent.statistics:
            stats_str += "--- Extrema ---\n"
            peak = self.parent.statistics['Peak']
            valley = self.parent.statistics['Valley']
            stats_str += f"{'Peak (x, y):':<15} ({peak[0]:.2f}, {peak[1]:.2f})\n"
            stats_str += f"{'Valley (x, y):':<15} ({valley[0]:.2f}, {valley[1]:.2f})\n\n"
            
        stats_str += "--- X-Axis Stats ---\n"
        stats_str += f"{'Mean:':<15} {self.parent.statistics.get('X Mean', 0):.4f}\n"
        stats_str += f"{'Std Dev:':<15} {self.parent.statistics.get('X Std Dev', 0):.4f}\n"
        stats_str += f"{'Skewness:':<15} {self.parent.statistics.get('X Skew', 0):.4f}\n"
        stats_str += f"{'Kurtosis:':<15} {self.parent.statistics.get('X Kurtosis', 0):.4f}\n\n"
        
        stats_str += "--- Y-Axis Stats ---\n"
        stats_str += f"{'Mean:':<15} {self.parent.statistics.get('Y Mean', 0):.4f}\n"
        stats_str += f"{'Std Dev:':<15} {self.parent.statistics.get('Y Std Dev', 0):.4f}\n"
        stats_str += f"{'Skewness:':<15} {self.parent.statistics.get('Y Skew', 0):.4f}\n"
        stats_str += f"{'Kurtosis:':<15} {self.parent.statistics.get('Y Kurtosis', 0):.4f}\n\n"
        
        if self.parent.best_fit_equation:
            stats_str += "--- Best Fit Line ---\n"
            stats_str += f"{'Equation:':<15} {self.parent.best_fit_equation}\n"
            stats_str += f"{'R-squared:':<15} {self.parent.statistics.get('r_squared', 0):.6f}\n"
            
        self.stats_text.insert(1.0, stats_str)
        self.stats_text.config(state=tk.DISABLED)

class RamanDataDigitizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectra Data Digitizer")
        self.root.geometry("1400x900")
        
        # Modern Color Scheme
        self.colors = {
            'bg_main': '#fafbfc',           # Very light gray background
            'bg_panel': '#ffffff',          # Pure white for panels
            'bg_secondary': '#f8f9fa',      # Slightly darker light gray
            'primary': '#4f46e5',           # Modern indigo
            'primary_hover': '#4338ca',     # Darker indigo for hover
            'success': '#10b981',           # Modern green
            'warning': '#f59e0b',           # Modern amber
            'danger': '#ef4444',            # Modern red
            'info': '#06b6d4',              # Modern cyan
            'secondary': '#6b7280',         # Modern gray
            'text_primary': '#111827',      # Dark gray for text
            'text_secondary': '#6b7280',    # Medium gray for secondary text
            'border': '#e5e7eb',            # Light border
            'accent': '#8b5cf6'             # Purple accent
        }
        
        self.root.configure(bg=self.colors['bg_main'])

        # --- State Variables ---
        self.image = None
        self.image_array = None
        self.photo = None
        self.original_image = None  # Store original for zoom operations
        self.zoom_level = 1.0  # Track current zoom level
        self.calibration_points = []
        self.selected_color = None
        self.color_tolerance = 0
        self.extracted_points = []
        self.real_coordinates = []
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_mode = "normal"
        self.statistics = {}
        self.best_fit_equation = ""
        self.best_fit_line_pixel_coords = None
        
        # --- Interactive State ---
        self.exclusion_zones = []
        self.temp_rect_start = None
        self.temp_rect_id = None
        self.dragging_line_index = None # 0:top, 1:right, 2:bottom, 3:left

        # --- Detection Method Results ---
        self.method1_points = []  # Original method (main4)
        self.method2_points = []  # New method (from main3)
        self.current_method = "auto"  # "auto", "method1", "method2"

        # --- Line Masking System - NEW FEATURE ---
        self.line_masking = LineMaskingSystem(self)

        # --- Statistics Panel - Initialize after other components ---
        self.stats_panel = MovableStatisticsPanel(self)

        self.setup_ui()
        self.setup_bindings()
        
        # Initialize dock tab after a delay to ensure UI is ready
        self.root.after(500, self.init_dock_tab)

    def init_dock_tab(self):
        """Initialize the dock tab after UI is ready."""
        try:
            if hasattr(self, 'stats_panel') and self.stats_panel.is_closed:
                self.stats_panel.create_dock_tab()
        except Exception as e:
            print(f"Error initializing dock tab: {e}")

    def setup_ui(self):
        """Sets up the main user interface layout and widgets."""
        # --- Main Menu ---
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.load_image)
        file_menu.add_command(label="Load JSON...", command=self.load_json)
        file_menu.add_command(label="Paste from Clipboard", command=self.paste_image)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All", command=self.clear_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools Menu
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Statistics Panel", command=self.stats_panel.show_panel)

        # --- Main Frame ---
        main_frame = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # --- Left Panel (Image) ---
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_panel'], relief=tk.FLAT, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        
        canvas_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(canvas_frame, bg='#f1f3f4', cursor='arrow', highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # --- Right Panel (Controls) - Now with scrollbar ---
        right_outer_frame = tk.Frame(main_frame, bg=self.colors['bg_main'], width=360)
        right_outer_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_outer_frame.pack_propagate(False)
        
        # Create scrollable frame
        canvas_right = tk.Canvas(right_outer_frame, bg=self.colors['bg_main'], width=360, highlightthickness=0)
        scrollbar_right = ttk.Scrollbar(right_outer_frame, orient="vertical", command=canvas_right.yview)
        scrollable_frame = tk.Frame(canvas_right, bg=self.colors['bg_main'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all"))
        )
        
        canvas_right.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_right.configure(yscrollcommand=scrollbar_right.set)
        
        canvas_right.pack(side="left", fill="both", expand=True)
        scrollbar_right.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas_right.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_right.bind("<MouseWheel>", _on_mousewheel)

        # Now use scrollable_frame as the right panel
        right_panel = scrollable_frame

        # ========== STEP 1: ZOOM PREVIEW AND CONTROLS ==========
        zoom_outer_frame = tk.LabelFrame(right_panel, text="🔍 Step 1: Zoom Preview", 
                                       bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                       font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        zoom_outer_frame.pack(fill=tk.X, pady=(0, 8), padx=6)
        
        # Zoom controls
        zoom_controls = tk.Frame(zoom_outer_frame, bg=self.colors['bg_panel'])
        zoom_controls.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Button(zoom_controls, text="−", command=self.zoom_out, 
                 bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'), 
                 width=2, relief=tk.FLAT, bd=0).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_controls, text="+", command=self.zoom_in, 
                 bg=self.colors['success'], fg='white', font=('Segoe UI', 8, 'bold'), 
                 width=2, relief=tk.FLAT, bd=0).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_controls, text="1:1", command=self.zoom_reset, 
                 bg=self.colors['secondary'], fg='white', font=('Segoe UI', 7, 'bold'), 
                 width=3, relief=tk.FLAT, bd=0).pack(side=tk.LEFT, padx=2)
        
        # Zoom level display
        self.zoom_label = tk.Label(zoom_controls, text="100%", 
                                  bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                                  font=('Segoe UI', 8))
        self.zoom_label.pack(side=tk.RIGHT, padx=8)
        
        # Zoom canvas
        self.zoom_canvas = tk.Canvas(zoom_outer_frame, width=120, height=80, 
                                   bg=self.colors['bg_secondary'], highlightthickness=1,
                                   highlightbackground=self.colors['border'])
        self.zoom_canvas.pack(pady=6, padx=8)

        # ========== STEP 2: COLOR SELECTION ==========
        color_frame = tk.LabelFrame(right_panel, text="🎨 Step 2: Color Selection", 
                                  bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        color_frame.pack(fill=tk.X, pady=(0, 8), padx=6)
        
        color_btn_frame = tk.Frame(color_frame, bg=self.colors['bg_panel'])
        color_btn_frame.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Button(color_btn_frame, text="Pick Color", command=self.enter_pick_color_mode, 
                 bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        self.tolerance_var = tk.StringVar(value="40")
        tk.Spinbox(color_btn_frame, from_=0, to=100, textvariable=self.tolerance_var, 
                  width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1).pack(side=tk.RIGHT, padx=(8,0))
        tk.Label(color_btn_frame, text="Tolerance:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.RIGHT)
        
        self.color_display = tk.Label(color_frame, text="No Color Selected", 
                                     bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                     relief=tk.FLAT, bd=1, height=1, font=('Segoe UI', 8))
        self.color_display.pack(fill=tk.X, pady=(0, 6), padx=8)

        # ========== STEP 3: CALIBRATION ==========
        calib_frame = tk.LabelFrame(right_panel, text="📐 Step 3: Calibration", 
                                  bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        calib_frame.pack(fill=tk.X, pady=(0, 8), padx=6)
        
        calib_button_frame = tk.Frame(calib_frame, bg=self.colors['bg_panel'])
        calib_button_frame.pack(fill=tk.X, padx=8, pady=6)
        
        tk.Button(calib_button_frame, text="Auto Calibrate", command=self.auto_calibrate, 
                 bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        tk.Button(calib_button_frame, text="Manual", command=self.enter_manual_calibrate_mode, 
                 bg=self.colors['info'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(calib_button_frame, text="Clear", command=self.clear_calibration, 
                 bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))

        # Axis Values
        axis_frame = tk.LabelFrame(calib_frame, text="Axis Values", 
                                 bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                                 font=('Segoe UI', 8), relief=tk.FLAT, bd=1)
        axis_frame.pack(fill=tk.X, pady=6, padx=8)
        
        axis_grid = tk.Frame(axis_frame, bg=self.colors['bg_panel'])
        axis_grid.pack(fill=tk.X, pady=4, padx=6)
        
        tk.Label(axis_grid, text="X:", bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                font=('Segoe UI', 8, 'bold')).grid(row=0, column=0, sticky='w', padx=(0,4))
        self.x_min_var = tk.StringVar(value="0")
        tk.Entry(axis_grid, textvariable=self.x_min_var, width=6, font=('Segoe UI', 8),
                relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=2)
        tk.Label(axis_grid, text="to", bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                font=('Segoe UI', 8)).grid(row=0, column=2, padx=4)
        self.x_max_var = tk.StringVar(value="100")
        tk.Entry(axis_grid, textvariable=self.x_max_var, width=6, font=('Segoe UI', 8),
                relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=2)
        
        tk.Label(axis_grid, text="Y:", bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                font=('Segoe UI', 8, 'bold')).grid(row=1, column=0, sticky='w', padx=(0,4), pady=(4,0))
        self.y_min_var = tk.StringVar(value="0")
        tk.Entry(axis_grid, textvariable=self.y_min_var, width=6, font=('Segoe UI', 8),
                relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=1, column=1, padx=2, pady=(4,0))
        tk.Label(axis_grid, text="to", bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                font=('Segoe UI', 8)).grid(row=1, column=2, padx=4, pady=(4,0))
        self.y_max_var = tk.StringVar(value="100")
        tk.Entry(axis_grid, textvariable=self.y_max_var, width=6, font=('Segoe UI', 8),
                relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=1, column=3, padx=2, pady=(4,0))

        # ========== STEP 4: SMART EXCLUSION TOOLS ==========
        exclusion_frame = tk.LabelFrame(right_panel, text="🚫 Step 4: Smart Exclusion Tools", 
                                      bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                      font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        exclusion_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        # Auto exclusion buttons
        auto_buttons_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        auto_buttons_frame.pack(fill=tk.X, padx=8, pady=6)
        
        tk.Button(auto_buttons_frame, text="🤖 Auto-Exclude Text", command=self.auto_create_text_exclusions, 
                 bg=self.colors['success'], fg='white', font=('Segoe UI', 9, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(auto_buttons_frame, text="🔲 Auto-Exclude Borders", command=self.auto_exclude_borders, 
                 bg=self.colors['info'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

        # Manual exclusion tools
        manual_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        manual_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(manual_frame, text="Define Zone", command=self.enter_exclusion_mode, 
                 bg=self.colors['warning'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        tk.Button(manual_frame, text="Delete Zone", command=self.enter_zone_deletion_mode, 
                 bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(manual_frame, text="Clear All", command=self.clear_exclusion_zones, 
                 bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))

        # Status display for exclusion zones
        exclusion_status_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        exclusion_status_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(exclusion_status_frame, text="Exclusion Zones:", 
                bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.exclusion_count_label = tk.Label(exclusion_status_frame, text="0", 
                                             bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                             relief=tk.FLAT, bd=1, font=('Segoe UI', 8, 'bold'), width=8)
        self.exclusion_count_label.pack(side=tk.RIGHT)

        # ========== STEP 5: LINE MASKING ==========
        line_masking_frame = tk.LabelFrame(right_panel, text="🎯 Step 5: Line Masking", 
                                         bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                         font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        line_masking_frame.pack(fill=tk.X, pady=(0, 8), padx=6)
        
        # Detection parameters
        params_frame = tk.Frame(line_masking_frame, bg=self.colors['bg_panel'])
        params_frame.pack(fill=tk.X, pady=6, padx=8)
        
        # First row of parameters
        param_row1 = tk.Frame(params_frame, bg=self.colors['bg_panel'])
        param_row1.pack(fill=tk.X, pady=2)
        
        tk.Label(param_row1, text="Edge Threshold:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=0, sticky='w')
        self.edge_threshold_var = tk.StringVar(value="30")
        tk.Spinbox(param_row1, from_=10, to=150, textvariable=self.edge_threshold_var, 
                  width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                  bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=(4,12))
        
        tk.Label(param_row1, text="Min Length:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=2, sticky='w')
        self.min_length_var = tk.StringVar(value="30")
        tk.Spinbox(param_row1, from_=10, to=200, textvariable=self.min_length_var, 
                  width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                  bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=4)
        
        # Second row of parameters
        param_row2 = tk.Frame(params_frame, bg=self.colors['bg_panel'])
        param_row2.pack(fill=tk.X, pady=2)
        
        tk.Label(param_row2, text="Cluster Distance:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=0, sticky='w')
        self.cluster_dist_var = tk.StringVar(value="15")
        tk.Spinbox(param_row2, from_=5, to=50, textvariable=self.cluster_dist_var, 
                  width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                  bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=(4,12))
        
        tk.Label(param_row2, text="Min Points:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=2, sticky='w')
        self.min_points_var = tk.StringVar(value="20")
        tk.Spinbox(param_row2, from_=10, to=100, textvariable=self.min_points_var, 
                  width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                  bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=4)
        
        # Detection buttons
        detect_frame = tk.Frame(line_masking_frame, bg=self.colors['bg_panel'])
        detect_frame.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Button(detect_frame, text="🔍 Simple Detection", command=self.simple_line_detection, 
                 bg=self.colors['warning'], fg='white', font=('Segoe UI', 9, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(detect_frame, text="🎯 Advanced Detection", command=self.detect_lines, 
                 bg=self.colors['accent'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        
        # Line selection controls
        line_select_frame = tk.Frame(line_masking_frame, bg=self.colors['bg_panel'])
        line_select_frame.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Button(line_select_frame, text="◀", command=self.previous_line, 
                 bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'), 
                 width=3, relief=tk.FLAT, bd=0).pack(side=tk.LEFT, padx=2)
        
        self.line_info_label = tk.Label(line_select_frame, text="No lines detected", 
                                       bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                       relief=tk.FLAT, bd=1, font=('Segoe UI', 8))
        self.line_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        
        tk.Button(line_select_frame, text="▶", command=self.next_line, 
                 bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'), 
                 width=3, relief=tk.FLAT, bd=0).pack(side=tk.RIGHT, padx=2)
        
        # Line preview
        self.line_preview_canvas = tk.Canvas(line_masking_frame, width=150, height=25, 
                                           bg=self.colors['bg_secondary'], highlightthickness=1,
                                           highlightbackground=self.colors['border'])
        self.line_preview_canvas.pack(pady=4, padx=8)
        
        # Debug info display
        self.debug_label = tk.Label(line_masking_frame, text="Debug: Ready", 
                                   bg=self.colors['bg_panel'], fg=self.colors['info'],
                                   font=('Segoe UI', 7), wraplength=250)
        self.debug_label.pack(pady=4, padx=8)
        
        # Clear lines button
        tk.Button(line_masking_frame, text="Clear Detection", command=self.clear_line_detection, 
                 bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=(0, 6), padx=8)

        # ========== STEP 6: DATA EXTRACTION ==========
        extract_frame = tk.LabelFrame(right_panel, text="📊 Step 6: Data Extraction", 
                                    bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                    font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        extract_frame.pack(fill=tk.X, pady=(0, 8), padx=6)
        
        # Method selection
        method_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        method_frame.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Label(method_frame, text="Method:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="Auto (Best)")
        method_dropdown = ttk.Combobox(method_frame, textvariable=self.method_var, width=12, 
                                      state="readonly", font=('Segoe UI', 8))
        method_dropdown['values'] = ("Auto (Best)", "Method 1", "Method 2")
        method_dropdown.pack(side=tk.RIGHT)
        method_dropdown.bind('<<ComboboxSelected>>', self.on_method_changed)
        
        # Mode and value
        mode_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        mode_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        
        tk.Label(mode_frame, text="Mode:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Step between points")
        mode_dropdown = ttk.Combobox(mode_frame, textvariable=self.mode_var, width=12, 
                                    state="readonly", font=('Segoe UI', 8))
        mode_dropdown['values'] = ("Step between points", "Number of points")
        mode_dropdown.pack(side=tk.RIGHT)
        
        # Value input
        value_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        value_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(value_frame, text="Value:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.value_var = tk.StringVar(value="10")
        tk.Entry(value_frame, textvariable=self.value_var, width=8, font=('Segoe UI', 8),
                relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).pack(side=tk.RIGHT)
        
        # Action buttons
        action_button_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        action_button_frame.pack(fill=tk.X, pady=6, padx=8)
        
        tk.Button(action_button_frame, text="📊 Extract Data", command=self.extract_data_dual, 
                 bg=self.colors['primary'], fg='white', font=('Segoe UI', 9, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(action_button_frame, text="🔍 Find Peak/Valley", command=self.find_extrema, 
                 bg=self.colors['accent'], fg='white', font=('Segoe UI', 8, 'bold'),
                 relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

        # Toggle Options
        toggle_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        toggle_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        
        self.show_best_fit = tk.BooleanVar(value=True)
        tk.Checkbutton(toggle_frame, text="Show Best Fit Line", variable=self.show_best_fit, 
                      command=self.toggle_best_fit_line, bg=self.colors['bg_panel'], 
                      fg=self.colors['text_primary'], font=('Segoe UI', 8),
                      selectcolor=self.colors['bg_secondary']).pack(side=tk.LEFT)
        
        self.use_interpolation = tk.BooleanVar(value=True)
        tk.Checkbutton(toggle_frame, text="Use Interpolation", variable=self.use_interpolation, 
                      bg=self.colors['bg_panel'], fg=self.colors['text_primary'], 
                      font=('Segoe UI', 8), selectcolor=self.colors['bg_secondary']).pack(side=tk.RIGHT)

        # Method Results Display
        results_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        results_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(results_frame, text="Results:", bg=self.colors['bg_panel'], 
                fg=self.colors['text_primary'], font=('Segoe UI', 8, 'bold')).pack(anchor='w')
        self.method_results_label = tk.Label(results_frame, text="No data extracted", 
                                           bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                           relief=tk.FLAT, bd=1, wraplength=280, font=('Segoe UI', 7),
                                           justify=tk.LEFT, anchor='w')
        self.method_results_label.pack(fill=tk.X, pady=(2, 0))

        # ========== STEP 7: EXPORT ==========
        export_frame = tk.LabelFrame(right_panel, text="💾 Step 7: Export", 
                                   bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                   font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        export_frame.pack(fill=tk.X, pady=8, padx=6)
        
        tk.Button(export_frame, text="💾 Export to CSV", command=self.export_csv, 
                 bg=self.colors['success'], fg='white', font=('Segoe UI', 10, 'bold'),
                 relief=tk.FLAT, bd=0, pady=8).pack(fill=tk.X, pady=8, padx=8)

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(right_panel, orient='horizontal', mode='determinate')
        
        # Update the old color references to use new color scheme
        self.create_color = self.colors['primary']
        self.clear_color = self.colors['danger']
        self.export_color = self.colors['success']
        self.calib_color = self.colors['info']
        self.peak_color = '#ff00ff'  # Keep magenta for peaks
        self.valley_color = '#ffa500'  # Keep orange for valleys

    # === NEW: Line Masking System Functions ===
    def detect_lines(self):
        """Trigger line detection using the advanced masking system."""
        self.line_masking.detect_all_lines()
        self.update_line_info_display()
        self.update_line_preview()
    
    def simple_line_detection(self):
        """Simple line detection specifically for spectral data - more reliable fallback. Now respects exclusion zones."""
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            self.debug_label.config(text="Debug: Running simple detection...")
            self.root.update_idletasks()
            
            # Convert to grayscale
            img_array = np.array(self.image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # NEW: Create exclusion mask
            exclusion_mask = self.line_masking._create_exclusion_mask(gray.shape)
            if exclusion_mask is not None:
                self.debug_label.config(text=f"Debug: Simple detection using {len(self.exclusion_zones)} exclusion zones...")
                self.root.update_idletasks()
                print(f"Simple detection applying {len(self.exclusion_zones)} exclusion zones")
            
            # Clear previous results
            self.line_masking.detected_lines.clear()
            self.line_masking.line_colors.clear()
            self.line_masking.line_previews.clear()
            
            # Simple threshold-based detection
            # Invert image so dark lines become bright
            inverted = 255 - gray
            
            # Apply threshold to isolate dark lines
            _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)
            
            # NEW: Apply exclusion mask if available
            if exclusion_mask is not None:
                binary = cv2.bitwise_and(binary, exclusion_mask)
            
            # Clean up with morphological operations
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components (potential lines)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            detected_masks = []
            
            for i in range(1, num_labels):  # Skip background (label 0)
                # Get component statistics
                x, y, w, h, area = stats[i]
                
                # Filter for line-like shapes
                aspect_ratio = w / h if h > 0 else 0
                
                # Look for horizontal line-like shapes or reasonable areas
                if (area > 100 and  # Minimum area
                    (aspect_ratio > 2 or area > 500) and  # Either long or substantial
                    w > 20 and h > 2):  # Minimum dimensions
                    
                    # Create mask for this component
                    component_mask = (labels == i).astype(np.uint8) * 255
                    
                    # Dilate slightly to ensure we capture the full line
                    kernel = np.ones((3,3), np.uint8)
                    component_mask = cv2.dilate(component_mask, kernel, iterations=1)
                    
                    detected_masks.append(component_mask)
            
            if not detected_masks:
                self.debug_label.config(text="Debug: No lines found. Try adjusting threshold or use color picking.")
                message = "Simple detection found no lines.\nTry:\n1. Ensure image has dark lines on light background\n2. Use color-based detection instead\n3. Adjust the image contrast"
                if not self.exclusion_zones:
                    message += "\n4. First try 'Auto-Exclude Text' to remove labels"
                messagebox.showinfo("No Lines", message)
                return
            
            # Store the detected masks
            self.line_masking.detected_lines = detected_masks
            
            # Generate colors and previews
            self.line_masking._generate_line_visualization()
            
            self.debug_label.config(text=f"Debug: Simple detection found {len(detected_masks)} lines")
            messagebox.showinfo("Simple Detection Complete", 
                              f"Found {len(detected_masks)} lines using simple detection.\n" +
                              "Use arrow buttons to cycle through them.")
            
            self.update_line_info_display()
            self.update_line_preview()
            self.redraw_overlays()
            
        except Exception as e:
            self.debug_label.config(text=f"Debug: Simple detection failed - {str(e)[:30]}...")
            messagebox.showerror("Simple Detection Error", f"Simple detection failed: {e}")
    
    def next_line(self):
        """Switch to the next detected line."""
        if self.line_masking.next_line():
            self.update_line_info_display()
            self.update_line_preview()
            self.redraw_overlays()
    
    def previous_line(self):
        """Switch to the previous detected line."""
        if self.line_masking.previous_line():
            self.update_line_info_display()
            self.update_line_preview()
            self.redraw_overlays()
    
    def update_line_info_display(self):
        """Update the line information display."""
        if not self.line_masking.detected_lines:
            self.line_info_label.config(text="No lines detected")
        else:
            current = self.line_masking.current_line_index + 1
            total = len(self.line_masking.detected_lines)
            if self.line_masking.line_colors and self.line_masking.current_line_index < len(self.line_masking.line_colors):
                color = self.line_masking.line_colors[self.line_masking.current_line_index]
                self.line_info_label.config(text=f"Line {current}/{total} ({color})")
            else:
                self.line_info_label.config(text=f"Line {current}/{total}")
    
    def update_line_preview(self):
        """Update the line preview canvas."""
        self.line_preview_canvas.delete("all")
        
        if (self.line_masking.detected_lines and 
            self.line_masking.current_line_index < len(self.line_masking.line_previews) and
            self.line_masking.line_previews[self.line_masking.current_line_index] is not None):
            
            preview_img = self.line_masking.line_previews[self.line_masking.current_line_index]
            if preview_img:
                preview_photo = ImageTk.PhotoImage(preview_img)
                self.line_preview_canvas.create_image(100, 20, image=preview_photo)
                # Keep reference to prevent garbage collection
                self.line_preview_canvas.preview_photo = preview_photo
        else:
            # Show a simple line representation if no preview available
            if self.line_masking.detected_lines:
                self.line_preview_canvas.create_line(10, 20, 190, 20, fill='red', width=3)
                self.line_preview_canvas.create_text(100, 30, text="Line mask", font=('Arial', 8))
    
    def clear_line_detection(self):
        """Clear all detected lines."""
        self.line_masking.detected_lines.clear()
        self.line_masking.line_colors.clear()
        self.line_masking.line_previews.clear()
        self.line_masking.current_line_index = 0
        
        self.update_line_info_display()
        self.update_line_preview()
        self.redraw_overlays()
        
        self.debug_label.config(text="Debug: Line detection cleared")

    def zoom_in(self):
        """Zoom in the image by 25%."""
        if self.original_image is None:
            return
        self.zoom_level *= 1.25
        self.apply_zoom()
    
    def zoom_out(self):
        """Zoom out the image by 25%."""
        if self.original_image is None:
            return
        self.zoom_level *= 0.8
        if self.zoom_level < 0.1:
            self.zoom_level = 0.1
        self.apply_zoom()
    
    def zoom_reset(self):
        """Reset zoom to 100%."""
        if self.original_image is None:
            return
        self.zoom_level = 1.0
        self.apply_zoom()
    
    def apply_zoom(self):
        """Apply the current zoom level to the image."""
        if self.original_image is None:
            return
        
        # Calculate new size
        orig_width, orig_height = self.original_image.size
        new_width = int(orig_width * self.zoom_level)
        new_height = int(orig_height * self.zoom_level)
        
        # Resize image
        if self.zoom_level != 1.0:
            resample_method = Image.Resampling.LANCZOS if self.zoom_level > 1.0 else Image.Resampling.LANCZOS
            self.image = self.original_image.resize((new_width, new_height), resample_method)
        else:
            self.image = self.original_image.copy()
        
        self.image_array = np.array(self.image)
        
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        
        # Redisplay image
        self.display_image_on_canvas()

    def update_statistics_display(self):
        """Update the statistics panel if it's open."""
        if hasattr(self, 'stats_panel') and hasattr(self.stats_panel, 'window') and self.stats_panel.window and self.stats_panel.window.winfo_exists():
            self.stats_panel.update_content()

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.root.bind("<Control-v>", self.paste_image)
        self.root.bind("<Escape>", self.escape_mode)

    # --- File Operations ---
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path: 
            try:
                img = Image.open(file_path)
                self.process_new_image(img)
                print(f"Successfully loaded image: {file_path}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load image.\nError: {e}")
                print(f"Error loading image: {e}")

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'r') as f: data = json.load(f)
            if not ('x' in data and 'y' in data): raise ValueError("JSON must contain 'x' and 'y' keys.")
            if len(data['x']) != len(data['y']): raise ValueError("JSON 'x' and 'y' arrays must have the same length.")
            self.clear_all()
            self.real_coordinates = list(zip(data['x'], data['y']))
            messagebox.showinfo("JSON Loaded", f"Successfully loaded {len(self.real_coordinates)} points from JSON.")
            self.calculate_statistics()
            self.calculate_best_fit()
        except Exception as e: messagebox.showerror("JSON Load Error", f"Failed to load or parse JSON file.\nError: {e}")

    def paste_image(self, event=None):
        try:
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image): 
                self.process_new_image(img)
                print("Successfully pasted image from clipboard")
            else:
                messagebox.showwarning("Paste Error", "No image found in clipboard or clipboard content is not an image.")
        except Exception as e: 
            messagebox.showerror("Paste Error", f"Could not paste image from clipboard.\nError: {e}")
            print(f"Paste error: {e}")

    def process_new_image(self, img):
        try:
            print("Starting image processing...")
            self.clear_all()
            self.original_image = img.convert("RGB")  # Store original
            self.image = self.original_image.copy()   # Working copy
            self.zoom_level = 1.0
            if hasattr(self, 'zoom_label'):
                self.zoom_label.config(text="100%")
            self.image_array = np.array(self.image)
            self.display_image_on_canvas()
            print(f"Image processed successfully. Size: {self.image.size}")
        except Exception as e:
            messagebox.showerror("Image Processing Error", f"Could not process image.\nError: {e}")
            print(f"Image processing error: {e}")

    def clear_all(self):
        self.image = None
        self.original_image = None
        self.image_array = None
        self.zoom_level = 1.0
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text="100%")
        self.canvas.delete("all")
        self.zoom_canvas.delete("all")
        self.clear_calibration()
        self.clear_data()
        self.clear_exclusion_zones()
        self.clear_line_detection()  # NEW: Clear line detection
        self.selected_color = None
        self.color_display.config(text="No Color Selected", bg=self.colors['bg_secondary'])
        self.method_results_label.config(text="No data extracted")
        # Update statistics panel if open and create dock tab if closed
        if hasattr(self, 'stats_panel'):
            if hasattr(self.stats_panel, 'window') and self.stats_panel.window and self.stats_panel.window.winfo_exists():
                self.stats_panel.update_content()
            elif hasattr(self.stats_panel, 'is_closed') and self.stats_panel.is_closed:
                self.stats_panel.create_dock_tab()

    def display_image_on_canvas(self):
        if self.image is None: 
            print("No image to display")
            return
        
        print("Displaying image on canvas...")
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="image")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.redraw_overlays()
        
        # Show dock tab if stats panel is closed
        try:
            if hasattr(self, 'stats_panel') and hasattr(self.stats_panel, 'is_closed') and self.stats_panel.is_closed:
                self.stats_panel.create_dock_tab()
        except Exception as e:
            print(f"Error with dock tab: {e}")
        
        print("Image displayed successfully")

    def redraw_overlays(self):
        self.draw_calibration_grid(); self.draw_calibration_points()
        self.draw_extracted_points(); self.draw_best_fit_line()
        self.draw_exclusion_zones(); self.draw_extrema_points()
        self.draw_current_line_mask()  # NEW: Draw current line mask
        
    # === NEW: Line Mask Visualization ===
    def draw_current_line_mask(self):
        """Draw the currently selected line mask as an overlay."""
        self.canvas.delete("line_mask")
        
        current_mask = self.line_masking.get_current_line_mask()
        if current_mask is None:
            return
        
        # Convert mask to overlay
        y_coords, x_coords = np.where(current_mask > 0)
        
        if len(self.line_masking.line_colors) > self.line_masking.current_line_index:
            color = self.line_masking.line_colors[self.line_masking.current_line_index]
        else:
            color = 'red'
        
        # Draw mask points
        for x, y in zip(x_coords[::5], y_coords[::5]):  # Sample every 5th point for performance
            self.canvas.create_oval(x-1, y-1, x+1, y+1, fill=color, outline='', tags="line_mask")
        
    # --- Mode Management ---
    def enter_pick_color_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "pick_color"; self.canvas.config(cursor="crosshair")

    def enter_manual_calibrate_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "manual_calibrate"; self.canvas.config(cursor="crosshair")
        messagebox.showinfo("Manual Calibration", "Click corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

    def enter_exclusion_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "exclusion"; self.canvas.config(cursor="tcross")

    def escape_mode(self, event=None):
        self.current_mode = "normal"; self.canvas.config(cursor="arrow")
        if self.temp_rect_id: self.canvas.delete(self.temp_rect_id)
        self.temp_rect_start = self.temp_rect_id = self.dragging_line_index = None

    # --- Canvas Events (Enhanced with Smart Exclusion Support) ---
    def on_canvas_click(self, event):
        if self.image is None: return
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

        line_index = self.get_line_at_pos(x, y)
        if line_index is not None:
            self.dragging_line_index = line_index; self.current_mode = "adjust_calibration"; return

        if self.current_mode == "pick_color": 
            self.select_color_at_point(x, y)
        elif self.current_mode == "manual_calibrate": 
            self.add_calibration_point(x, y)
        elif self.current_mode == "delete_zone":
            # New functionality: delete exclusion zone when clicked
            zone_index = self.find_zone_at_position(x, y)
            if zone_index is not None:
                removed_zone = self.exclusion_zones.pop(zone_index)
                self.draw_exclusion_zones()  # Redraw to show the change
                messagebox.showinfo("Zone Deleted", f"Removed exclusion zone at {removed_zone}")
            else:
                messagebox.showinfo("No Zone", "No exclusion zone found at this location.")
        elif self.current_mode == "exclusion":
            self.temp_rect_start = (x, y)
            self.temp_rect_id = self.canvas.create_rectangle(x, y, x, y, outline='red', width=2, dash=(5,3))

    def on_canvas_motion(self, event):
        if self.image is None: return
        self.mouse_x, self.mouse_y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if self.current_mode == "normal":
            line_index = self.get_line_at_pos(self.mouse_x, self.mouse_y)
            if line_index in [0, 2]: self.canvas.config(cursor="sb_v_double_arrow")
            elif line_index in [1, 3]: self.canvas.config(cursor="sb_h_double_arrow")
            else: self.canvas.config(cursor="arrow")
        self.update_zoom_preview()

    def on_canvas_drag(self, event):
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if self.current_mode == "adjust_calibration" and self.dragging_line_index is not None:
            self.adjust_calibration_box(x, y)
        elif self.current_mode == "exclusion" and self.temp_rect_start:
            x0, y0 = self.temp_rect_start; self.canvas.coords(self.temp_rect_id, x0, y0, x, y)

    def on_canvas_release(self, event):
        if self.current_mode == "adjust_calibration": self.escape_mode()
        elif self.current_mode == "exclusion" and self.temp_rect_start:
            x0, y0 = self.temp_rect_start
            x1, y1 = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
            self.exclusion_zones.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_start = self.temp_rect_id = None
            self.escape_mode(); self.draw_exclusion_zones()

    # --- SMART EXCLUSION SYSTEM - Automatic Text Detection ---
    def detect_text_regions(self):
        """
        Automatically detect text and numerical regions using multiple computer vision techniques.
        This comprehensive approach combines morphological operations, edge detection, and 
        connected component analysis to identify text while avoiding spectral data.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return []
        
        print("Starting automatic text detection...")
        
        # Convert image to formats needed for different detection methods
        img_array = np.array(self.image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        detected_regions = []
        
        # Method 1: Morphological Text Detection
        # This method uses shape-based operations to find text-like structures
        try:
            # Apply adaptive thresholding to highlight text regions
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Create kernels that connect text characters
            # Horizontal kernel to connect letters in words
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
            
            # Vertical kernel to connect parts of individual characters
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
            
            # Combine horizontal and vertical detections
            text_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Find contours of potential text regions
            contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle for each potential text region
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on size and aspect ratio to eliminate noise and spectral lines
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Text regions have specific characteristics that distinguish them from spectral data
                if (area > 100 and area < 50000 and  # Size constraints
                    aspect_ratio > 0.2 and aspect_ratio < 20 and  # Aspect ratio constraints
                    w > 10 and h > 5):  # Minimum dimensions
                    
                    # Add padding around detected text to ensure complete exclusion
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(self.image.width - x, w + 2*padding)
                    h = min(self.image.height - y, h + 2*padding)
                    
                    detected_regions.append((x, y, x + w, y + h))
            
            print(f"Method 1 detected {len(detected_regions)} potential text regions")
            
        except Exception as e:
            print(f"Text detection method 1 failed: {e}")
        
        # Method 2: Edge-based detection for sharp, geometric shapes
        # This catches text that might have been missed by the morphological approach
        try:
            # Use Canny edge detection to find sharp edges characteristic of text
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply morphological operations to connect nearby edges
            kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours of edge-based regions
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            edge_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Look for small to medium rectangular regions with lots of edges
                if (area > 50 and area < 10000 and 
                    w > 8 and h > 8 and w < 200 and h < 100):
                    
                    # Check edge density within this region - text has high edge density
                    roi_edges = edges[y:y+h, x:x+w]
                    edge_density = np.sum(roi_edges > 0) / area
                    
                    # If this region has high edge density, it's likely text
                    if edge_density > 0.02:  # Threshold based on text characteristics
                        padding = 3
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(self.image.width - x, w + 2*padding)
                        h = min(self.image.height - y, h + 2*padding)
                        
                        edge_regions.append((x, y, x + w, y + h))
            
            print(f"Method 2 detected {len(edge_regions)} edge-based regions")
            detected_regions.extend(edge_regions)
            
        except Exception as e:
            print(f"Edge detection method failed: {e}")
        
        # Method 3: Connected Component Analysis for Individual Characters
        # This specifically targets numerical labels and axis annotations
        try:
            # Apply binary threshold to isolate dark text on light background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find connected components (individual characters or character groups)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            digit_regions = []
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                
                # Look for character-sized components
                if (area > 20 and area < 1000 and 
                    w > 3 and w < 50 and h > 5 and h < 50):
                    
                    # Check if this looks like a character based on aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 3.0:  # Reasonable character proportions
                        digit_regions.append((x, y, w, h))
            
            # Group nearby digit regions into larger exclusion zones
            # This catches multi-digit numbers and text strings
            if digit_regions:
                # Sort by x-coordinate to group horizontally aligned digits
                digit_regions.sort(key=lambda r: r[0])
                
                grouped_regions = []
                current_group = [digit_regions[0]]
                
                for i in range(1, len(digit_regions)):
                    prev_region = current_group[-1]
                    curr_region = digit_regions[i]
                    
                    # Check if current digit is close enough to be part of the same number/word
                    x_distance = curr_region[0] - (prev_region[0] + prev_region[2])
                    y_distance = abs(curr_region[1] - prev_region[1])
                    
                    if x_distance < 20 and y_distance < 10:  # Close enough to group together
                        current_group.append(curr_region)
                    else:
                        # Process the current group and start a new one
                        if len(current_group) >= 1:  # At least one character
                            # Create bounding box for entire group
                            min_x = min(r[0] for r in current_group)
                            min_y = min(r[1] for r in current_group)
                            max_x = max(r[0] + r[2] for r in current_group)
                            max_y = max(r[1] + r[3] for r in current_group)
                            
                            # Add padding around the grouped text
                            padding = 8
                            min_x = max(0, min_x - padding)
                            min_y = max(0, min_y - padding)
                            max_x = min(self.image.width, max_x + padding)
                            max_y = min(self.image.height, max_y + padding)
                            
                            grouped_regions.append((min_x, min_y, max_x, max_y))
                        
                        current_group = [curr_region]
                
                # Don't forget the last group
                if len(current_group) >= 1:
                    min_x = min(r[0] for r in current_group)
                    min_y = min(r[1] for r in current_group)
                    max_x = max(r[0] + r[2] for r in current_group)
                    max_y = max(r[1] + r[3] for r in current_group)
                    
                    padding = 8
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(self.image.width, max_x + padding)
                    max_y = min(self.image.height, max_y + padding)
                    
                    grouped_regions.append((min_x, min_y, max_x, max_y))
                
                print(f"Method 3 detected {len(grouped_regions)} numerical regions")
                detected_regions.extend(grouped_regions)
        
        except Exception as e:
            print(f"Numerical detection method failed: {e}")
        
        # Remove duplicate and overlapping regions
        # This prevents redundant exclusions from multiple detection methods
        filtered_regions = self.remove_overlapping_regions(detected_regions)
        
        # Final safety check: ensure we're not excluding the main spectral data area
        filtered_regions = self.filter_spectral_line_regions(filtered_regions)
        
        print(f"Final result: {len(filtered_regions)} text regions detected after filtering")
        return filtered_regions

    def remove_overlapping_regions(self, regions):
        """
        Remove duplicate and heavily overlapping exclusion regions.
        This prevents redundant exclusions and improves performance.
        """
        if not regions:
            return []
        
        # Sort regions by area (largest first) to prioritize keeping larger regions
        sorted_regions = sorted(regions, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
        filtered = []
        
        for region in sorted_regions:
            # Check if this region significantly overlaps with any already accepted region
            overlaps_significantly = False
            
            for existing in filtered:
                overlap_area = self.calculate_overlap_area(region, existing)
                region_area = (region[2] - region[0]) * (region[3] - region[1])
                
                # If more than 70% of this region overlaps with an existing one, skip it
                if overlap_area > 0.7 * region_area:
                    overlaps_significantly = True
                    break
            
            if not overlaps_significantly:
                filtered.append(region)
        
        return filtered

    def calculate_overlap_area(self, region1, region2):
        """Calculate the area of overlap between two rectangular regions."""
        x1_max = max(region1[0], region2[0])
        y1_max = max(region1[1], region2[1])
        x2_min = min(region1[2], region2[2])
        y2_min = min(region1[3], region2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0  # No overlap
        
        return (x2_min - x1_max) * (y2_min - y1_max)

    def filter_spectral_line_regions(self, regions):
        """
        Remove any exclusion regions that might interfere with the main spectral line.
        This is a critical safety check to ensure we never exclude actual data.
        """
        if len(self.calibration_points) != 4:
            return regions  # Can't check without calibration
        
        # Calculate the calibrated area for safety checking
        calib_x = [p[0] for p in self.calibration_points]
        calib_y = [p[1] for p in self.calibration_points]
        calib_area = (max(calib_x) - min(calib_x)) * (max(calib_y) - min(calib_y))
        
        # Identify the likely spectral line path if we have extracted points
        spectral_path_regions = []
        if hasattr(self, 'extracted_points') and self.extracted_points:
            # Create protective zones around known spectral data points
            for point in self.extracted_points:
                # Create a small protective zone around each data point
                padding = 15
                x, y = point
                spectral_path_regions.append((
                    max(0, x - padding),
                    max(0, y - padding),
                    min(self.image.width, x + padding),
                    min(self.image.height, y + padding)
                ))
        
        filtered = []
        for region in regions:
            region_area = (region[2] - region[0]) * (region[3] - region[1])
            
            # Reject regions that are too large (might cover the main graph)
            if region_area > 0.3 * calib_area:
                print(f"Rejecting large region that might contain spectral data: {region}")
                continue
            
            # Check if this region overlaps with known spectral data points
            overlaps_spectral_data = False
            for spectral_region in spectral_path_regions:
                overlap = self.calculate_overlap_area(region, spectral_region)
                if overlap > 0:
                    overlaps_spectral_data = True
                    break
            
            if overlaps_spectral_data:
                print(f"Rejecting region that overlaps with spectral data: {region}")
                continue
            
            filtered.append(region)
        
        return filtered

    def auto_exclude_borders(self):
        """
        Automatically detect and exclude rectangular borders/frames around graph panels.
        This helps separate multi-panel plots like (A), (B), (C), (D) layouts.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            print("Starting automatic border detection...")
            
            # Convert image to formats needed for detection
            img_array = np.array(self.image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            detected_borders = []
            
            # Method 1: Strong edge detection for borders
            try:
                # Use strong edge detection to find clear boundaries
                edges = cv2.Canny(gray, 100, 200, apertureSize=3)
                
                # Dilate edges to connect broken border lines
                kernel = np.ones((3,3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=2)
                
                # Find contours of potential borders
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    
                    # Look for large rectangular areas that could be panel borders
                    # Filter for substantial rectangles that aren't too thin
                    if (area > 10000 and  # Large enough to be a panel
                        w > 100 and h > 100 and  # Reasonable minimum dimensions
                        area < 0.8 * self.image.width * self.image.height):  # Not the entire image
                        
                        # Check if this contour is roughly rectangular
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                        
                        # If it approximates to 4 corners, it's likely a rectangle
                        if len(approx) >= 4:
                            # Create border exclusion zone (the border itself, not the interior)
                            border_thickness = 8  # Pixels to exclude around the border
                            
                            # Top border
                            detected_borders.append((
                                max(0, x - border_thickness), 
                                max(0, y - border_thickness),
                                min(self.image.width, x + w + border_thickness), 
                                min(self.image.height, y + border_thickness*2)
                            ))
                            
                            # Bottom border  
                            detected_borders.append((
                                max(0, x - border_thickness), 
                                max(0, y + h - border_thickness*2),
                                min(self.image.width, x + w + border_thickness), 
                                min(self.image.height, y + h + border_thickness)
                            ))
                            
                            # Left border
                            detected_borders.append((
                                max(0, x - border_thickness), 
                                max(0, y - border_thickness),
                                min(self.image.width, x + border_thickness*2), 
                                min(self.image.height, y + h + border_thickness)
                            ))
                            
                            # Right border
                            detected_borders.append((
                                max(0, x + w - border_thickness*2), 
                                max(0, y - border_thickness),
                                min(self.image.width, x + w + border_thickness), 
                                min(self.image.height, y + h + border_thickness)
                            ))
                
                print(f"Method 1 detected {len(detected_borders)} border segments")
                
            except Exception as e:
                print(f"Border detection method 1 failed: {e}")
            
            # Method 2: Line detection for straight borders
            try:
                # Use HoughLines to detect long straight lines (likely borders)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                
                if lines is not None:
                    # Group lines by orientation (horizontal vs vertical)
                    horizontal_lines = []
                    vertical_lines = []
                    
                    for line in lines:
                        rho, theta = line[0]
                        
                        # Classify as horizontal or vertical based on angle
                        angle_deg = np.degrees(theta)
                        
                        if (angle_deg < 10 or angle_deg > 170):  # Near horizontal
                            horizontal_lines.append((rho, theta))
                        elif (80 < angle_deg < 100):  # Near vertical
                            vertical_lines.append((rho, theta))
                    
                    # Create exclusion zones for detected border lines
                    border_thickness = 6
                    
                    for rho, theta in horizontal_lines[:10]:  # Limit to avoid too many
                        # Convert to cartesian coordinates
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        
                        # Calculate line endpoints across image width
                        y_line = int(y0)
                        if 0 < y_line < self.image.height:
                            detected_borders.append((
                                0, 
                                max(0, y_line - border_thickness),
                                self.image.width, 
                                min(self.image.height, y_line + border_thickness)
                            ))
                    
                    for rho, theta in vertical_lines[:10]:  # Limit to avoid too many
                        # Convert to cartesian coordinates  
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        
                        # Calculate line endpoints across image height
                        x_line = int(x0)
                        if 0 < x_line < self.image.width:
                            detected_borders.append((
                                max(0, x_line - border_thickness), 
                                0,
                                min(self.image.width, x_line + border_thickness), 
                                self.image.height
                            ))
                    
                    print(f"Method 2 detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical border lines")
                
            except Exception as e:
                print(f"Border detection method 2 failed: {e}")
            
            if not detected_borders:
                messagebox.showinfo("No Borders", "No clear panel borders detected.\n" +
                                  "This works best with images that have distinct rectangular frames around each graph panel.")
                return
            
            # Remove overlapping border regions to avoid redundancy
            filtered_borders = self.remove_overlapping_regions(detected_borders)
            
            # Add detected border regions to existing exclusion zones
            self.exclusion_zones.extend(filtered_borders)
            
            # Redraw to show the new exclusion zones
            self.draw_exclusion_zones()
            
            messagebox.showinfo("Border Exclusion Complete", 
                              f"Added {len(filtered_borders)} border exclusion zones.\n"
                              f"Total exclusion zones: {len(self.exclusion_zones)}\n\n"
                              f"This should help separate individual graph panels for line detection.")
            
        except Exception as e:
            messagebox.showerror("Border Exclusion Error", f"Failed to detect borders: {e}")

    def auto_create_text_exclusions(self):
        """
        Main function to automatically create exclusion zones for text and numbers.
        This is the primary interface function called from the UI button.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            # Detect text regions using our comprehensive multi-method approach
            text_regions = self.detect_text_regions()
            
            if not text_regions:
                messagebox.showinfo("Auto-Exclusion", "No text regions detected.")
                return
            
            # Add detected regions to existing exclusion zones
            # We extend rather than replace to preserve any manual exclusions
            self.exclusion_zones.extend(text_regions)
            
            # Redraw to show the new exclusion zones
            self.draw_exclusion_zones()
            
            messagebox.showinfo("Auto-Exclusion Complete", 
                              f"Added {len(text_regions)} automatic text exclusion zones.\n"
                              f"Total exclusion zones: {len(self.exclusion_zones)}")
            
        except Exception as e:
            messagebox.showerror("Auto-Exclusion Error", f"Failed to create automatic exclusions: {e}")

    def enter_zone_deletion_mode(self):
        """
        Enter interactive mode where clicking on an exclusion zone removes it.
        This provides precise control over which zones to keep or remove.
        """
        if not self.exclusion_zones:
            messagebox.showinfo("No Zones", "No exclusion zones to delete.")
            return
        
        self.current_mode = "delete_zone"
        self.canvas.config(cursor="pirate")  # Visual indicator for deletion mode
        messagebox.showinfo("Zone Deletion Mode", 
                           "Click on any exclusion zone to remove it.\n"
                           "Press Escape when finished.")

    def find_zone_at_position(self, x, y):
        """
        Find which exclusion zone (if any) contains the given coordinates.
        Returns the index of the zone, or None if no zone contains the point.
        """
        for i, (x0, y0, x1, y1) in enumerate(self.exclusion_zones):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    # --- Method Selection ---
    def on_method_changed(self, event=None):
        """Handle method dropdown changes."""
        selected = self.method_var.get()
        if selected == "Auto (Best)":
            self.current_method = "auto"
        elif selected == "Method 1":
            self.current_method = "method1"
        elif selected == "Method 2":
            self.current_method = "method2"
        
        # If we have already extracted data, switch to the selected method
        if self.method1_points or self.method2_points:
            self.apply_selected_method()

    def apply_selected_method(self):
        """Apply the currently selected method to display data."""
        if self.current_method == "auto":
            # Use the method with more points
            if len(self.method1_points) >= len(self.method2_points):
                self.extracted_points = self.method1_points
                selected_text = f"Auto: Method 1 ({len(self.method1_points)} pts)"
            else:
                self.extracted_points = self.method2_points
                selected_text = f"Auto: Method 2 ({len(self.method2_points)} pts)"
        elif self.current_method == "method1":
            self.extracted_points = self.method1_points
            selected_text = f"Method 1: {len(self.method1_points)} points"
        elif self.current_method == "method2":
            self.extracted_points = self.method2_points
            selected_text = f"Method 2: {len(self.method2_points)} points"
        
        # Update results display
        results_text = f"Method 1: {len(self.method1_points)} pts\n"
        results_text += f"Method 2: {len(self.method2_points)} pts\n"
        results_text += f"Using: {selected_text}"
        self.method_results_label.config(text=results_text)
        
        # Update visualization
        self.convert_to_real_coordinates()
        self.draw_extracted_points()
        self.calculate_statistics()
        self.calculate_best_fit()

    # --- Feature Implementations ---
    def update_zoom_preview(self):
        if self.image is None: return
        size = 50
        x_start, y_start = max(0, self.mouse_x - size//2), max(0, self.mouse_y - size//2)
        x_end, y_end = x_start + size, y_start + size
        
        zoom_region = self.image.crop((x_start, y_start, x_end, y_end))
        zoom_enlarged = zoom_region.resize((150, 150), Image.Resampling.NEAREST)  # Smaller zoom
        
        overlay_img = Image.new('RGBA', (150, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        
        for (x0, y0, x1, y1) in self.exclusion_zones:
            clipped_x0 = max(x0, x_start); clipped_y0 = max(y0, y_start)
            clipped_x1 = min(x1, x_end); clipped_y1 = min(y1, y_end)
            if clipped_x1 > clipped_x0 and clipped_y1 > clipped_y0:
                zoom_x0 = (clipped_x0 - x_start) * (150/size); zoom_y0 = (clipped_y0 - y_start) * (150/size)
                zoom_x1 = (clipped_x1 - x_start) * (150/size); zoom_y1 = (clipped_y1 - y_start) * (150/size)
                draw.rectangle([zoom_x0, zoom_y0, zoom_x1, zoom_y1], fill=(255, 0, 0, 80))

        if self.extracted_points:
            point_color = self.get_opposite_color(self.selected_color)
            for px, py in self.extracted_points:
                if x_start <= px < x_end and y_start <= py < y_end:
                    zoom_x = (px - x_start) * (150/size); zoom_y = (py - y_start) * (150/size)
                    draw.ellipse((zoom_x-2, zoom_y-2, zoom_x+2, zoom_y+2), fill=point_color)
        
        if self.best_fit_line_pixel_coords and self.show_best_fit.get():
            self.draw_line_in_zoom(draw, self.best_fit_line_pixel_coords, 'purple', size, x_start, y_start, 150)
            
        if len(self.calibration_points) == 4:
            points_in_zoom = []
            for i in range(4):
                px, py = self.calibration_points[i]
                zoom_x, zoom_y = (px - x_start) * (150/size), (py - y_start) * (150/size)
                points_in_zoom.append((zoom_x, zoom_y))
                if 0 <= zoom_x < 150 and 0 <= zoom_y < 150:
                    draw.ellipse((zoom_x-4, zoom_y-4, zoom_x+4, zoom_y+4), fill=self.calib_color, outline='black')
            for i in range(4):
                draw.line([points_in_zoom[i], points_in_zoom[(i+1)%4]], fill=self.calib_color, width=2)
        
        zoom_enlarged.paste(overlay_img, (0, 0), overlay_img)
        
        final_draw = ImageDraw.Draw(zoom_enlarged)
        center = 75  # Adjusted for smaller zoom
        final_draw.line([(center, 0), (center, 150)], fill='red', width=1)
        final_draw.line([(0, center), (150, center)], fill='red', width=1)

        self.zoom_photo = ImageTk.PhotoImage(zoom_enlarged)
        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(75, 75, image=self.zoom_photo)  # Adjusted center
        coords_text = f"({self.mouse_x}, {self.mouse_y})"
        self.zoom_canvas.create_text(5, 5, text=coords_text, anchor=tk.NW, fill='black', font=('Arial', 9, 'bold'))

    def draw_line_in_zoom(self, draw, line_coords, color, size, x_start, y_start, zoom_size=150):
        x1, y1, x2, y2 = line_coords
        zx1 = (x1 - x_start) * (zoom_size/size); zy1 = (y1 - y_start) * (zoom_size/size)
        zx2 = (x2 - x_start) * (zoom_size/size); zy2 = (y2 - y_start) * (zoom_size/size)
        draw.line([(zx1, zy1), (zx2, zy2)], fill=color, width=2)

    def select_color_at_point(self, x, y):
        if 0 <= y < self.image_array.shape[0] and 0 <= x < self.image_array.shape[1]:
            color = self.image_array[y, x]
            self.selected_color = tuple(color)
            color_hex = f'#{self.selected_color[0]:02x}{self.selected_color[1]:02x}{self.selected_color[2]:02x}'
            self.color_display.config(text=f"RGB: {self.selected_color}", bg=color_hex)
        self.escape_mode()

    # --- CALIBRATION ---
    def auto_calibrate(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        try:
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: raise ValueError("No contours found.")
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour); box = cv2.boxPoints(rect); box = box.astype(int) 
            points = sorted(box, key=lambda p: p[1])
            top_points = sorted(points[:2], key=lambda p: p[0]); bottom_points = sorted(points[2:], key=lambda p: p[0])
            self.calibration_points = [tuple(top_points[0]), tuple(top_points[1]), tuple(bottom_points[1]), tuple(bottom_points[0])]
            messagebox.showinfo("Auto-Calibration", "Found graph area. You can now drag the sides to fine-tune.")
            self.redraw_overlays(); self.escape_mode()
        except Exception as e:
            messagebox.showerror("Auto-Calibration Error", f"Could not detect a clear graph area. Using margins as a fallback.\nError: {e}")
            w, h = self.image.size; margin = min(w, h) // 10
            self.calibration_points = [(margin, margin), (w-margin, margin), (w-margin, h-margin), (margin, h-margin)]
            self.redraw_overlays()

    def add_calibration_point(self, x, y):
        if len(self.calibration_points) < 4:
            self.calibration_points.append((x, y)); self.redraw_overlays()
        if len(self.calibration_points) == 4:
            self.escape_mode(); messagebox.showinfo("Calibration Complete", "4 points set. Drag sides to adjust.")

    def clear_calibration(self):
        self.calibration_points = []; self.canvas.delete("calibration_points", "calibration_grid")

    def draw_calibration_points(self):
        self.canvas.delete("calibration_points")
        if not self.calibration_points: return
        for i, (x, y) in enumerate(self.calibration_points):
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=self.calib_color, outline='black', width=1, tags="calibration_points")

    def draw_calibration_grid(self):
        self.canvas.delete("calibration_grid")
        if len(self.calibration_points) != 4: return
        points = self.calibration_points
        for i in range(4):
            p1 = points[i]; p2 = points[(i + 1) % 4]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=self.calib_color, width=2, dash=(5,3), tags="calibration_grid")

    def get_line_at_pos(self, x, y, tolerance=5):
        if len(self.calibration_points) != 4: return None
        points = self.calibration_points
        lines = [(points[0], points[1]), (points[1], points[2]), (points[2], points[3]), (points[3], points[0])]
        for i, (p1, p2) in enumerate(lines):
            p1, p2 = np.array(p1), np.array(p2); p3 = np.array([x, y])
            if np.dot(p3 - p1, p2 - p1) >= 0 and np.dot(p3 - p2, p1 - p2) >= 0:
                dist = np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
                if dist < tolerance: return i
        return None

    def adjust_calibration_box(self, x, y):
        p = self.calibration_points; idx = self.dragging_line_index
        if idx == 0: p[0] = (p[0][0], y); p[1] = (p[1][0], y)
        elif idx == 1: p[1] = (x, p[1][1]); p[2] = (x, p[2][1])
        elif idx == 2: p[2] = (p[2][0], y); p[3] = (p[3][0], y)
        elif idx == 3: p[3] = (x, p[3][1]); p[0] = (x, p[0][1])
        self.redraw_overlays()

    # --- DATA EXTRACTION & ANALYSIS ---
    def clear_exclusion_zones(self):
        self.exclusion_zones = []; self.draw_exclusion_zones()

    def draw_exclusion_zones(self):
        self.canvas.delete("exclusion")
        for (x0, y0, x1, y1) in self.exclusion_zones:
            self.canvas.create_rectangle(x0, y0, x1, y1, fill='red', stipple='gray25', outline='', tags="exclusion")
        
        # Update the count display for the enhanced UI
        if hasattr(self, 'exclusion_count_label'):
            self.exclusion_count_label.config(text=f"{len(self.exclusion_zones)}")

    def is_in_exclusion_zone(self, x, y):
        for (x0, y0, x1, y1) in self.exclusion_zones:
            if x0 <= x <= x1 and y0 <= y <= y1: return True
        return False

    # --- Toggles ---
    def toggle_best_fit_line(self):
        """Toggle the visibility of the best fit line."""
        self.draw_best_fit_line()

    # --- METHOD 1: Original Enhanced Detection ---
    def extract_data_method1(self):
        """Original method with color masking and weighted averaging."""
        try:
            self.color_tolerance = int(self.tolerance_var.get())
            
            # NEW: Apply line mask if available
            if self.line_masking.detected_lines:
                mask = self.line_masking.get_current_line_mask()
                if mask is not None:
                    # Use the line mask instead of color detection
                    gray_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
                else:
                    mask = self.enhance_line_detection(self.image_array, self.selected_color, self.color_tolerance)
                    gray_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            else:
                mask = self.enhance_line_detection(self.image_array, self.selected_color, self.color_tolerance)
                gray_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)

            calib_x = [p[0] for p in self.calibration_points]; calib_y = [p[1] for p in self.calibration_points]
            min_x, max_x, min_y, max_y = min(calib_x), max(calib_x), min(calib_y), max(calib_y)
            
            all_points = []
            for x in range(min_x, max_x):
                y_indices = np.where(mask[min_y:max_y, x] > 0)[0]
                
                if y_indices.size > 0:
                    y_coords_in_col = y_indices + min_y
                    
                    grayscale_values = gray_image[y_coords_in_col, x]
                    weights = 255.0 - grayscale_values.astype(np.float32)
                    
                    if np.sum(weights) > 0:
                        y_center = np.sum(y_coords_in_col * weights) / np.sum(weights)
                        if not self.is_in_exclusion_zone(x, int(y_center)):
                            all_points.append((x, int(y_center)))
                    else: 
                        y_center = np.mean(y_coords_in_col)
                        if not self.is_in_exclusion_zone(x, int(y_center)):
                            all_points.append((x, int(y_center)))

            if not all_points: 
                return []
            
            # Adaptive Interpolation for Steep Slopes (if enabled)
            if self.use_interpolation.get() and len(all_points) > 1:
                interpolated_points = [all_points[0]]
                for j in range(len(all_points) - 1):
                    p1 = all_points[j]
                    p2 = all_points[j+1]
                    y_dist = abs(p2[1] - p1[1])
                    
                    if y_dist > 2:
                        num_new_points = int(y_dist * (1))
                        if num_new_points > 1:
                            x_vals = np.linspace(p1[0], p2[0], num_new_points, dtype=int)
                            y_vals = np.linspace(p1[1], p2[1], num_new_points, dtype=int)
                            for k in range(1, num_new_points):
                                interpolated_points.append((x_vals[k], y_vals[k]))
                    
                    interpolated_points.append(p2)
                
                return sorted(list(set(interpolated_points)), key=lambda p: p[0])
            else:
                return all_points

        except Exception as e:
            print(f"Method 1 failed: {e}")
            return []

    # --- METHOD 2: Advanced Detection ---
    def find_line_y_in_column(self, column_data, last_y=None, max_thickness=15):
        """Find the most probable Y-coordinate of the line in a given column."""
        y_coords_detected = np.where(column_data > 0)[0]
        if y_coords_detected.size == 0:
            return None

        # Split detected pixels into contiguous segments
        diffs = np.diff(y_coords_detected)
        segments = np.split(y_coords_detected, np.where(diffs > 1.5)[0] + 1)
        
        if not segments or segments[0].size == 0:
            return None
            
        # Filter out segments that are too thick
        valid_segments = []
        for s in segments:
            if s.size > 0 and s.size <= max_thickness:
                valid_segments.append(s)

        if not valid_segments:
            return None

        # Calculate the center (mean) of each valid segment
        segment_centers = [int(np.mean(s)) for s in valid_segments]
        
        # If no previous point, choose the first valid segment
        if last_y is None:
            return segment_centers[0]

        # If there's a previous point, find the valid segment closest to it
        closest_y = -1
        min_dist = float('inf')
        for y_center in segment_centers:
            dist = abs(y_center - last_y)
            if dist < min_dist:
                min_dist = dist
                closest_y = y_center
        
        # Add tolerance: if the closest point is still too far, it might be noise
        if min_dist > 50:
            return None 

        return closest_y

    def extract_data_method2(self):
        """Advanced method with thickness filtering and line masking support."""
        try:
            # NEW: Check if we should use line mask instead of color detection
            if self.line_masking.detected_lines:
                final_mask = self.line_masking.get_current_line_mask()
                if final_mask is None:
                    return []
            else:
                # Original color-based detection
                q_color = self.selected_color
                bgr_color = np.array([q_color[2], q_color[1], q_color[0]])  # Convert RGB to BGR
                tolerance = int(self.tolerance_var.get())
                lower_bound = np.clip(bgr_color - tolerance, 0, 255)
                upper_bound = np.clip(bgr_color + tolerance, 0, 255)
                
                # Convert image to BGR for OpenCV
                bgr_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)
                color_mask = cv2.inRange(bgr_image, lower_bound, upper_bound)

                # Create a mask for the plot area defined by calibration points
                plot_mask = np.zeros_like(color_mask)
                calib_pts_int = np.array(self.calibration_points, dtype=np.int32)
                cv2.fillPoly(plot_mask, [calib_pts_int], 255)

                # Erode the mask to avoid axes
                kernel = np.ones((3,3), np.uint8)
                eroded_plot_mask = cv2.erode(plot_mask, kernel, iterations=2)

                # Combine masks to get only the line within the eroded plot area
                final_mask = cv2.bitwise_and(color_mask, eroded_plot_mask)

            # Determine which X-values to sample based on user settings
            min_x = min(p[0] for p in self.calibration_points)
            max_x = max(p[0] for p in self.calibration_points)
            min_y = min(p[1] for p in self.calibration_points)
            max_y = max(p[1] for p in self.calibration_points)

            # Parse extraction mode and value
            mode = self.mode_var.get()
            try:
                value = float(self.value_var.get())
            except ValueError:
                value = 10  # Default value

            # Determine X values to sample based on mode
            x_values_to_sample = []
            if mode == "Number of points":
                num_points = int(value)
                if num_points < 2:
                    num_points = 2
                x_values_to_sample = np.linspace(min_x, max_x, num_points, dtype=int)
            else:  # Step between points
                step = int(value)
                if step <= 0:
                    step = 1
                x_values_to_sample = range(min_x, max_x + 1, step)

            # Find the Y-coordinate of the line for each sampled X
            pixel_points = []
            last_found_y = None 
            img_width = self.image_array.shape[1]

            for x in x_values_to_sample:
                if 0 <= x < img_width:
                    col_data = final_mask[:, x]
                    # Use the refined helper function with thickness check
                    y = self.find_line_y_in_column(col_data, last_found_y)
                    if y is not None and not self.is_in_exclusion_zone(x, y):
                        pixel_points.append((x, y))
                        last_found_y = y

            # Add Highest and Lowest Points
            max_intensity_y_pixel = -1
            min_intensity_y_pixel = -1
            max_intensity_x_pixel = -1
            min_intensity_x_pixel = -1

            # Scan for highest point (smallest Y pixel value)
            found_max = False
            for y_scan in range(min_y, max_y + 1):
                for x_scan in range(min_x, max_x + 1):
                    if final_mask[y_scan, x_scan] > 0:
                        max_intensity_y_pixel = y_scan
                        max_intensity_x_pixel = x_scan
                        found_max = True
                        break
                if found_max:
                    break

            # Scan for lowest point (largest Y pixel value)
            found_min = False
            for y_scan in range(max_y, min_y - 1, -1):
                for x_scan in range(min_x, max_x + 1):
                    if final_mask[y_scan, x_scan] > 0:
                        min_intensity_y_pixel = y_scan
                        min_intensity_x_pixel = x_scan
                        found_min = True
                        break
                if found_min:
                    break

            if max_intensity_x_pixel != -1 and max_intensity_y_pixel != -1:
                highest_point = (max_intensity_x_pixel, max_intensity_y_pixel)
                is_duplicate = False
                for p in pixel_points:
                    if abs(p[0] - highest_point[0]) < 5 and abs(p[1] - highest_point[1]) < 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    pixel_points.append(highest_point)

            if min_intensity_x_pixel != -1 and min_intensity_y_pixel != -1:
                lowest_point = (min_intensity_x_pixel, min_intensity_y_pixel)
                is_duplicate = False
                for p in pixel_points:
                    if abs(p[0] - lowest_point[0]) < 5 and abs(p[1] - lowest_point[1]) < 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    pixel_points.append(lowest_point)

            # Ensure points are sorted by X-coordinate
            pixel_points.sort(key=lambda p: p[0])

            return pixel_points

        except Exception as e:
            print(f"Method 2 failed: {e}")
            return []

    def extract_data_dual(self):
        """Run both detection methods and display based on selected method."""
        if not all([self.image, len(self.calibration_points) == 4]):
            missing = []
            if not self.image: missing.append("An image is loaded")
            if len(self.calibration_points) != 4: missing.append("4 calibration points are set")
            if not self.selected_color and not self.line_masking.detected_lines: missing.append("A line color is selected OR lines are detected")
            
            messagebox.showwarning("Prerequisites Missing", f"Please ensure:\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(missing)]))
            return
        
        self.clear_data()
        
        try:
            self.progress_bar.pack(fill=tk.X, pady=2, padx=5)
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = 100
            self.root.update_idletasks()

            # Run Method 1
            self.progress_bar['value'] = 25
            self.root.update_idletasks()
            self.method1_points = self.extract_data_method1()
            
            # Run Method 2
            self.progress_bar['value'] = 75
            self.root.update_idletasks()
            self.method2_points = self.extract_data_method2()
            
            self.progress_bar['value'] = 100
            self.root.update_idletasks()

            # Apply the selected method
            self.apply_selected_method()

        except Exception as e:
            messagebox.showerror("Extraction Error", f"Failed during data extraction.\nError: {e}")
            return
        finally:
            self.progress_bar['value'] = 0
            self.progress_bar.pack_forget()

    def enhance_line_detection(self, image_array, target_color, tolerance):
        """Enhanced line detection used by Method 1."""
        img_float = image_array.astype(np.float32)
        target_color_float = np.array(target_color, dtype=np.float32)
        dist_sq = np.sum((img_float - target_color_float) ** 2, axis=-1)
        mask = (dist_sq <= tolerance ** 2).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def clear_data(self):
        self.extracted_points, self.real_coordinates, self.statistics, self.best_fit_equation = [], [], {}, ""
        self.best_fit_line_pixel_coords = None
        self.method1_points, self.method2_points = [], []
        self.method_results_label.config(text="No data extracted")
        self.canvas.delete("extracted", "best_fit", "extrema")

    def get_opposite_color(self, rgb):
        return f'#{255-rgb[0]:02x}{255-rgb[1]:02x}{255-rgb[2]:02x}' if rgb else "#000000"

    def draw_extracted_points(self):
        self.canvas.delete("extracted")
        if not self.extracted_points: return
        point_color = self.get_opposite_color(self.selected_color)
        for x, y in self.extracted_points:
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=point_color, outline="", tags="extracted")

    def find_extrema(self):
        if not self.real_coordinates: messagebox.showwarning("No Data", "Please extract data first."); return
        y_values = [p[1] for p in self.real_coordinates]
        max_idx = np.argmax(y_values); min_idx = np.argmin(y_values)
        self.statistics['Peak'] = self.real_coordinates[max_idx]
        self.statistics['Valley'] = self.real_coordinates[min_idx]
        self.draw_extrema_points(); self.update_statistics_display()

    def draw_extrema_points(self):
        self.canvas.delete("extrema")
        if 'Peak' not in self.statistics or not self.extracted_points: return
        peak_real = self.statistics['Peak']; valley_real = self.statistics['Valley']
        peak_idx = np.argmin([np.linalg.norm(np.array(p) - np.array(peak_real)) for p in self.real_coordinates])
        valley_idx = np.argmin([np.linalg.norm(np.array(p) - np.array(valley_real)) for p in self.real_coordinates])
        
        peak_px_x, peak_px_y = self.extracted_points[peak_idx]
        self.canvas.create_oval(peak_px_x-5, peak_px_y-5, peak_px_x+5, peak_px_y+5, fill=self.peak_color, outline='black', width=1, tags='extrema')
        valley_px_x, valley_px_y = self.extracted_points[valley_idx]
        self.canvas.create_oval(valley_px_x-5, valley_px_y-5, valley_px_x+5, valley_px_y+5, fill=self.valley_color, outline='black', width=1, tags='extrema')

    def convert_to_real_coordinates(self):
        if len(self.calibration_points) != 4 or not self.extracted_points: return
        x_min, x_max = float(self.x_min_var.get()), float(self.x_max_var.get())
        y_min, y_max = float(self.y_min_var.get()), float(self.y_max_var.get())
        src = np.array(self.calibration_points, dtype='float32')
        dst = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src, dst)
        self.real_coordinates = cv2.perspectiveTransform(np.array([self.extracted_points], dtype='float32'), matrix)[0].tolist()

    def calculate_statistics(self):
        if not self.real_coordinates: return
        x_vals, y_vals = [p[0] for p in self.real_coordinates], [p[1] for p in self.real_coordinates]
        self.statistics = {
            'Points': len(self.real_coordinates),
            'X Mean': np.mean(x_vals), 'X Std Dev': np.std(x_vals),
            'X Skew': stats.skew(x_vals), 'X Kurtosis': stats.kurtosis(x_vals),
            'Y Mean': np.mean(y_vals), 'Y Std Dev': np.std(y_vals),
            'Y Skew': stats.skew(y_vals), 'Y Kurtosis': stats.kurtosis(y_vals),
            'Correlation': np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0,
            'Area (Trapezoid)': np.trapz(y_vals, x_vals) if len(x_vals) > 1 else 0
        }
        self.update_statistics_display()

    def calculate_best_fit(self):
        if len(self.real_coordinates) < 2: return
        x_vals, y_vals = np.array([p[0] for p in self.real_coordinates]), np.array([p[1] for p in self.real_coordinates])
        slope, intercept, r_val, _, _ = stats.linregress(x_vals, y_vals)
        self.statistics['r_squared'] = r_val**2
        self.best_fit_equation = f"y = {slope:.4f}x + {intercept:.4f}"
        
        x_vals_real = [p[0] for p in self.real_coordinates]; y_vals_real = [p[1] for p in self.real_coordinates]
        start_real_x = min(x_vals_real); end_real_x = max(x_vals_real)
        start_real_y = slope * start_real_x + intercept; end_real_y = slope * end_real_x + intercept
        start_real, end_real = (start_real_x, start_real_y), (end_real_x, end_real_y)
        x_min, x_max = float(self.x_min_var.get()), float(self.x_max_var.get())
        y_min, y_max = float(self.y_min_var.get()), float(self.y_max_var.get())
        dst = np.array(self.calibration_points, dtype='float32')
        src = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype='float32')
        inv_matrix = cv2.getPerspectiveTransform(src, dst)
        pixel_pts = cv2.perspectiveTransform(np.array([[start_real, end_real]], dtype='float32'), inv_matrix)[0]
        self.best_fit_line_pixel_coords = (pixel_pts[0][0], pixel_pts[0][1], pixel_pts[1][0], pixel_pts[1][1])
        
        self.draw_best_fit_line(); self.update_statistics_display()

    def draw_best_fit_line(self):
        self.canvas.delete("best_fit")
        if self.best_fit_line_pixel_coords and self.show_best_fit.get():
            x1, y1, x2, y2 = self.best_fit_line_pixel_coords
            self.canvas.create_line(x1, y1, x2, y2, fill='purple', width=2, tags="best_fit")

    def export_csv(self):
        if not self.real_coordinates: messagebox.showwarning("No Data", "No data has been extracted to export."); return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path: return
        try:
            df = pd.DataFrame(self.real_coordinates, columns=['X_Value', 'Y_Value'])
            with open(file_path, 'w') as f:
                f.write("# Raman Data Digitizer Export - Advanced Line Masking System\n")
                f.write(f"# Detection Method: {self.method_var.get()}\n")
                f.write(f"# Method 1 Points: {len(self.method1_points)}\n")
                f.write(f"# Method 2 Points: {len(self.method2_points)}\n")
                if self.line_masking.detected_lines:
                    f.write(f"# Lines Detected: {len(self.line_masking.detected_lines)}\n")
                    f.write(f"# Current Line: {self.line_masking.current_line_index + 1}\n")
                f.write(f"# Interpolation: {'Enabled' if self.use_interpolation.get() else 'Disabled'}\n")
                f.write(f"# Exclusion Zones: {len(self.exclusion_zones)}\n")
                if self.best_fit_equation:
                    f.write(f"# Best Fit Line: {self.best_fit_equation}\n")
                    f.write(f"# R-squared: {self.statistics.get('r_squared', 0):.6f}\n")
                f.write("#\n")
            df.to_csv(file_path, index=False, mode='a')
            messagebox.showinfo("Success", f"Data successfully exported to\n{file_path}")
        except Exception as e: messagebox.showerror("Export Error", f"Failed to export data.\nError: {e}")

# Main execution block - this runs when the script is executed directly
if __name__ == "__main__":
    # Create the main tkinter window
    root = tk.Tk()
    
    # Create and run the Enhanced Raman Data Digitizer application with Line Masking
    app = RamanDataDigitizer(root)
    
    # Start the tkinter event loop - this keeps the application running
    # and responsive to user interactions until the window is closed
    root.mainloop()