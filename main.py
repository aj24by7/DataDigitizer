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
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter, find_peaks


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

    def _create_exclusion_mask(self, img_shape):
        """Create a mask that excludes regions defined in exclusion_zones."""
        if not self.parent.exclusion_zones:
            return None

        # Create mask where excluded regions are 0, allowed regions are 255
        mask = np.ones(img_shape, dtype=np.uint8) * 255

        for (x0, y0, x1, y1) in self.parent.exclusion_zones:
            # Ensure coordinates are within image bounds
            x0 = max(0, min(x0, img_shape[1] - 1))
            y0 = max(0, min(y0, img_shape[0] - 1))
            x1 = max(0, min(x1, img_shape[1] - 1))
            y1 = max(0, min(y1, img_shape[0] - 1))

            # Set excluded region to 0
            mask[y0:y1 + 1, x0:x1 + 1] = 0

        return mask

    def _detect_lines_hough(self, gray, edge_thresh=30, exclusion_mask=None):
        """Use Hough Line Transform to detect straight line segments, respecting exclusion zones."""
        # Enhanced edge detection with adjustable threshold
        edges = cv2.Canny(gray, edge_thresh, edge_thresh * 3, apertureSize=3)

        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            edges = cv2.bitwise_and(edges, exclusion_mask)

        # Apply HoughLinesP for line segment detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=max(15, edge_thresh // 2),
                                minLineLength=self.detection_parameters['min_line_length'],
                                maxLineGap=self.detection_parameters['max_line_gap'])

        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Sample points along each detected line
                num_points = max(5, int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 3))
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
        kernel = np.ones((2, 2), np.uint8)
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
                            (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[
                                y, x] > 0)):
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
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

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
                    (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[
                        y, x] > 0)):
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
                if abs(y_sorted[i][1] - y_sorted[i - 1][1]) < 20:  # Same line
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
            kernel = np.ones((3, 3), np.uint8)
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
            color_map = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
                         'orange': (255, 165, 0), 'purple': (128, 0, 128), 'brown': (165, 42, 42),
                         'pink': (255, 192, 203), 'gray': (128, 128, 128)}
            rgb_color = color_map.get(color, (255, 255, 255))

            # Draw a simple horizontal line
            preview_rgb[12:18, 10:90] = rgb_color
            return Image.fromarray(preview_rgb)

        # Extract region and resize for preview
        try:
            line_region = mask[min_y:max_y + 1, min_x:max_x + 1]

            # Resize to thumbnail size
            thumbnail_size = (100, 30)
            line_region_resized = cv2.resize(line_region, thumbnail_size)

            # Convert to RGB and apply color
            preview_rgb = np.zeros((*thumbnail_size[::-1], 3), dtype=np.uint8)
            color_map = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
                         'orange': (255, 165, 0), 'purple': (128, 0, 128), 'brown': (165, 42, 42),
                         'pink': (255, 192, 203), 'gray': (128, 128, 128)}

            rgb_color = color_map.get(color, (255, 255, 255))
            for c in range(3):
                preview_rgb[:, :, c] = (line_region_resized / 255) * rgb_color[c]

            return Image.fromarray(preview_rgb)
        except Exception as e:
            print(f"Error creating detailed preview: {e}")
            # Fallback to simple line preview
            preview_rgb = np.zeros((30, 100, 3), dtype=np.uint8)
            color_map = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
                         'orange': (255, 165, 0), 'purple': (128, 0, 128), 'brown': (165, 42, 42),
                         'pink': (255, 192, 203), 'gray': (128, 128, 128)}
            rgb_color = color_map.get(color, (255, 255, 255))
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
                self.parent.debug_label.config(
                    text=f"Debug: Applying {len(self.parent.exclusion_zones)} exclusion zones...")
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
                messagebox.showinfo("No Lines",
                                    "No distinct lines detected. Try:\n1. Lower edge threshold\n2. Smaller min length\n3. Use Simple Detection button\n4. First run Auto-Exclude Text if not done")
                return

            # Cluster points into separate lines using custom clustering
            self.parent.debug_label.config(text="Debug: Clustering points into lines...")
            self.parent.root.update_idletasks()
            separated_lines = self._separate_lines_clustering(all_line_points)
            print(f"Clustering produced {len(separated_lines)} lines")

            if not separated_lines:
                self.parent.debug_label.config(text="Debug: Clustering failed. Try larger cluster distance.")
                messagebox.showinfo("No Lines",
                                    "Clustering failed to separate lines. Try:\n1. Increase cluster distance\n2. Decrease min points\n3. Use Simple Detection")
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

    def _separate_lines_by_color_and_position(self, points_with_colors):
        """
        Separates lines using DBSCAN-like clustering based on both spatial
        position (x, y) and color (r, g, b). This is highly effective for
        separating lines of different colors that are touching or overlapping.
        """
        if not points_with_colors:
            return []

        self.parent.debug_label.config(text="Debug: Clustering by color and position...")
        self.parent.root.update_idletasks()
        print("Separating lines using color and position clustering...")

        points_array = np.array(points_with_colors, dtype=np.float32)

        # Normalize the features to give them comparable scales.
        max_x = self.parent.image.width
        max_y = self.parent.image.height

        normalized_points = points_array.copy()
        normalized_points[:, 0] /= max_x  # Normalize x
        normalized_points[:, 1] /= max_y  # Normalize y
        normalized_points[:, 2:] /= 255.0  # Normalize r, g, b

        # MODIFIED: Weights to balance the importance of position vs. color.
        # Color is now much more important to separate touching lines of different colors.
        pos_weight = 1.0
        color_weight = 5.0

        normalized_points[:, 0] *= pos_weight
        normalized_points[:, 1] *= pos_weight
        normalized_points[:, 2:] *= color_weight

        # Adapt the UI's 'cluster_dist' to our normalized space.
        normalized_eps = self.detection_parameters['clustering_eps'] / np.mean([max_x, max_y]) * pos_weight
        min_samples = self.detection_parameters['min_samples']

        clusters = []
        visited = np.zeros(len(normalized_points), dtype=bool)

        for i in range(len(normalized_points)):
            if visited[i]:
                continue

            # Find neighbors using Euclidean distance in the weighted, normalized space
            distances = np.sqrt(np.sum((normalized_points - normalized_points[i]) ** 2, axis=1))
            neighbors_indices = np.where(distances <= normalized_eps)[0]

            if len(neighbors_indices) < min_samples:
                continue  # Not a core point, treat as noise

            # Start a new cluster
            new_cluster = []
            queue = list(neighbors_indices)

            while queue:
                current_idx = queue.pop(0)
                if visited[current_idx]:
                    continue

                visited[current_idx] = True
                new_cluster.append(points_array[current_idx])

                # Find neighbors of the current point to expand the cluster
                current_point_norm = normalized_points[current_idx]
                distances = np.sqrt(np.sum((normalized_points - current_point_norm) ** 2, axis=1))
                expansion_indices = np.where(distances <= normalized_eps)[0]

                for neighbor_idx in expansion_indices:
                    if not visited[neighbor_idx]:
                        queue.append(neighbor_idx)

            if len(new_cluster) >= min_samples:
                # Convert back to original points (just x, y) for mask creation
                cluster_points_xy = np.array(new_cluster)[:, :2]
                clusters.append(cluster_points_xy)

        print(f"Color clustering produced {len(clusters)} distinct lines.")
        return clusters


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
                tab_x + tab_width // 2, tab_y + tab_height // 2,
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
            pass
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
            self.stats_text.insert(tk.END,
                                   "No data available for statistics.\n\nExtract data first to see:\n• Point counts\n• Statistical measures\n• Best fit analysis\n• Extrema detection")
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


class ColorIndex:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.analysis_results = {}

    def analyze_image_colors(self, image_array):
        """
        Scans the image, counts colors, and determines the dominant line color
        and graph type with improved background purging and new logic.
        MODIFIED to ensure all color values are stored as standard Python tuples of integers.
        """
        if image_array is None:
            return None

        pixels = image_array.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        color_counts = {}
        total_non_background_pixels = 0

        # --- NEW: More aggressive background color purging ---
        # Purges colors where all channels are above 220 (~86% white)
        # This catches light grays and off-whites effectively.
        for i, color_np in enumerate(unique_colors):
            # First, check for background color using the NumPy array for efficiency.
            if np.all(color_np > 220):
                continue

            # MODIFICATION: Convert the NumPy array (e.g., array([10, 20, 30], dtype=uint8))
            # to a standard Python tuple of integers (e.g., (10, 20, 30)).
            # This is the key step to prevent "np.uint8" from appearing in the UI.
            color_key = tuple(int(c) for c in color_np)

            color_counts[color_key] = counts[i]
            total_non_background_pixels += counts[i]

        if total_non_background_pixels == 0:
            return {'dominant_color': None, 'graph_type': 'Unknown', 'palette': []}

        # Since the keys of color_counts are now clean tuples, everything derived from it will be clean.
        sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)

        palette = []
        for color, count in sorted_colors:
            percentage = (count / total_non_background_pixels) * 100
            if percentage > 0.5:
                # 'color' is already a clean tuple here because it's a key from color_counts.
                palette.append({'rgb': color, 'percentage': percentage})

        dominant_color = None
        if palette:
            # Rule 1: Black dominance (80%+)
            # MODIFICATION: Ensure we use a standard tuple for the key.
            black_color_key = (0, 0, 0)
            if black_color_key in color_counts:
                black_percentage = (color_counts[black_color_key] / total_non_background_pixels) * 100
                if black_percentage >= 80.0:
                    dominant_color = black_color_key

            # Rule 2: Any color > 20%
            if dominant_color is None and palette[0]['percentage'] > 20.0:
                dominant_color = palette[0]['rgb']  # This is now guaranteed to be a clean tuple.

            # Rule 3: Pick the most populous as a fallback
            if dominant_color is None:
                dominant_color = palette[0]['rgb']  # This is also a clean tuple.

        # --- NEW: Updated Graph Type Logic ---
        # If the most dominant color is over 40%, it's likely a simple linear plot.
        # Otherwise, multiple colors have significant presence, suggesting overlap.
        graph_type = "Overlapping"
        if palette and palette[0]['percentage'] >= 40.0:
            graph_type = "Stacking/Linear"

        self.analysis_results = {
            'dominant_color': dominant_color,  # Guaranteed to be a clean tuple or None.
            'graph_type': graph_type,
            'palette': palette[:10]  # Palette contains clean tuples.
        }
        return self.analysis_results


class RamanDataDigitizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectra Data Digitizer - Advanced Line Masking System")
        self.root.geometry("1400x900")

        # Modern Color Scheme
        self.colors = {
            'bg_main': '#fafbfc',  # Very light gray background
            'bg_panel': '#ffffff',  # Pure white for panels
            'bg_secondary': '#f8f9fa',  # Slightly darker light gray
            'primary': '#4f46e5',  # Modern indigo
            'primary_hover': '#4338ca',  # Darker indigo for hover
            'success': '#10b981',  # Modern green
            'warning': '#f59e0b',  # Modern amber
            'danger': '#ef4444',  # Modern red
            'info': '#06b6d4',  # Modern cyan
            'secondary': '#6b7280',  # Modern gray
            'text_primary': '#111827',  # Dark gray for text
            'text_secondary': '#6b7280',  # Medium gray for secondary text
            'border': '#e5e7eb',  # Light border
            'accent': '#8b5cf6'  # Purple accent
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
        self.color_tolerance = 5
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
        self.dragging_line_index = None  # 0:top, 1:right, 2:bottom, 3:left

        # --- Detection Method Results ---
        self.method1_points = []  # Original method (main4)
        self.method2_points = []  # New method (from main3)
        self.current_method = "auto"  # "auto", "method1", "method2"

        # --- Line Masking System - NEW FEATURE ---
        self.line_masking = LineMaskingSystem(self)
        # --- Color Index System - NEW FEATURE ---
        self.color_index = ColorIndex(self)
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

        # File Menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.load_image)
        file_menu.add_command(label="Load JSON...", command=self.load_json)
        file_menu.add_command(label="Paste from Clipboard", command=self.paste_image)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All", command=self.clear_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Export Menu (NEW)
        export_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export to CSV...", command=self.export_csv)
        export_menu.add_command(label="Export to JSON...", command=self.export_json)
        export_menu.add_separator()
        export_menu.add_command(label="Export Plot as PNG...", command=self.export_plot_as_png)

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

        # --- Right Panel (Controls) - Now scrollable to fit all screens ---
        right_outer_frame = tk.Frame(main_frame, bg=self.colors['bg_main'], width=380)
        right_outer_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_outer_frame.pack_propagate(False)

        canvas_right = tk.Canvas(right_outer_frame, bg=self.colors['bg_main'], highlightthickness=0)
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

        def _on_mousewheel(event):
            canvas_right.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mouse(event):
            canvas_right.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mouse(event):
            canvas_right.unbind_all("<MouseWheel>")

        right_outer_frame.bind('<Enter>', _bind_mouse)
        right_outer_frame.bind('<Leave>', _unbind_mouse)

        right_panel = scrollable_frame

        # ========== STEP 1: ZOOM PREVIEW AND CONTROLS ==========
        zoom_outer_frame = tk.LabelFrame(right_panel, text="🔍 Step 1: Zoom Preview",
                                         bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                         font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        zoom_outer_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

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

        self.zoom_label = tk.Label(zoom_controls, text="100%",
                                   bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                                   font=('Segoe UI', 8))
        self.zoom_label.pack(side=tk.RIGHT, padx=8)

        self.zoom_canvas = tk.Canvas(zoom_outer_frame, width=250, height=200,
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

        # --- MODIFICATION: Tolerance spinbox and label have been removed ---
        tk.Button(color_btn_frame, text="Pick Color", command=self.enter_pick_color_mode,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.color_display = tk.Label(color_frame, text="No Color Selected",
                                      bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                      relief=tk.FLAT, bd=1, height=1, font=('Segoe UI', 8))
        self.color_display.pack(fill=tk.X, pady=(0, 6), padx=8)

        # ========== NEW: COLOR INDEX ANALYSIS ==========
        color_index_frame = tk.LabelFrame(right_panel, text="🎨 Color Index Analysis",
                                          bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                          font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        color_index_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        index_content = tk.Frame(color_index_frame, bg=self.colors['bg_panel'])
        index_content.pack(fill=tk.X, padx=8, pady=6)

        # Graph Type Display
        graph_type_frame = tk.Frame(index_content, bg=self.colors['bg_panel'])
        graph_type_frame.pack(fill=tk.X, pady=2)
        tk.Label(graph_type_frame, text="Graph Type:", bg=self.colors['bg_panel'],
                 font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT)
        self.graph_type_label = tk.Label(graph_type_frame, text="N/A", bg=self.colors['bg_secondary'],
                                         fg=self.colors['text_secondary'], relief=tk.FLAT, bd=1,
                                         font=('Segoe UI', 8))
        self.graph_type_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Palette Display
        self.palette_frame = tk.Frame(index_content, bg=self.colors['bg_panel'])
        self.palette_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Label(self.palette_frame, text="Detected Colors:", bg=self.colors['bg_panel'],
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w')

        # ========== STEP 3: CALIBRATION ==========
        calib_frame = tk.LabelFrame(right_panel, text="📐 Step 3: Calibration",
                                    bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                    font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        calib_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        calib_button_frame = tk.Frame(calib_frame, bg=self.colors['bg_panel'])
        calib_button_frame.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(calib_button_frame, text="Auto Calibrate", command=self.auto_calibrate,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(calib_button_frame, text="Manual", command=self.enter_manual_calibrate_mode,
                  bg=self.colors['info'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(calib_button_frame, text="Clear", command=self.clear_calibration,
                  bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        axis_frame = tk.LabelFrame(calib_frame, text="Axis Values",
                                   bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                                   font=('Segoe UI', 8), relief=tk.FLAT, bd=1)
        axis_frame.pack(fill=tk.X, pady=6, padx=8)

        axis_grid = tk.Frame(axis_frame, bg=self.colors['bg_panel'])
        axis_grid.pack(fill=tk.X, pady=4, padx=6)

        tk.Label(axis_grid, text="X:", bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 8, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 4))
        self.x_min_var = tk.StringVar(value="0")
        tk.Entry(axis_grid, textvariable=self.x_min_var, width=6, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=2)
        tk.Label(axis_grid, text="to", bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                 font=('Segoe UI', 8)).grid(row=0, column=2, padx=4)
        self.x_max_var = tk.StringVar(value="100")
        tk.Entry(axis_grid, textvariable=self.x_max_var, width=6, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=2)

        tk.Label(axis_grid, text="Y:", bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 8, 'bold')).grid(row=1, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.y_min_var = tk.StringVar(value="0")
        tk.Entry(axis_grid, textvariable=self.y_min_var, width=6, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=1, column=1, padx=2, pady=(4, 0))
        tk.Label(axis_grid, text="to", bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                 font=('Segoe UI', 8)).grid(row=1, column=2, padx=4, pady=(4, 0))
        self.y_max_var = tk.StringVar(value="100")
        tk.Entry(axis_grid, textvariable=self.y_max_var, width=6, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).grid(row=1, column=3, padx=2, pady=(4, 0))

        # ========== STEP 4: SMART EXCLUSION TOOLS ==========
        exclusion_frame = tk.LabelFrame(right_panel, text="🚫 Step 4: Smart Exclusion Tools",
                                        bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                        font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        exclusion_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        auto_buttons_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        auto_buttons_frame.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(auto_buttons_frame, text="🤖 Auto-Exclude Text", command=self.auto_create_text_exclusions,
                  bg=self.colors['success'], fg='white', font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(auto_buttons_frame, text="🔲 Auto-Exclude Borders", command=self.auto_exclude_borders,
                  bg=self.colors['info'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

        manual_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        manual_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(manual_frame, text="Define Zone", command=self.enter_exclusion_mode,
                  bg=self.colors['warning'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(manual_frame, text="Delete Zone", command=self.enter_zone_deletion_mode,
                  bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(manual_frame, text="Clear All", command=self.clear_exclusion_zones,
                  bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

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

        params_frame = tk.Frame(line_masking_frame, bg=self.colors['bg_panel'])
        params_frame.pack(fill=tk.X, pady=6, padx=8)

        param_row1 = tk.Frame(params_frame, bg=self.colors['bg_panel'])
        param_row1.pack(fill=tk.X, pady=2)

        tk.Label(param_row1, text="Edge Threshold:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=0, sticky='w')
        self.edge_threshold_var = tk.StringVar(value="30")
        tk.Spinbox(param_row1, from_=10, to=150, textvariable=self.edge_threshold_var,
                   width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                   bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=(4, 12))

        tk.Label(param_row1, text="Min Length:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=2, sticky='w')
        self.min_length_var = tk.StringVar(value="30")
        tk.Spinbox(param_row1, from_=10, to=200, textvariable=self.min_length_var,
                   width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                   bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=4)

        param_row2 = tk.Frame(params_frame, bg=self.colors['bg_panel'])
        param_row2.pack(fill=tk.X, pady=2)

        tk.Label(param_row2, text="Cluster Distance:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=0, sticky='w')
        self.cluster_dist_var = tk.StringVar(value="15")
        tk.Spinbox(param_row2, from_=5, to=50, textvariable=self.cluster_dist_var,
                   width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                   bg=self.colors['bg_secondary']).grid(row=0, column=1, padx=(4, 12))

        tk.Label(param_row2, text="Min Points:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).grid(row=0, column=2, sticky='w')
        self.min_points_var = tk.StringVar(value="20")
        tk.Spinbox(param_row2, from_=10, to=100, textvariable=self.min_points_var,
                   width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                   bg=self.colors['bg_secondary']).grid(row=0, column=3, padx=4)

        detect_frame = tk.Frame(line_masking_frame, bg=self.colors['bg_panel'])
        detect_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(detect_frame, text="🔍 Simple Detection", command=self.simple_line_detection,
                  bg=self.colors['warning'], fg='white', font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(detect_frame, text="🎯 Advanced Detection", command=self.detect_lines,
                  bg=self.colors['accent'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

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

        self.line_preview_canvas = tk.Canvas(line_masking_frame, width=150, height=25,
                                             bg=self.colors['bg_secondary'], highlightthickness=1,
                                             highlightbackground=self.colors['border'])
        self.line_preview_canvas.pack(pady=4, padx=8)

        self.debug_label = tk.Label(line_masking_frame, text="Debug: Ready",
                                    bg=self.colors['bg_panel'], fg=self.colors['info'],
                                    font=('Segoe UI', 7), wraplength=250)
        self.debug_label.pack(pady=4, padx=8)

        tk.Button(line_masking_frame, text="Clear Detection", command=self.clear_line_detection,
                  bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=(0, 6), padx=8)

        # ========== STEP 6: INTERPOLATION (NEW) ==========
        interpolation_frame = tk.LabelFrame(right_panel, text="🔬 Step 6: Interpolation",
                                            bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                            font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        interpolation_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        interp_button_frame = tk.Frame(interpolation_frame, bg=self.colors['bg_panel'])
        interp_button_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(interp_button_frame, text="Simple Interpolation", command=self.simple_interpolation,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        # Placeholder for complex interpolation
        tk.Button(interp_button_frame, text="Complex Interpolation",
                  command=lambda: messagebox.showinfo("Not Implemented", "Complex interpolation is not yet available."),
                  bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # ========== STEP 7: DATA EXTRACTION (Renumbered) ==========
        extract_frame = tk.LabelFrame(right_panel, text="📊 Step 7: Data Extraction",
                                      bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                      font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        extract_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        mode_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        mode_frame.pack(fill=tk.X, pady=(0, 6), padx=8)

        tk.Label(mode_frame, text="Mode:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Step between points")
        mode_dropdown = ttk.Combobox(mode_frame, textvariable=self.mode_var, width=12,
                                     state="readonly", font=('Segoe UI', 8))
        mode_dropdown['values'] = ("Step between points", "Number of points")
        mode_dropdown.pack(side=tk.RIGHT)

        value_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        value_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(value_frame, text="Value:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.value_var = tk.StringVar(value="10")
        tk.Entry(value_frame, textvariable=self.value_var, width=8, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).pack(side=tk.RIGHT)

        action_button_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        action_button_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(action_button_frame, text="📊 Extract Data", command=self.extract_data_streamlined,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(action_button_frame, text="🔍 Find Peak/Valley", command=self.find_extrema,
                  bg=self.colors['accent'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

        toggle_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        toggle_frame.pack(fill=tk.X, pady=(0, 6), padx=8)

        self.show_best_fit = tk.BooleanVar(value=True)
        tk.Checkbutton(toggle_frame, text="Show Best Fit Line", variable=self.show_best_fit,
                       command=self.toggle_best_fit_line, bg=self.colors['bg_panel'],
                       fg=self.colors['text_primary'], font=('Segoe UI', 8),
                       selectcolor=self.colors['bg_secondary']).pack(side=tk.LEFT)

        results_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        results_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(results_frame, text="Results:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_primary'], font=('Segoe UI', 8, 'bold')).pack(anchor='w')
        self.method_results_label = tk.Label(results_frame, text="No data extracted",
                                             bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                             relief=tk.FLAT, bd=1, wraplength=280, font=('Segoe UI', 7),
                                             justify=tk.LEFT, anchor='w')
        self.method_results_label.pack(fill=tk.X, pady=(2, 0))

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(right_panel, orient='horizontal', mode='determinate')

        self.create_color = self.colors['primary']
        self.clear_color = self.colors['danger']
        self.export_color = self.colors['success']
        self.calib_color = self.colors['info']
        self.peak_color = '#ff00ff'
        self.valley_color = '#ffa500'

    # === NEW: Line Masking System Functions ===
    def detect_lines(self):
        """Trigger line detection using the advanced masking system."""
        self.line_masking.detect_all_lines()
        self.update_line_info_display()
        self.update_line_preview()

    def simple_line_detection(self):
        """
        Simple line detection for spectral data.
        MODIFIED to merge all detected line-like fragments into a single, unified mask
        to ensure the entire data curve is processed at once.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            self.debug_label.config(text="Debug: Finding line fragments...")
            self.root.update_idletasks()

            img_array = np.array(self.image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            exclusion_mask = self.line_masking._create_exclusion_mask(gray.shape)

            self.line_masking.detected_lines.clear()
            self.line_masking.line_colors.clear()
            self.line_masking.line_previews.clear()

            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 7)

            if exclusion_mask is not None:
                binary = cv2.bitwise_and(binary, exclusion_mask)

            kernel_small = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

            unified_mask = np.zeros_like(binary)
            found_fragments = 0
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                aspect_ratio = w / h if h > 0 else 0

                if (area > 80 and (aspect_ratio > 1.5 or area > 300) and w > 15 and h > 2):
                    component_mask = (labels == i).astype(np.uint8) * 255
                    unified_mask = cv2.bitwise_or(unified_mask, component_mask)
                    found_fragments += 1

            if found_fragments == 0:
                self.debug_label.config(text="Debug: No line fragments found.")
                messagebox.showinfo("No Lines", "Simple detection found no line fragments to process.")
                return

            # --- MODIFICATION: Using a larger kernel to more aggressively connect fragments ---
            kernel_large = np.ones((9, 9), np.uint8)
            unified_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

            unified_mask = cv2.dilate(unified_mask, kernel_small, iterations=1)

            self.line_masking.detected_lines = [unified_mask]

            self.line_masking._generate_line_visualization()

            self.debug_label.config(text=f"Debug: Found and merged {found_fragments} fragments.")
            messagebox.showinfo("Simple Detection Complete",
                                f"Detected and merged {found_fragments} fragments into a single line.\n" +
                                "You can now proceed to 'Extract Data'.")

            self.update_line_info_display()
            self.update_line_preview()
            self.redraw_overlays()

        except Exception as e:
            self.debug_label.config(text=f"Debug: Simple detection failed - {str(e)[:30]}...")
            messagebox.showerror("Simple Detection Error", f"An error occurred during simple detection: {e}")

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
            if self.line_masking.line_colors and self.line_masking.current_line_index < len(
                    self.line_masking.line_colors):
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
                self.line_preview_canvas.create_image(75, 12, anchor=tk.CENTER, image=preview_photo)
                # Keep reference to prevent garbage collection
                self.line_preview_canvas.preview_photo = preview_photo
        else:
            # Show a simple line representation if no preview available
            if self.line_masking.detected_lines:
                self.line_preview_canvas.create_line(10, 12, 140, 12, fill='red', width=3)

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
        if hasattr(self, 'stats_panel') and hasattr(self.stats_panel,
                                                    'window') and self.stats_panel.window and self.stats_panel.window.winfo_exists():
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
            with open(file_path, 'r') as f:
                data = json.load(f)
            if not ('x' in data and 'y' in data): raise ValueError("JSON must contain 'x' and 'y' keys.")
            if len(data['x']) != len(data['y']): raise ValueError("JSON 'x' and 'y' arrays must have the same length.")
            self.clear_all()
            self.real_coordinates = list(zip(data['x'], data['y']))
            messagebox.showinfo("JSON Loaded", f"Successfully loaded {len(self.real_coordinates)} points from JSON.")
            self.calculate_statistics()
            self.calculate_best_fit()
        except Exception as e:
            messagebox.showerror("JSON Load Error", f"Failed to load or parse JSON file.\nError: {e}")

    def paste_image(self, event=None):
        try:
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                self.process_new_image(img)
                print("Successfully pasted image from clipboard")
            else:
                messagebox.showwarning("Paste Error",
                                       "No image found in clipboard or clipboard content is not an image.")
        except Exception as e:
            messagebox.showerror("Paste Error", f"Could not paste image from clipboard.\nError: {e}")
            print(f"Paste error: {e}")

    def process_new_image(self, img):
        try:
            print("Starting image processing...")
            self.clear_all()
            self.original_image = img.convert("RGB")  # Store original
            self.image = self.original_image.copy()  # Working copy
            self.zoom_level = 1.0
            if hasattr(self, 'zoom_label'):
                self.zoom_label.config(text="100%")
            self.image_array = np.array(self.image)
            self.display_image_on_canvas()
            print(f"Image processed successfully. Size: {self.image.size}")

            # --- NEW: Trigger color analysis upon loading a new image ---
            self.run_color_analysis()

        except Exception as e:
            messagebox.showerror("Image Processing Error", f"Could not process image.\nError: {e}")
            print(f"Image processing error: {e}")

    def run_color_analysis(self):
        """
        Triggers the color analysis and updates the UI with the results.
        """
        if self.image_array is None:
            return

        results = self.color_index.analyze_image_colors(self.image_array)

        if results and results.get('dominant_color'):
            # Automatically set the selected color
            dominant_color = results['dominant_color']
            self.selected_color = dominant_color
            color_hex = f'#{dominant_color[0]:02x}{dominant_color[1]:02x}{dominant_color[2]:02x}'

            # Update the main color display
            self.color_display.config(text=f"Auto-Selected: RGB {dominant_color}", bg=color_hex)

            # Update the color index UI
            self.update_color_index_ui(results)
        else:
            # Handle case where no dominant color is found
            self.update_color_index_ui({'graph_type': 'Unknown', 'palette': []})

    def set_selected_color_from_palette(self, rgb_color):
        """
        Sets the application's selected color from the color index palette.
        """
        self.selected_color = rgb_color
        color_hex = f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'
        self.color_display.config(text=f"RGB: {self.selected_color}", bg=color_hex)
        print(f"Color switched to: {rgb_color}")

    def update_color_index_ui(self, results):
        """
        Updates the Color Index section of the UI with a redesigned, interactive layout.
        """
        # Clear previous palette display
        for widget in self.palette_frame.winfo_children():
            # More robustly clear the frame that holds the color rows
            if isinstance(widget, tk.Frame) or (hasattr(widget, 'winfo_class') and widget.winfo_class() == 'TLabel'):
                if widget.master == self.palette_frame:  # Only destroy direct children
                    if not isinstance(widget.winfo_children(), tuple):  # Don't destroy the header
                        widget.destroy()

        # Update graph type
        self.graph_type_label.config(text=results.get('graph_type', 'N/A'))

        palette = results.get('palette', [])
        if not palette:
            # Check if a "no colors" label already exists to prevent duplicates
            if not any(isinstance(w, tk.Label) and "No significant" in w.cget("text") for w in
                       self.palette_frame.winfo_children()):
                no_color_label = tk.Label(self.palette_frame, text="No significant colors detected.",
                                          bg=self.colors['bg_panel'], font=('Segoe UI', 8))
                no_color_label.pack(anchor='w')
            return

        # Create a header if it doesn't exist
        if not any(
                isinstance(w, tk.Frame) and len(w.winfo_children()) == 2 for w in self.palette_frame.winfo_children()):
            header_frame = tk.Frame(self.palette_frame, bg=self.colors['bg_secondary'])
            header_frame.pack(fill=tk.X, pady=(4, 2))
            tk.Label(header_frame, text="Color", bg=self.colors['bg_secondary'], font=('Segoe UI', 8, 'bold')).pack(
                side=tk.LEFT, padx=5, pady=2)
            tk.Label(header_frame, text="Details", bg=self.colors['bg_secondary'], font=('Segoe UI', 8, 'bold')).pack(
                side=tk.LEFT, padx=15, pady=2)

        # Display the new color palette with interactive buttons
        for color_info in palette:
            rgb = color_info['rgb']
            percentage = color_info['percentage']
            color_hex = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

            row = tk.Frame(self.palette_frame, bg=self.colors['bg_panel'])
            row.pack(fill=tk.X, pady=2)

            # Color swatch
            swatch = tk.Label(row, text="", bg=color_hex, width=3, relief=tk.SUNKEN)
            swatch.pack(side=tk.LEFT, padx=(5, 10), ipady=5)

            # Color info label (using a fixed-width font for alignment)
            info_text = f"({percentage:>4.1f}%) RGB: {str(rgb)}"
            info_label = tk.Label(row, text=info_text, bg=self.colors['bg_panel'],
                                  font=('Courier New', 9))
            info_label.pack(side=tk.LEFT, anchor='w', expand=True)

            # Select Button
            select_btn = tk.Button(row, text="Select",
                                   command=lambda c=rgb: self.set_selected_color_from_palette(c),
                                   bg=self.colors['info'], fg='white', font=('Segoe UI', 7, 'bold'),
                                   relief=tk.FLAT, bd=0, activebackground=self.colors['primary_hover'])
            select_btn.pack(side=tk.RIGHT, padx=5)

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
            if hasattr(self.stats_panel,
                       'window') and self.stats_panel.window and self.stats_panel.window.winfo_exists():
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
        self.draw_calibration_grid();
        self.draw_calibration_points()
        self.draw_extracted_points();
        self.draw_best_fit_line()
        self.draw_exclusion_zones();
        self.draw_extrema_points()
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
            self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=color, outline='', tags="line_mask")

    # --- Mode Management ---
    def enter_pick_color_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "pick_color";
        self.canvas.config(cursor="crosshair")

    def enter_manual_calibrate_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "manual_calibrate";
        self.canvas.config(cursor="crosshair")
        messagebox.showinfo("Manual Calibration", "Click corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

    def enter_exclusion_mode(self):
        if self.image is None: messagebox.showwarning("Warning", "Please load an image first."); return
        self.current_mode = "exclusion";
        self.canvas.config(cursor="tcross")

    def escape_mode(self, event=None):
        self.current_mode = "normal";
        self.canvas.config(cursor="arrow")
        if self.temp_rect_id: self.canvas.delete(self.temp_rect_id)
        self.temp_rect_start = self.temp_rect_id = self.dragging_line_index = None

    # --- Canvas Events (Enhanced with Smart Exclusion Support) ---
    def on_canvas_click(self, event):
        if self.image is None: return
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

        line_index = self.get_line_at_pos(x, y)
        if line_index is not None:
            self.dragging_line_index = line_index;
            self.current_mode = "adjust_calibration";
            return

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
            self.temp_rect_id = self.canvas.create_rectangle(x, y, x, y, outline='red', width=2, dash=(5, 3))

    def on_canvas_motion(self, event):
        if self.image is None: return
        self.mouse_x, self.mouse_y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if self.current_mode == "normal":
            line_index = self.get_line_at_pos(self.mouse_x, self.mouse_y)
            if line_index in [0, 2]:
                self.canvas.config(cursor="sb_v_double_arrow")
            elif line_index in [1, 3]:
                self.canvas.config(cursor="sb_h_double_arrow")
            else:
                self.canvas.config(cursor="arrow")
        self.update_zoom_preview()

    def on_canvas_drag(self, event):
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if self.current_mode == "adjust_calibration" and self.dragging_line_index is not None:
            self.adjust_calibration_box(x, y)
        elif self.current_mode == "exclusion" and self.temp_rect_start:
            x0, y0 = self.temp_rect_start;
            self.canvas.coords(self.temp_rect_id, x0, y0, x, y)

    def on_canvas_release(self, event):
        if self.current_mode == "adjust_calibration":
            self.escape_mode()
        elif self.current_mode == "exclusion" and self.temp_rect_start:
            x0, y0 = self.temp_rect_start
            x1, y1 = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
            self.exclusion_zones.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_start = self.temp_rect_id = None
            self.escape_mode();
            self.draw_exclusion_zones()

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
                    w = min(self.image.width - x, w + 2 * padding)
                    h = min(self.image.height - y, h + 2 * padding)

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
            kernel = np.ones((3, 3), np.uint8)
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
                    roi_edges = edges[y:y + h, x:x + w]
                    edge_density = np.sum(roi_edges > 0) / area

                    # If this region has high edge density, it's likely text
                    if edge_density > 0.02:  # Threshold based on text characteristics
                        padding = 3
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(self.image.width - x, w + 2 * padding)
                        h = min(self.image.height - y, h + 2 * padding)

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
        sorted_regions = sorted(regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
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

    def is_point_inside_polygon(self, x, y, polygon):
        """
        Checks if a point (x, y) is inside a polygon using the ray casting algorithm.
        The polygon is a list of (x, y) tuples.
        """
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def filter_spectral_line_regions(self, regions):
        """
        Remove any exclusion regions that are inside the main calibrated graph area.
        This is a critical safety check to ensure we never exclude actual data.
        """
        if len(self.calibration_points) != 4:
            # If no calibration is set, we can't perform this check.
            return regions

        filtered_regions = []
        calib_poly = self.calibration_points

        for region in regions:
            x0, y0, x1, y1 = region

            # Calculate the center of the potential exclusion zone
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2

            # Check if the center of this region is inside the calibration polygon.
            # If it is, we assume it's part of the data/graph and should NOT be excluded.
            if self.is_point_inside_polygon(center_x, center_y, calib_poly):
                print(
                    f"INFO: Rejecting potential exclusion zone {region} because it is inside the calibrated graph area.")
                continue  # Skip this region, do not add it to the filtered list

            # If the region is outside, it's safe to add
            filtered_regions.append(region)

        return filtered_regions

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
                kernel = np.ones((3, 3), np.uint8)
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
                                min(self.image.height, y + border_thickness * 2)
                            ))

                            # Bottom border
                            detected_borders.append((
                                max(0, x - border_thickness),
                                max(0, y + h - border_thickness * 2),
                                min(self.image.width, x + w + border_thickness),
                                min(self.image.height, y + h + border_thickness)
                            ))

                            # Left border
                            detected_borders.append((
                                max(0, x - border_thickness),
                                max(0, y - border_thickness),
                                min(self.image.width, x + border_thickness * 2),
                                min(self.image.height, y + h + border_thickness)
                            ))

                            # Right border
                            detected_borders.append((
                                max(0, x + w - border_thickness * 2),
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
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

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

                    print(
                        f"Method 2 detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical border lines")

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
            # ADDED: Ask for confirmation if zones already exist
            if self.exclusion_zones:
                if not messagebox.askyesno("Confirm",
                                           "This will add new text-based exclusion zones to the existing ones. Continue?"):
                    return

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
        """Update the enlarged zoom preview window that follows the mouse."""
        if self.image is None:
            return

        new_zoom_size = 250
        new_canvas_width = 250
        new_canvas_height = 200
        crop_size = 80

        x_start = max(0, self.mouse_x - crop_size // 2)
        y_start = max(0, self.mouse_y - crop_size // 2)
        x_end = x_start + crop_size
        y_end = y_start + crop_size

        zoom_region = self.image.crop((x_start, y_start, x_end, y_end))
        zoom_enlarged = zoom_region.resize((new_zoom_size, new_zoom_size), Image.Resampling.NEAREST)

        overlay_img = Image.new('RGBA', (new_zoom_size, new_zoom_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)

        scale = new_zoom_size / crop_size

        for (x0, y0, x1, y1) in self.exclusion_zones:
            clipped_x0 = max(x0, x_start);
            clipped_y0 = max(y0, y_start)
            clipped_x1 = min(x1, x_end);
            clipped_y1 = min(y1, y_end)
            if clipped_x1 > clipped_x0 and clipped_y1 > clipped_y0:
                zoom_x0 = (clipped_x0 - x_start) * scale
                zoom_y0 = (clipped_y0 - y_start) * scale
                zoom_x1 = (clipped_x1 - x_start) * scale
                zoom_y1 = (clipped_y1 - y_start) * scale
                draw.rectangle([zoom_x0, zoom_y0, zoom_x1, zoom_y1], fill=(255, 0, 0, 80))

        if self.extracted_points:
            point_color = self.get_opposite_color(self.selected_color)
            for px, py in self.extracted_points:
                if x_start <= px < x_end and y_start <= py < y_end:
                    zoom_x = (px - x_start) * scale;
                    zoom_y = (py - y_start) * scale
                    draw.ellipse((zoom_x - 2, zoom_y - 2, zoom_x + 2, zoom_y + 2), fill=point_color)

        if self.best_fit_line_pixel_coords and self.show_best_fit.get():
            self.draw_line_in_zoom(draw, self.best_fit_line_pixel_coords, 'purple', crop_size, x_start, y_start,
                                   new_zoom_size)

        if len(self.calibration_points) == 4:
            points_in_zoom = []
            for i in range(4):
                px, py = self.calibration_points[i]
                zoom_x, zoom_y = (px - x_start) * scale, (py - y_start) * scale
                points_in_zoom.append((zoom_x, zoom_y))
                if 0 <= zoom_x < new_zoom_size and 0 <= zoom_y < new_zoom_size:
                    draw.ellipse((zoom_x - 4, zoom_y - 4, zoom_x + 4, zoom_y + 4), fill=self.calib_color,
                                 outline='black')
            for i in range(4):
                draw.line([points_in_zoom[i], points_in_zoom[(i + 1) % 4]], fill=self.calib_color, width=2)

        zoom_enlarged.paste(overlay_img, (0, 0), overlay_img)

        final_draw = ImageDraw.Draw(zoom_enlarged)
        center = new_zoom_size // 2
        final_draw.line([(center, 0), (center, new_zoom_size)], fill='red', width=1)
        final_draw.line([(0, center), (new_zoom_size, center)], fill='red', width=1)

        self.zoom_photo = ImageTk.PhotoImage(zoom_enlarged)
        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(new_canvas_width / 2, new_canvas_height / 2, anchor=tk.CENTER,
                                      image=self.zoom_photo)

        coords_text = f"({self.mouse_x}, {self.mouse_y})"
        self.zoom_canvas.create_text(5, 5, text=coords_text, anchor=tk.NW, fill='white', font=('Arial', 9, 'bold'))

    def draw_line_in_zoom(self, draw, line_coords, color, crop_size, x_start, y_start, zoom_size):
        x1, y1, x2, y2 = line_coords
        scale = zoom_size / crop_size
        zx1 = (x1 - x_start) * scale
        zy1 = (y1 - y_start) * scale
        zx2 = (x2 - x_start) * scale
        zy2 = (y2 - y_start) * scale
        draw.line([(zx1, zy1), (zx2, zy2)], fill=color, width=2)

    def select_color_at_point(self, x, y):
        if 0 <= y < self.image_array.shape[0] and 0 <= x < self.image_array.shape[1]:
            # MODIFICATION: Get the color as a NumPy array from the image data.
            color_np = self.image_array[y, x]
            # MODIFICATION: Convert the NumPy array to a standard Python tuple of integers.
            # This ensures the color is stored and displayed cleanly without dtype info.
            self.selected_color = tuple(int(c) for c in color_np)
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
            rect = cv2.minAreaRect(largest_contour);
            box = cv2.boxPoints(rect);
            box = box.astype(int)
            points = sorted(box, key=lambda p: p[1])
            top_points = sorted(points[:2], key=lambda p: p[0]);
            bottom_points = sorted(points[2:], key=lambda p: p[0])
            self.calibration_points = [tuple(top_points[0]), tuple(top_points[1]), tuple(bottom_points[1]),
                                       tuple(bottom_points[0])]
            messagebox.showinfo("Auto-Calibration", "Found graph area. You can now drag the sides to fine-tune.")
            self.redraw_overlays();
            self.escape_mode()
        except Exception as e:
            messagebox.showerror("Auto-Calibration Error",
                                 f"Could not detect a clear graph area. Using margins as a fallback.\nError: {e}")
            w, h = self.image.size;
            margin = min(w, h) // 10
            self.calibration_points = [(margin, margin), (w - margin, margin), (w - margin, h - margin),
                                       (margin, h - margin)]
            self.redraw_overlays()

    def add_calibration_point(self, x, y):
        if len(self.calibration_points) < 4:
            self.calibration_points.append((x, y));
            self.redraw_overlays()
        if len(self.calibration_points) == 4:
            self.escape_mode();
            messagebox.showinfo("Calibration Complete", "4 points set. Drag sides to adjust.")

    def clear_calibration(self):
        self.calibration_points = [];
        self.canvas.delete("calibration_points", "calibration_grid")

    def draw_calibration_points(self):
        self.canvas.delete("calibration_points")
        if not self.calibration_points: return
        for i, (x, y) in enumerate(self.calibration_points):
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=self.calib_color, outline='black', width=1,
                                    tags="calibration_points")

    def draw_calibration_grid(self):
        self.canvas.delete("calibration_grid")
        if len(self.calibration_points) != 4: return
        points = self.calibration_points
        for i in range(4):
            p1 = points[i];
            p2 = points[(i + 1) % 4]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=self.calib_color, width=2, dash=(5, 3),
                                    tags="calibration_grid")

    def get_line_at_pos(self, x, y, tolerance=5):
        if len(self.calibration_points) != 4: return None
        points = self.calibration_points
        lines = [(points[0], points[1]), (points[1], points[2]), (points[2], points[3]), (points[3], points[0])]
        for i, (p1, p2) in enumerate(lines):
            p1, p2 = np.array(p1), np.array(p2);
            p3 = np.array([x, y])
            if np.dot(p3 - p1, p2 - p1) >= 0 and np.dot(p3 - p2, p1 - p2) >= 0:
                dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                if dist < tolerance: return i
        return None

    def adjust_calibration_box(self, x, y):
        p = self.calibration_points;
        idx = self.dragging_line_index
        if idx == 0:
            p[0] = (p[0][0], y); p[1] = (p[1][0], y)
        elif idx == 1:
            p[1] = (x, p[1][1]); p[2] = (x, p[2][1])
        elif idx == 2:
            p[2] = (p[2][0], y); p[3] = (p[3][0], y)
        elif idx == 3:
            p[3] = (x, p[3][1]); p[0] = (x, p[0][1])
        self.redraw_overlays()

    # --- DATA EXTRACTION & ANALYSIS ---
    def clear_exclusion_zones(self):
        self.exclusion_zones = [];
        self.draw_exclusion_zones()

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

    def simple_interpolation(self):
        """
        Performs simple linear interpolation between extracted points that are far apart.
        MODIFIED to be more robust for steep/jagged lines. Instead of checking a single
        pixel for validation, it now checks a small neighborhood around the interpolated
        point, making it much more likely to find the line mask.
        """
        if not self.extracted_points:
            messagebox.showwarning("No Data", "Please extract data before running interpolation.")
            return

        current_mask = self.line_masking.get_current_line_mask()
        if current_mask is None:
            messagebox.showwarning("No Mask",
                                   "A line mask must be detected or selected to validate interpolated points.")
            return

        initial_point_count = len(self.extracted_points)
        # Ensure points are sorted by x-coordinate before starting
        points_to_process = sorted(self.extracted_points)

        # The new list will store original and interpolated points
        interpolated_points = []
        if not points_to_process:
            return  # Should not happen due to the check above, but for safety

        # Start with the first point, which is always valid
        interpolated_points.append(points_to_process[0])

        # --- NEW: Define a search radius for validation ---
        # This makes the check robust to pixelated lines. A radius of 2 checks a 5x5 area.
        validation_radius = 2

        # Iterate through adjacent points
        for i in range(len(points_to_process) - 1):
            p1 = np.array(points_to_process[i])
            p2 = np.array(points_to_process[i + 1])

            # Always add the start point of the segment
            # interpolated_points.append(p1)

            # Calculate Euclidean distance
            dist = np.linalg.norm(p2 - p1)

            # If distance is large enough, interpolate
            if dist > 5:
                # We want to add 4 points, so we need 5 segments (including the end point)
                num_new_points = 4
                for j in range(1, num_new_points + 1):  # j will be 1, 2, 3, 4
                    # Linear interpolation formula
                    t = j / (num_new_points + 1)
                    new_point = p1 * (1 - t) + p2 * t

                    ix, iy = int(round(new_point[0])), int(round(new_point[1]))

                    # --- MODIFIED VALIDATION LOGIC ---
                    # Check a small neighborhood around the interpolated point
                    is_valid = False
                    y_min = max(0, iy - validation_radius)
                    y_max = min(current_mask.shape[0], iy + validation_radius + 1)
                    x_min = max(0, ix - validation_radius)
                    x_max = min(current_mask.shape[1], ix + validation_radius + 1)

                    # If the neighborhood slice has any non-zero pixels, it's valid
                    if np.any(current_mask[y_min:y_max, x_min:x_max]):
                        is_valid = True

                    if is_valid:
                        interpolated_points.append((ix, iy))

            # Always add the next original point
            interpolated_points.append(tuple(p2))

        # Remove duplicates that might have been added and re-sort
        self.extracted_points = sorted(list(dict.fromkeys(interpolated_points)))

        final_point_count = len(self.extracted_points)
        points_added = final_point_count - initial_point_count

        if points_added > 0:
            messagebox.showinfo("Interpolation Complete", f"Successfully added {points_added} new data points.")

            # Update everything that depends on the extracted points
            self.convert_to_real_coordinates()
            self.redraw_overlays()  # This calls draw_extracted_points
            self.calculate_statistics()
            self.calculate_best_fit()
            self.method_results_label.config(text=f"Interpolated to {final_point_count} points.")
        else:
            messagebox.showinfo("Interpolation",
                                "No new points were added. The data is already dense enough or interpolated points fall outside the line mask.")

    def extract_data_streamlined(self):
        """
        Data extraction method implementing the user-specified adaptive thresholding algorithm
        to intelligently process both thick baselines and sharp peaks.
        """
        if not all([self.image, len(self.calibration_points) == 4]):
            messagebox.showwarning("Prerequisites Missing",
                                   "Please ensure an image is loaded and 4 calibration points are set.")
            return

        # The check for color/line selection is now inside create_final_mask, so it's streamlined here
        self.clear_data()

        try:
            self.progress_bar.pack(fill=tk.X, pady=2, padx=5)
            self.root.update_idletasks()

            # Step 1: Create the final, clean mask of the line to be processed.
            self.progress_bar['value'] = 10
            self.root.update_idletasks()
            final_mask = self.create_final_mask()
            if cv2.countNonZero(final_mask) < 50:
                messagebox.showwarning("Mask Error",
                                       "The detected line mask is too small to extract data. Please try another detection method or adjust parameters.")
                self.progress_bar.pack_forget()
                return

            min_x = min(p[0] for p in self.calibration_points)
            max_x = max(p[0] for p in self.calibration_points)

            # --- NEW ADAPTIVE THRESHOLD LOGIC ---
            # Step 2: First pass - scan the mask to measure the thickness of the line at every x-coordinate.
            self.progress_bar['value'] = 25
            self.root.update_idletasks()

            for x in range(min_x, max_x):
                if x >= final_mask.shape[1]: continue

                y_indices = np.where(final_mask[:, x] > 0)[0]
                if y_indices.size == 0:
                    continue

                y_top = y_indices[0]
                y_bottom = y_indices[-1]
                thickness = y_bottom - y_top

                y_final = 0
                if thickness <= adaptive_threshold:
                    # Case 1: The line is of normal or thin thickness (a peak). Average to find the centerline.
                    y_final = y_top + thickness / 2
                else:
                    # Case 2: The line is unusually thick (a noisy baseline). Take the top point as requested.
                    y_final = y_top

                high_res_path.append((x, int(round(y_final))))
                thicknesses = []

            if len(high_res_path) < 20:
                messagebox.showwarning("Extraction Failed",
                                       f"Only found {len(high_res_path)} data points. The final path is too short.")
                self.progress_bar.pack_forget()
                return

            # Step 5: Sample the final number of points from the high-resolution path.
            self.progress_bar['value'] = 90
            self.root.update_idletasks()
            self.extracted_points = self.sample_points_from_path(high_res_path)

            # Step 6: Finalize and display the results.
            self.method_results_label.config(text=f"Extracted {len(self.extracted_points)} points.")
            self.convert_to_real_coordinates()
            self.draw_extracted_points()
            self.calculate_statistics()
            self.calculate_best_fit()

        except Exception as e:
            messagebox.showerror("Extraction Error", f"An error occurred during data extraction: {e}")
        finally:
            self.progress_bar.pack_forget()

    def create_final_mask(self):
        """
        Creates the final binary mask for data extraction by intelligently combining
        line detection, color selection, and the calibrated plot area.
        """
        # 1. Create a base mask from the calibrated plot area
        plot_mask = np.zeros(self.image_array.shape[:2], dtype=np.uint8)
        if len(self.calibration_points) == 4:
            calib_pts_int = np.array(self.calibration_points, dtype=np.int32)
            cv2.fillPoly(plot_mask, [calib_pts_int], 255)
            # Erode slightly to avoid capturing axis lines
            kernel = np.ones((3, 3), np.uint8)
            plot_mask = cv2.erode(plot_mask, kernel, iterations=2)
        else:
            # If not calibrated, use the whole image as a fallback (though extraction requires calibration)
            plot_mask.fill(255)

        # 2. Determine the primary data source mask (either from line detection or color)
        source_mask = None
        line_selection_mask = self.line_masking.get_current_line_mask()

        if line_selection_mask is not None:
            # PRIORITIZE: If a line is explicitly selected via the masking tools, use it.
            print("Using Line Detection mask as primary source.")
            source_mask = line_selection_mask
        elif self.selected_color:
            # FALLBACK: If no line is selected, but a color is, use the color mask.
            print("Using Color Selection mask as primary source.")
            source_mask = self.enhance_line_detection(self.image_array, self.selected_color)

        if source_mask is None:
            # If no data source is identified, return an empty mask.
            print("Warning: No data source (line detection or color) found for creating mask.")
            return np.zeros(self.image_array.shape[:2], dtype=np.uint8)

        # 3. Combine the plot area mask with the data source mask
        # This ensures we only look for data within the calibrated region.
        combined_mask = cv2.bitwise_and(plot_mask, source_mask)

        # 4. Apply exclusion zones to the final combined mask
        for (x0, y0, x1, y1) in self.exclusion_zones:
            # Ensure coordinates are within image bounds before modifying the mask
            y1_clamped, x1_clamped = min(y1, combined_mask.shape[0]), min(x1, combined_mask.shape[1])
            y0_clamped, x0_clamped = max(y0, 0), max(x0, 0)
            if y0_clamped < y1_clamped and x0_clamped < x1_clamped:
                combined_mask[y0_clamped:y1_clamped, x0_clamped:x1_clamped] = 0

        return combined_mask

    def trace_skeleton_path(self, skeleton):
        """
        Traces a skeleton from left to right to create an ordered path,
        robustly handling small gaps and branches using a KD-tree.
        """
        # Find all non-zero points (pixels) in the skeleton image
        y_coords, x_coords = np.where(skeleton > 0)
        if len(x_coords) == 0:
            print("Warning: Skeleton is empty, no path to trace.")
            return []

        # Combine individual x, y coordinates into a single array of points
        points = np.vstack((x_coords, y_coords)).T
        if points.shape[0] < 2:
            return points.tolist()  # Not enough points to trace a path

        tree = cKDTree(points)

        # Start tracing from the leftmost point of the skeleton
        start_index = np.argmin(points[:, 0])

        # Use a boolean array to keep track of visited points, which is faster than using a set
        visited = np.zeros(len(points), dtype=bool)

        path_indices = []
        current_idx = start_index

        # Set a max distance to jump across gaps. This is linked to the user-adjustable
        # clustering parameter for consistency, making it adaptable.
        max_gap_distance = self.line_masking.detection_parameters['clustering_eps'] * 2.0

        # Loop until we can't find any more connected points
        while current_idx is not None:
            path_indices.append(current_idx)
            visited[current_idx] = True

            # Query the KD-tree for all neighbors within the max_gap_distance
            # We look for a few potential neighbors (k=20) to find the best unvisited one.
            distances, indices = tree.query(points[current_idx], k=min(20, len(points) - 1),
                                            distance_upper_bound=max_gap_distance)

            next_idx = None
            # Find the closest unvisited neighbor from the query results
            for i, neighbor_idx in enumerate(indices):
                # The query returns infinity if no points are within the bound or invalid indices
                if np.isinf(distances[i]) or neighbor_idx >= len(points):
                    break  # No more valid neighbors found

                if not visited[neighbor_idx]:
                    next_idx = neighbor_idx
                    break  # Found the closest unvisited neighbor, so we use it

            current_idx = next_idx  # Move to the next point for the next iteration

        # Convert the path of indices back into a list of (x, y) coordinates
        ordered_path = points[path_indices]

        # To ensure the path is a clean function (one y for each x), we average y-values
        # for any x-coordinates that have multiple points (e.g., in vertical sections).
        path_dict = {}
        for x, y in ordered_path:
            if x not in path_dict:
                path_dict[x] = []
            path_dict[x].append(y)

        # Calculate the final, clean path
        final_path = []
        for x in sorted(path_dict.keys()):
            final_path.append((x, int(np.mean(path_dict[x]))))

        print(f"Traced a path with {len(final_path)} points.")
        return final_path

    def sample_points_from_path(self, path):
        """
        Samples points from the full path based on UI settings.
        MODIFIED to use scipy.signal.find_peaks to ensure all local peaks and valleys
        are preserved in the final sampled output, which is critical for accurate
        interpolation and data representation.
        """
        if not path:
            return []

        # --- NEW: Peak detection to preserve important features ---
        # We find peaks on the negated Y-axis because lower Y values are higher on the screen.
        y_values = -np.array([p[1] for p in path])

        # Much lower prominence to catch subtle features
        # Distance ensures we don't oversample very close peaks
        peak_indices, _ = find_peaks(y_values, prominence=2, width=1, distance=3)
        valley_indices, _ = find_peaks(-y_values, prominence=2, width=1, distance=3)

        # Also detect inflection points (where curvature changes)
        if len(y_values) > 10:
            # Second derivative to find inflection points
            second_deriv = np.diff(y_values, n=2)
            # Sign changes indicate inflection
            inflection_indices = np.where(np.diff(np.sign(second_deriv)))[0] + 1
            feature_indices = set(peak_indices) | set(valley_indices) | set(inflection_indices)
        else:
            feature_indices = set(peak_indices) | set(valley_indices)

        # Combine the indices of peaks and valleys into a set for efficient lookup.
        feature_indices = set(peak_indices) | set(valley_indices)

        # --- Original sampling logic ---
        mode = self.mode_var.get()
        try:
            value = float(self.value_var.get())
        except ValueError:
            value = 10

        sampled_points = []
        if mode == "Number of points":
            num_points = int(value)
            if num_points < 2:
                num_points = 2
            indices = np.linspace(0, len(path) - 1, num_points, dtype=int)

            # --- MODIFICATION: Add feature indices to the regularly sampled indices ---
            # This ensures that even if regular sampling misses a peak, it gets included.
            final_indices = sorted(list(set(indices) | feature_indices))
            sampled_points = [path[i] for i in final_indices]

        else:  # "Step between points"
            step = int(value)
            if step <= 0:
                step = 1
            last_x = -step

            # --- MODIFICATION: Integrate peak preservation with step-based sampling ---
            # We iterate through the path and add points based on step, but also add any
            # feature points (peaks/valleys) we encounter along the way.
            for i, (x, y) in enumerate(path):
                is_feature = i in feature_indices
                is_step_point = x >= last_x + step

                if is_step_point or is_feature:
                    sampled_points.append((x, y))
                    if is_step_point:
                        last_x = x

        # Ensure first and last points are included
        if path[0] not in sampled_points:
            sampled_points.insert(0, path[0])
        if path[-1] not in sampled_points:
            sampled_points.append(path[-1])

        # The old global min/max check is now redundant because find_peaks is more comprehensive.

        # Return a sorted list with duplicates removed
        return sorted(list(dict.fromkeys(sampled_points)))

    def enhance_line_detection(self, image_array, target_color):
        """
        Creates a binary mask for pixels near the target_color using an adaptive 5% tolerance.
        This is more robust than a fixed integer tolerance.
        """
        if target_color is None:
            return np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

        target_np = np.array(target_color, dtype=np.int32)

        # Calculate a 5% tolerance for each RGB channel.
        # A minimum tolerance of 12 is added to handle dark colors where 5% would be too small.
        tolerance_values = np.maximum(12, (target_np * 0.05)).astype(np.int32)

        # Define the lower and upper bounds for the color range, ensuring they are within [0, 255]
        lower_bound = np.clip(target_np - tolerance_values, 0, 255).astype(np.uint8)
        upper_bound = np.clip(target_np + tolerance_values, 0, 255).astype(np.uint8)

        # Create the mask using cv2.inRange, which is highly efficient
        mask = cv2.inRange(image_array, lower_bound, upper_bound)

        # Clean up the resulting mask to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def clear_data(self):
        self.extracted_points, self.real_coordinates, self.statistics, self.best_fit_equation = [], [], {}, ""
        self.best_fit_line_pixel_coords = None
        self.method1_points, self.method2_points = [], []
        self.method_results_label.config(text="No data extracted")
        self.canvas.delete("extracted", "best_fit", "extrema")

    def get_opposite_color(self, rgb):
        return f'#{255 - rgb[0]:02x}{255 - rgb[1]:02x}{255 - rgb[2]:02x}' if rgb else "#000000"

    def draw_extracted_points(self):
        self.canvas.delete("extracted")
        if not self.extracted_points: return
        point_color = self.get_opposite_color(self.selected_color)
        for x, y in self.extracted_points:
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=point_color, outline="", tags="extracted")

    def find_extrema(self):
        if not self.real_coordinates: messagebox.showwarning("No Data", "Please extract data first."); return
        y_values = [p[1] for p in self.real_coordinates]
        max_idx = np.argmax(y_values);
        min_idx = np.argmin(y_values)
        self.statistics['Peak'] = self.real_coordinates[max_idx]
        self.statistics['Valley'] = self.real_coordinates[min_idx]
        self.draw_extrema_points();
        self.update_statistics_display()

    def draw_extrema_points(self):
        self.canvas.delete("extrema")
        if 'Peak' not in self.statistics or not self.extracted_points: return
        peak_real = self.statistics['Peak'];
        valley_real = self.statistics['Valley']
        peak_idx = np.argmin([np.linalg.norm(np.array(p) - np.array(peak_real)) for p in self.real_coordinates])
        valley_idx = np.argmin([np.linalg.norm(np.array(p) - np.array(valley_real)) for p in self.real_coordinates])

        peak_px_x, peak_px_y = self.extracted_points[peak_idx]
        self.canvas.create_oval(peak_px_x - 5, peak_px_y - 5, peak_px_x + 5, peak_px_y + 5, fill=self.peak_color,
                                outline='black', width=1, tags='extrema')
        valley_px_x, valley_px_y = self.extracted_points[valley_idx]
        self.canvas.create_oval(valley_px_x - 5, valley_px_y - 5, valley_px_x + 5, valley_px_y + 5,
                                fill=self.valley_color, outline='black', width=1, tags='extrema')

    def convert_to_real_coordinates(self):
        if len(self.calibration_points) != 4 or not self.extracted_points: return
        try:
            x_min, x_max = float(self.x_min_var.get()), float(self.x_max_var.get())
            y_min, y_max = float(self.y_min_var.get()), float(self.y_max_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Calibration axis values must be numbers.")
            return

        src = np.array(self.calibration_points, dtype='float32')
        dst = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src, dst)
        self.real_coordinates = cv2.perspectiveTransform(np.array([self.extracted_points], dtype='float32'), matrix)[
            0].tolist()

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
        x_vals, y_vals = np.array([p[0] for p in self.real_coordinates]), np.array(
            [p[1] for p in self.real_coordinates])
        slope, intercept, r_val, _, _ = stats.linregress(x_vals, y_vals)
        self.statistics['r_squared'] = r_val ** 2
        self.best_fit_equation = f"y = {slope:.4f}x + {intercept:.4f}"

        x_vals_real = [p[0] for p in self.real_coordinates];
        y_vals_real = [p[1] for p in self.real_coordinates]
        start_real_x = min(x_vals_real);
        end_real_x = max(x_vals_real)
        start_real_y = slope * start_real_x + intercept;
        end_real_y = slope * end_real_x + intercept
        start_real, end_real = (start_real_x, start_real_y), (end_real_x, end_real_y)
        x_min, x_max = float(self.x_min_var.get()), float(self.x_max_var.get())
        y_min, y_max = float(self.y_min_var.get()), float(self.y_max_var.get())
        dst = np.array(self.calibration_points, dtype='float32')
        src = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype='float32')
        inv_matrix = cv2.getPerspectiveTransform(src, dst)
        pixel_pts = cv2.perspectiveTransform(np.array([[start_real, end_real]], dtype='float32'), inv_matrix)[0]
        self.best_fit_line_pixel_coords = (pixel_pts[0][0], pixel_pts[0][1], pixel_pts[1][0], pixel_pts[1][1])

        self.draw_best_fit_line();
        self.update_statistics_display()

    def draw_best_fit_line(self):
        self.canvas.delete("best_fit")
        if self.best_fit_line_pixel_coords and self.show_best_fit.get():
            x1, y1, x2, y2 = self.best_fit_line_pixel_coords
            self.canvas.create_line(x1, y1, x2, y2, fill='purple', width=2, tags="best_fit")

    def export_csv(self):
        """Exports the extracted data to a clean CSV file, compatible with the Accuracy Tester."""
        if not self.real_coordinates:
            messagebox.showwarning("No Data", "No data has been extracted to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            # Create a DataFrame with standard 'x' and 'y' column names for compatibility
            df = pd.DataFrame(self.real_coordinates, columns=['x', 'y'])

            # Export directly to CSV without any commented metadata headers
            df.to_csv(file_path, index=False)

            messagebox.showinfo("Success", f"Data successfully exported to\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data.\nError: {e}")

    def export_json(self):
        """Exports the extracted data and statistics to a JSON file."""
        if not self.real_coordinates:
            messagebox.showwarning("No Data", "No data has been extracted to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        try:
            x_vals = [p[0] for p in self.real_coordinates]
            y_vals = [p[1] for p in self.real_coordinates]
            data_to_export = {'x': x_vals, 'y': y_vals, 'stats': self.statistics}
            with open(file_path, 'w') as f:
                json.dump(data_to_export, f, indent=4)
            messagebox.showinfo("Success", f"Data successfully exported to\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data to JSON.\nError: {e}")

    def export_plot_as_png(self):
        """Placeholder function to export the plot view as a PNG image."""
        messagebox.showinfo("Not Implemented", "This feature is not yet available.")


# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = RamanDataDigitizer(root)
    root.mainloop()

