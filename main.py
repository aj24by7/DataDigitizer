import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import cv2
import pandas as pd
import json
import os
import re
import pytesseract
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter, find_peaks


def remove_shallow_components(mask, min_height_ratio=0.02, min_area_ratio=0.0005, min_aspect=0.08):
    """
    Remove thin horizontal components (like axes/bridges) from a binary mask.
    """
    if mask is None:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    mask_h, mask_w = mask.shape[:2]
    min_height = max(3, int(mask_h * min_height_ratio))
    min_area = max(40, int(mask_h * mask_w * min_area_ratio))

    cleaned = np.zeros_like(mask)
    kept_any = False
    tallest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_HEIGHT]) if num_labels > 1 else 0

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        aspect = height / max(1, width)
        if height >= min_height or (area >= min_area and aspect >= min_aspect):
            cleaned[labels == idx] = 255
            kept_any = True

    if not kept_any and tallest_idx > 0:
        cleaned[labels == tallest_idx] = 255

    return cleaned if cv2.countNonZero(cleaned) > 0 else mask


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
                self.parent.debug_label.config(
                    text=f"Debug: Applying {len(self.parent.exclusion_zones)} exclusion zones...")
                self.parent.root.update_idletasks()
                print(f"Applying {len(self.parent.exclusion_zones)} exclusion zones to line detection")

            # Clear previous results
            self.detected_lines.clear()
            self.line_colors.clear()
            self.line_previews.clear()
            self.current_line_index = 0

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
                                    "No distinct lines detected. Try:\n1. Lower edge threshold\n2. Smaller min length\n3. Use Line Isolation button\n4. First run Auto-Exclude Text if not done")
                return

            # Cluster points into separate lines using custom clustering
            self.parent.debug_label.config(text="Debug: Clustering points into lines...")
            self.parent.root.update_idletasks()
            separated_lines = self._separate_lines_clustering(all_line_points)
            print(f"Clustering produced {len(separated_lines)} lines")

            if not separated_lines:
                self.parent.debug_label.config(text="Debug: Clustering failed. Try larger cluster distance.")
                messagebox.showinfo("No Lines",
                                    "Clustering failed to separate lines. Try:\n1. Increase cluster distance\n2. Decrease min points\n3. Use Line Isolation")
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

            filtered_points = []
            for point in line_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                    if hasattr(self.parent, 'is_in_exclusion_zone') and self.parent.is_in_exclusion_zone(x, y):
                        continue
                    filtered_points.append((x, y))

            if len(filtered_points) < 2:
                continue

            # Create binary mask
            mask = np.zeros(img_shape, dtype=np.uint8)

            unique_points = np.unique(np.array(filtered_points, dtype=np.int32), axis=0)
            if len(unique_points) < 2:
                continue

            sort_idx = np.lexsort((unique_points[:, 1], unique_points[:, 0]))
            ordered_points = unique_points[sort_idx]
            polyline = ordered_points.reshape(-1, 1, 2)

            brush = max(1, int(max(1, self.detection_parameters['thickness_tolerance']) * 0.6))
            cv2.polylines(mask, [polyline], False, 255, brush)

            refinement_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, refinement_kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, refinement_kernel, iterations=1)
            if min(mask.shape[:2]) >= 3:
                mask = cv2.medianBlur(mask, 3)
            mask = remove_shallow_components(mask, min_height_ratio=0.015)
            if hasattr(self.parent, 'remove_outer_border_components'):
                mask = self.parent.remove_outer_border_components(mask)

            if hasattr(self.parent, 'apply_axis_guards'):
                mask = self.parent.apply_axis_guards(mask)

            if cv2.countNonZero(mask):
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

        self._order_lines_by_score()

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

    def _order_lines_by_score(self):
        """Prioritize lines that look most like the primary spectrum."""
        if not self.detected_lines:
            return
        scores = []
        for mask in self.detected_lines:
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) == 0:
                scores.append(0)
                continue
            span = np.max(x_coords) - np.min(x_coords)
            area = cv2.countNonZero(mask)
            score = span + (area / 500.0)
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        self.detected_lines = [self.detected_lines[i] for i in order]
        self.line_colors = [self.line_colors[i] for i in order]
        self.line_previews = [self.line_previews[i] for i in order]
        self.current_line_index = 0

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


class AxisNumberDetector:
    """Lightweight OCR helper that finds numeric axis labels along the plot edges."""

    number_regex = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

    def __init__(self):
        self.available = self._init_tesseract()
        if not self.available:
            print("Warning: Tesseract OCR not found. Smart calibration will be unavailable.")

    def _init_tesseract(self):
        possible_paths = [
            os.environ.get("TESSERACT_PATH"),
            os.path.join(os.getcwd(), "Tesseract", "tesseract.exe"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for path in possible_paths:
            if not path:
                continue
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    pytesseract.get_tesseract_version()
                    return True
                except pytesseract.pytesseract.TesseractNotFoundError:
                    continue
        # Final attempt without specifying path
        try:
            pytesseract.get_tesseract_version()
            return True
        except pytesseract.pytesseract.TesseractNotFoundError:
            return False

    def detect_numbers(self, pil_image):
        if not self.available:
            return {'x': [], 'y': []}
        if pil_image is None:
            return {'x': [], 'y': []}
        img_rgb = np.array(pil_image)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]

        x_band_start = int(height * 0.65)
        y_band_end = int(width * 0.35)

        x_roi = gray[x_band_start:, :]
        y_roi = gray[:, :y_band_end]

        x_labels = self._extract_axis_labels(
            x_roi, offset=(0, x_band_start), axis='x')
        y_labels = self._extract_axis_labels(
            y_roi, offset=(0, 0), axis='y')

        x_labels = self.normalize_axis_labels(x_labels, axis='x')
        y_labels = self.normalize_axis_labels(y_labels, axis='y')
        return {'x': x_labels, 'y': y_labels}

    def _extract_axis_labels(self, roi, offset=(0, 0), axis='x'):
        """
        Run OCR on the given ROI using multiple preprocessing variants/configs.
        """
        offset_x, offset_y = offset
        variants = self._prepare_variants(roi)
        configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789.-',
            '--psm 7 -c tessedit_char_whitelist=0123456789.-',
            '--psm 11 -c tessedit_char_whitelist=0123456789.-'
        ]

        collected = []
        seen_positions = set()

        for variant in variants:
            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        variant,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                except Exception as e:
                    print(f"OCR variant failed ({axis}): {e}")
                    continue

                n = len(data['text'])
                for i in range(n):
                    text = data['text'][i].strip()
                    if not text:
                        continue
                    sanitized = text.replace(' ', '').replace(',', '')
                    if not self.number_regex.match(sanitized):
                        continue
                    try:
                        conf = float(data['conf'][i])
                    except (ValueError, TypeError):
                        conf = 0.0
                    if conf < 40:
                        continue
                    value = self._to_float(sanitized)
                    if value is None:
                        continue
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    global_bbox = (x + offset_x, y + offset_y, w, h)
                    cx = global_bbox[0] + w / 2
                    cy = global_bbox[1] + h / 2
                    key = (round(cx / 5), round(cy / 5), round(value, 2))
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)
                    collected.append({
                        'value': value,
                        'confidence': conf / 100.0,
                        'bbox': global_bbox,
                        'cx': cx,
                        'cy': cy
                    })

        # Axis-specific filtering to keep only likely candidates
        if axis == 'x':
            min_y = offset_y
            collected = [c for c in collected if c['cy'] > min_y + roi.shape[0] * 0.15]
        else:
            max_x = offset_x + roi.shape[1]
            collected = [c for c in collected if c['cx'] < max_x - roi.shape[1] * 0.15]

        return collected

    def _prepare_variants(self, roi):
        """Generate a set of preprocessed images to improve OCR robustness."""
        variants = []
        roi_eq = cv2.equalizeHist(roi)
        variants.append(roi_eq)

        blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th)
        variants.append(th)

        dilated = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
        variants.append(dilated)

        inverted = cv2.bitwise_not(roi_eq)
        variants.append(inverted)

        eroded = cv2.erode(th, np.ones((2, 2), np.uint8), iterations=1)
        eroded = cv2.bitwise_not(eroded)
        variants.append(eroded)
        return variants

    def pick_best_pair(self, labels, axis=None):
        if len(labels) < 2:
            return None

        # Axis-based heuristic: pick farthest apart positions along that axis
        if axis == 'x':
            ordered = sorted(labels, key=lambda e: e['cx'])
            return (ordered[0], ordered[-1])
        if axis == 'y':
            ordered = sorted(labels, key=lambda e: e['cy'], reverse=True)  # bottom (min) to top (max)
            return (ordered[0], ordered[-1])

        best = None
        best_score = -1
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                spread = abs(a['value'] - b['value'])
                if spread <= 0:
                    continue
                axis_distance = abs(a['cx'] - b['cx']) + abs(a['cy'] - b['cy'])
                score = spread * (a['confidence'] + b['confidence']) - axis_distance * 0.01
                if score > best_score:
                    best_score = score
                    best = (a, b)
        return best

    def _to_float(self, text):
        try:
            return float(text)
        except ValueError:
            try:
                cleaned = text.replace('O', '0').replace('o', '0')
                return float(cleaned)
            except ValueError:
                return None

    def normalize_axis_labels(self, labels, axis):
        if not labels:
            return []
        if axis == 'x':
            labels = sorted(labels, key=lambda e: e['cx'])
        else:
            labels = sorted(labels, key=lambda e: e['cy'], reverse=True)

        values = [abs(lbl['value']) for lbl in labels]
        diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1) if abs(values[i + 1] - values[i]) > 0]
        median_step = float(np.median(diffs)) if diffs else None

        for lbl in labels:
            lbl['value'] = self._normalize_value(lbl['value'], median_step)
        return labels

    def _normalize_value(self, value, median_step):
        if median_step is None or median_step == 0:
            return value
        normalized = value
        limit = abs(median_step)

        for _ in range(4):
            rounded = int(round(abs(normalized)))
            if abs(normalized) > 12 * limit and abs(normalized) >= 100 and rounded % 10 == 0:
                normalized /= 10
            else:
                break

        for _ in range(4):
            if abs(normalized) < 0.25 * limit:
                normalized *= 10
            else:
                break
        return normalized


class RamanDataDigitizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectra Data Digitizer - Multiscale Extraction Studio")
        self.root.geometry("1500x950")

        # Refined red / blue palette
        self.colors = {
            'bg_main': '#e8ecff',  # soft blue background
            'bg_panel': '#ffffff',  # crisp cards
            'bg_secondary': '#c7d2fe',  # accent blocks
            'primary': '#2563eb',  # consistent bright blue
            'primary_hover': '#1d4ed8',  # hover state
            'success': '#0ea5e9',  # cyan accent
            'warning': '#f97316',  # vivid amber
            'danger': '#dc2626',  # bright red
            'info': '#2563eb',  # supporting blue
            'secondary': '#334155',  # neutral buttons/text
            'accent': '#ef4444',  # red accent
            'text_primary': '#0f172a',  # dark navy text
            'text_secondary': '#475569',  # muted navy
            'border': '#c3ccff',  # light border
            'panel_shadow': '#aab6f4'
        }

        self.root.configure(bg=self.colors['bg_main'])
        self.root.minsize(1200, 820)
        self.root.option_add('*Font', '{Segoe UI} 9')

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass
        self.style.configure('Rounded.TButton',
                             background=self.colors['primary'],
                             foreground='white',
                             padding=8,
                             borderwidth=0,
                             focusthickness=0)
        self.style.map('Rounded.TButton',
                       background=[('active', self.colors['primary_hover'])])
        self.style.configure('Accent.TButton',
                             background=self.colors['accent'],
                             foreground='white',
                             padding=8,
                             borderwidth=0,
                             focusthickness=0)
        self.style.map('Accent.TButton',
                       background=[('active', '#d9044d')])

        # --- State Variables ---
        self.image = None
        self.image_array = None
        self.photo = None
        self.original_image = None  # Store original for zoom operations
        self.zoom_level = 1.0  # Track current zoom level
        self.calibration_points = []
        self.selected_color = None
        self._last_color_mask_tolerance = None
        self.extracted_points = []
        self.real_coordinates = []
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_mode = "normal"
        self.manual_points = []

        # --- Interactive State ---
        self.exclusion_zones = []
        self.temp_rect_start = None
        self.temp_rect_id = None
        self.dragging_line_index = None  # 0:top, 1:right, 2:bottom, 3:left

        # --- Detection Method Results ---
        self.method1_points = []  # Original method (main4)
        self.method2_points = []  # New method (from main3)
        self.current_method = "auto"  # "auto", "method1", "method2"
        self.method_var = tk.StringVar(value="Auto (Best)")

        # --- Line Masking System - NEW FEATURE ---
        self.line_masking = LineMaskingSystem(self)
        self.number_detector = AxisNumberDetector()

        self.setup_ui()
        self.setup_bindings()

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

        # --- Main Frame ---
        main_frame = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # --- Left Panel (Image) ---
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_panel'], relief=tk.FLAT, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

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
        right_outer_frame = tk.Frame(main_frame, bg=self.colors['bg_main'], width=340)
        right_outer_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_outer_frame.pack_propagate(False)

        canvas_right = tk.Canvas(right_outer_frame, bg=self.colors['bg_main'], highlightthickness=0)
        scrollbar_right = ttk.Scrollbar(right_outer_frame, orient="vertical", command=canvas_right.yview)
        scrollable_frame = tk.Frame(canvas_right, bg=self.colors['bg_main'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all"))
        )

        scroll_window = canvas_right.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_right.configure(yscrollcommand=scrollbar_right.set)
        canvas_right.bind(
            "<Configure>",
            lambda e, c=canvas_right, w_id=scroll_window: c.itemconfig(w_id, width=e.width)
        )

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
        zoom_outer_frame = tk.LabelFrame(right_panel, text="Step 1: Zoom Preview",
                                         bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                         font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        zoom_outer_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        zoom_controls = tk.Frame(zoom_outer_frame, bg=self.colors['bg_panel'])
        zoom_controls.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(zoom_controls, text="-", command=self.zoom_out,
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
        color_frame = tk.LabelFrame(right_panel, text="Step 2: Color Selection",
                                    bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                    font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        color_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        color_btn_frame = tk.Frame(color_frame, bg=self.colors['bg_panel'])
        color_btn_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(color_btn_frame, text="Pick Color", command=self.enter_pick_color_mode,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(color_btn_frame, text="Refresh Palette", command=self.run_color_analysis,
                  bg=self.colors['secondary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.color_display = tk.Label(color_frame, text="No Color Selected",
                                      bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                      relief=tk.FLAT, bd=1, height=1, font=('Segoe UI', 8))
        self.color_display.pack(fill=tk.X, pady=(0, 6), padx=8)

        tolerance_frame = tk.Frame(color_frame, bg=self.colors['bg_panel'])
        tolerance_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(tolerance_frame, text="Color tolerance:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.color_tolerance_var = tk.IntVar(value=40)
        tk.Spinbox(tolerance_frame, from_=0, to=100, textvariable=self.color_tolerance_var,
                   width=4, font=('Segoe UI', 8), relief=tk.FLAT, bd=1,
                   bg=self.colors['bg_secondary']).pack(side=tk.RIGHT)

        tk.Label(color_frame, text="Suggested colors", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=8)
        self.palette_frame = tk.Frame(color_frame, bg=self.colors['bg_panel'])
        self.palette_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        placeholder = tk.Label(self.palette_frame, text="Palette will appear after loading an image.",
                               bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                               font=('Segoe UI', 8), wraplength=220, justify=tk.LEFT)
        placeholder.pack(anchor='w')

        # ========== NEW: COLOR INDEX ANALYSIS ==========
        # Removed legacy color index frame to declutter the UI

        # ========== STEP 3: CALIBRATION ==========
        calib_frame = tk.LabelFrame(right_panel, text="Step 3: Calibration",
                                    bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                    font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        calib_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        calib_button_row1 = tk.Frame(calib_frame, bg=self.colors['bg_panel'])
        calib_button_row1.pack(fill=tk.X, padx=8, pady=(6, 3))
        tk.Button(calib_button_row1, text="Auto Calibrate", command=self.auto_calibrate,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(calib_button_row1, text="Manual", command=self.enter_manual_calibrate_mode,
                  bg=self.colors['info'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(calib_button_row1, text="Clear", command=self.clear_calibration,
                  bg=self.colors['danger'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        tk.Button(calib_button_row1, text="Smart Auto (OCR)", command=self.auto_calibrate_from_numbers,
                  bg=self.colors['success'], fg='white', font=('Segoe UI', 8, 'bold'),
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

        self.limit_to_calibration = tk.BooleanVar(value=False)
        tk.Checkbutton(calib_frame,
                       text="Limit extraction to calibration box",
                       variable=self.limit_to_calibration,
                       bg=self.colors['bg_panel'],
                       fg=self.colors['text_secondary'],
                       selectcolor=self.colors['bg_secondary'],
                       font=('Segoe UI', 8),
                       activebackground=self.colors['bg_panel'],
                       activeforeground=self.colors['text_primary'],
                       anchor='w',
                       relief=tk.FLAT,
                       bd=0,
                       highlightthickness=0).pack(fill=tk.X, padx=8, pady=(0, 4))

        # ========== STEP 4: SMART EXCLUSION TOOLS ==========
        exclusion_frame = tk.LabelFrame(right_panel, text="Step 4: Smart Exclusion Tools",
                                        bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                        font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        exclusion_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        auto_buttons_frame = tk.Frame(exclusion_frame, bg=self.colors['bg_panel'])
        auto_buttons_frame.pack(fill=tk.X, padx=8, pady=6)

        ttk.Button(auto_buttons_frame, text="Auto Clean Plot", style='Rounded.TButton',
                   command=self.auto_cleanup_plot).pack(fill=tk.X, pady=2)
        tk.Label(auto_buttons_frame,
                 text="Removes labels + borders in one pass for cleaner masks.",
                 bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'],
                 font=('Segoe UI', 8, 'italic')).pack(fill=tk.X, pady=(0, 4))

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
        line_masking_frame = tk.LabelFrame(right_panel, text="Step 5: Line Isolation",
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

        tk.Button(detect_frame, text="Line Isolation", command=self.detect_lines,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 9, 'bold'),
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

        # ========== STEP 6: DATA EXTRACTION ==========
        extract_frame = tk.LabelFrame(right_panel, text="Step 6 - Data Extraction",
                                      bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                      font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        extract_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        method_select_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        method_select_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(method_select_frame, text="Method:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        method_dropdown = ttk.Combobox(method_select_frame, textvariable=self.method_var, width=14,
                                       state="readonly", font=('Segoe UI', 8))
        method_dropdown['values'] = ("Auto (Best)", "Method 1", "Method 2")
        method_dropdown.bind("<<ComboboxSelected>>", self.on_method_changed)
        method_dropdown.pack(side=tk.RIGHT)

        mode_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        mode_frame.pack(fill=tk.X, pady=(0, 6), padx=8)

        tk.Label(mode_frame, text="Mode:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Step between points")
        mode_dropdown = ttk.Combobox(mode_frame, textvariable=self.mode_var, width=14,
                                     state="readonly", font=('Segoe UI', 8))
        mode_dropdown['values'] = ("Step between points", "Number of points")
        mode_dropdown.pack(side=tk.RIGHT)

        value_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        value_frame.pack(fill=tk.X, pady=(0, 6), padx=8)
        tk.Label(value_frame, text="Value:", bg=self.colors['bg_panel'],
                 fg=self.colors['text_secondary'], font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.value_var = tk.StringVar(value="10")
        tk.Entry(value_frame, textvariable=self.value_var, width=10, font=('Segoe UI', 8),
                 relief=tk.FLAT, bd=1, bg=self.colors['bg_secondary']).pack(side=tk.RIGHT)

        action_button_frame = tk.Frame(extract_frame, bg=self.colors['bg_panel'])
        action_button_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(action_button_frame, text="Dual Extract",
                  command=self.extract_data_dual,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        tk.Button(action_button_frame, text="Manual Points Mode",
                  command=self.enable_manual_point_mode,
                  bg=self.colors['info'], fg='white', font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)

        self.export_normalized_var = tk.BooleanVar(value=False)
        tk.Checkbutton(extract_frame,
                       text="Include normalized Y (0-1) in export",
                       variable=self.export_normalized_var,
                       bg=self.colors['bg_panel'],
                       fg=self.colors['text_secondary'],
                       selectcolor=self.colors['bg_secondary'],
                       activebackground=self.colors['bg_panel'],
                       activeforeground=self.colors['text_primary'],
                       anchor='w',
                       relief=tk.FLAT,
                       bd=0,
                       highlightthickness=0,
                       font=('Segoe UI', 8)).pack(fill=tk.X, padx=8, pady=(2, 0))
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

        # ========== STEP 7: INTERPOLATION ==========
        interpolation_frame = tk.LabelFrame(right_panel, text="Step 7: Interpolation",
                                            bg=self.colors['bg_panel'], fg=self.colors['text_primary'],
                                            font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, bd=1)
        interpolation_frame.pack(fill=tk.X, pady=(0, 8), padx=6)

        interp_button_frame = tk.Frame(interpolation_frame, bg=self.colors['bg_panel'])
        interp_button_frame.pack(fill=tk.X, pady=6, padx=8)

        tk.Button(interp_button_frame, text="Simple Interpolation", command=self.simple_interpolation,
                  bg=self.colors['primary'], fg='white', font=('Segoe UI', 8, 'bold'),
                  relief=tk.FLAT, bd=0).pack(fill=tk.X)

    # === NEW: Line Masking System Functions ===
    def detect_lines(self):
        """Trigger line detection using the advanced masking system."""
        self.line_masking.detect_all_lines()
        self.line_masking.current_line_index = 0
        self.update_line_info_display()
        self.update_line_preview()
        self.redraw_overlays()

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
                messagebox.showinfo("No Lines", "Line Isolation found no line fragments to process.")
                return

            # --- MODIFICATION: Using a larger kernel to more aggressively connect fragments ---
            kernel_large = np.ones((9, 9), np.uint8)
            unified_mask = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

            unified_mask = cv2.dilate(unified_mask, kernel_small, iterations=1)

            self.line_masking.detected_lines = [unified_mask]

            self.line_masking._generate_line_visualization()

            self.debug_label.config(text=f"Debug: Found and merged {found_fragments} fragments.")
            messagebox.showinfo("Line Isolation Complete",
                                f"Detected and merged {found_fragments} fragments into a single line.\n" +
                                "You can now proceed to 'Extract Data'.")

            self.update_line_info_display()
            self.update_line_preview()
            self.redraw_overlays()

        except Exception as e:
            self.debug_label.config(text=f"Debug: Line Isolation failed - {str(e)[:30]}...")
            messagebox.showerror("Line Isolation Error", f"An error occurred during Line Isolation: {e}")

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
        Automatically pick a dominant line color so the user can start faster.
        """
        if self.image_array is None:
            return

        dominant_color, palette = self.analyze_palette(self.image_array)
        self.render_color_palette(palette)
        if dominant_color:
            self.apply_selected_color(dominant_color, label_prefix="Auto")
        else:
            self.apply_selected_color(None)

    def analyze_palette(self, image_array, max_colors=6):
        """
        Build a palette of likely line colors, ignoring bright backgrounds.
        Returns (dominant_color, palette_list).
        """
        if image_array is None:
            return None, []

        pixels = image_array.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        entries = []
        for color_np, count in zip(unique_colors, counts):
            if np.all(color_np > 220):
                continue
            entries.append((tuple(int(c) for c in color_np), int(count)))

        if not entries:
            return None, []

        entries.sort(key=lambda item: item[1], reverse=True)
        total = sum(count for _, count in entries)
        palette = []
        for color, count in entries:
            percentage = (count / total) * 100
            if percentage < 0.2 and palette:
                continue
            palette.append({'rgb': color, 'percentage': percentage})
            if len(palette) >= max_colors:
                break

        dominant_color = None
        if palette:
            if palette[0]['rgb'] == (0, 0, 0) and palette[0]['percentage'] >= 80:
                dominant_color = palette[0]['rgb']
            else:
                for entry in palette:
                    if entry['percentage'] >= 20:
                        dominant_color = entry['rgb']
                        break
                if dominant_color is None:
                    dominant_color = palette[0]['rgb']

        return dominant_color, palette

    def render_color_palette(self, palette):
        if not hasattr(self, 'palette_frame'):
            return
        for child in self.palette_frame.winfo_children():
            child.destroy()

        if not palette:
            tk.Label(self.palette_frame, text="Palette will appear after loading an image.",
                     bg=self.colors['bg_panel'], fg=self.colors['text_secondary'],
                     font=('Segoe UI', 8), wraplength=220, justify=tk.LEFT).pack(anchor='w')
            return

        for entry in palette[:6]:
            rgb = entry['rgb']
            pct = entry['percentage']
            color_hex = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

            row = tk.Frame(self.palette_frame, bg=self.colors['bg_panel'])
            row.pack(fill=tk.X, pady=2)

            swatch = tk.Label(row, text="", bg=color_hex, width=3, relief=tk.FLAT)
            swatch.pack(side=tk.LEFT, padx=(0, 6), ipady=6)

            info = tk.Label(row,
                            text=f"{pct:4.1f}%  {rgb}",
                            bg=self.colors['bg_panel'],
                            fg=self.colors['text_secondary'],
                            font=('Segoe UI', 8))
            info.pack(side=tk.LEFT, expand=True, anchor='w')

            tk.Button(row, text="Select",
                      command=lambda c=rgb: self.apply_selected_color(c),
                      bg=self.colors['primary'], fg='white',
                      font=('Segoe UI', 7, 'bold'),
                      relief=tk.FLAT, bd=0).pack(side=tk.RIGHT)

    def apply_selected_color(self, rgb_color, label_prefix="RGB"):
        display_widget = getattr(self, 'color_display', None)

        if rgb_color is None:
            self.selected_color = None
            self._last_color_mask_tolerance = None
            if display_widget is not None:
                display_widget.config(text="No Color Selected",
                                      bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_secondary'])
            return

        sanitized = tuple(int(c) for c in rgb_color)
        self.selected_color = sanitized
        self._last_color_mask_tolerance = None
        color_hex = f'#{sanitized[0]:02x}{sanitized[1]:02x}{sanitized[2]:02x}'
        if display_widget is not None:
            display_widget.config(text=f"{label_prefix}: {sanitized}", bg=color_hex,
                                  fg=self.get_contrasting_text_color(sanitized))

    def get_contrasting_text_color(self, rgb):
        if rgb is None:
            return self.colors['text_secondary']
        r, g, b = rgb
        brightness = (0.299 * r + 0.587 * g + 0.114 * b)
        return '#0f172a' if brightness > 186 else '#ffffff'

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
        self.apply_selected_color(None)
        self.render_color_palette([])
        if hasattr(self, 'limit_to_calibration'):
            self.limit_to_calibration.set(False)
        self.method_results_label.config(text="No data extracted")

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
        print("Image displayed successfully")

    def redraw_overlays(self):
        self.draw_current_line_mask()  # Keep masks beneath data points
        self.draw_calibration_grid()
        self.draw_calibration_points()
        self.draw_exclusion_zones()
        self.draw_extracted_points()

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
        elif self.current_mode == "manual_point_add":
            self.add_manual_point_at(x, y)
        elif self.current_mode == "delete_zone":
            # New functionality: delete exclusion zone when clicked
            zone_index = self.find_zone_at_position(x, y)
            if zone_index is not None:
                removed_zone = self.exclusion_zones.pop(zone_index)
                self.draw_exclusion_zones()  # Redraw to show the change
                messagebox.showinfo("Zone Deleted", f"Removed exclusion zone at {removed_zone}")
                self.escape_mode()
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

    def auto_exclude_borders(self, silent=False):
        """
        Automatically detect and exclude rectangular borders/frames around graph panels.
        This helps separate multi-panel plots like (A), (B), (C), (D) layouts.
        """
        if self.image is None:
            if not silent:
                messagebox.showwarning("No Image", "Please load an image first.")
            return 0

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
                if not silent:
                    messagebox.showinfo("No Borders",
                                        "No clear panel borders detected.\n"
                                        "This works best with images that have distinct rectangular frames around each graph panel.")
                return 0

            # Remove overlapping border regions to avoid redundancy
            filtered_borders = self.remove_overlapping_regions(detected_borders)

            # Add detected border regions to existing exclusion zones
            self.exclusion_zones.extend(filtered_borders)

            # Redraw to show the new exclusion zones
            self.draw_exclusion_zones()

            if not silent:
                messagebox.showinfo(
                    "Border Exclusion Complete",
                    f"Added {len(filtered_borders)} border exclusion zones.\n"
                    f"Total exclusion zones: {len(self.exclusion_zones)}\n\n"
                    f"This should help separate individual graph panels for line detection."
                )
            else:
                print(f"[Auto Borders] Added {len(filtered_borders)} zones.")
            return len(filtered_borders)

        except Exception as e:
            if not silent:
                messagebox.showerror("Border Exclusion Error", f"Failed to detect borders: {e}")
            else:
                print(f"[Auto Borders] Error: {e}")
            return 0

    def auto_cleanup_plot(self):
        """
        Consolidated helper that runs both text and border exclusion passes.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image before running Auto Clean.")
            return

        before = len(self.exclusion_zones)
        added_text = self.auto_create_text_exclusions(silent=True) or 0
        added_border = self.auto_exclude_borders(silent=True) or 0
        total_added = added_text + added_border

        if total_added == 0:
            messagebox.showinfo(
                "Auto Clean Plot",
                "No new text or border regions were detected. Try adjusting detection parameters."
            )
        else:
            messagebox.showinfo(
                "Auto Clean Plot",
                f"Added {total_added} automatic exclusion zones "
                f"(text: {added_text}, borders: {added_border}).\n"
                f"Total zones: {len(self.exclusion_zones)} (previously {before})."
            )

    def enable_manual_point_mode(self):
        """
        Enable manual point placement on the canvas.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Load an image before adding manual points.")
            return
        if len(self.calibration_points) != 4:
            messagebox.showwarning("Calibration Required", "Set the four calibration points before adding manual data.")
            return
        self.current_mode = "manual_point_add"
        self.canvas.config(cursor="tcross")
        messagebox.showinfo(
            "Manual Point Mode",
            "Click anywhere on the curve to drop a manual point.\n"
            "Press Esc to exit manual point mode."
        )

    def add_manual_point_at(self, x, y):
        """
        Record a manually placed point and refresh downstream artifacts.
        """
        self.manual_points.append((x, y))
        self.extracted_points.append((x, y))
        # Remove duplicates and keep X-sorted ordering
        unique = list(dict.fromkeys(self.extracted_points))
        self.extracted_points = sorted(unique, key=lambda p: (p[0], p[1]))
        self.method_results_label.config(
            text=f"Manual points: {len(self.manual_points)} | Total points: {len(self.extracted_points)}"
        )
        self.draw_extracted_points()
        self.convert_to_real_coordinates()

    def auto_create_text_exclusions(self, silent=False):
        """
        Main function to automatically create exclusion zones for text and numbers.
        This is the primary interface function called from the UI button.
        """
        if self.image is None:
            if not silent:
                messagebox.showwarning("No Image", "Please load an image first.")
            return 0

        try:
            # ADDED: Ask for confirmation if zones already exist
            if self.exclusion_zones and not silent:
                if not messagebox.askyesno("Confirm",
                                           "This will add new text-based exclusion zones to the existing ones. Continue?"):
                    return 0

            # Detect text regions using our comprehensive multi-method approach
            text_regions = self.detect_text_regions()

            if not text_regions:
                if not silent:
                    messagebox.showinfo("Auto-Exclusion", "No text regions detected.")
                return 0

            # Add detected regions to existing exclusion zones
            # We extend rather than replace to preserve any manual exclusions
            self.exclusion_zones.extend(text_regions)

            # Redraw to show the new exclusion zones
            self.draw_exclusion_zones()

            if not silent:
                messagebox.showinfo(
                    "Auto-Exclusion Complete",
                    f"Added {len(text_regions)} automatic text exclusion zones.\n"
                    f"Total exclusion zones: {len(self.exclusion_zones)}"
                )
            else:
                print(f"[Auto Text] Added {len(text_regions)} zones.")
            return len(text_regions)

        except Exception as e:
            if not silent:
                messagebox.showerror("Auto-Exclusion Error", f"Failed to create automatic exclusions: {e}")
            else:
                print(f"[Auto Text] Error: {e}")
            return 0

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
            color_np = self.image_array[y, x]
            self.apply_selected_color(tuple(int(c) for c in color_np))
        self.escape_mode()

    # --- CALIBRATION ---
    def auto_calibrate(self, show_messages=True):
        if self.image is None:
            if show_messages:
                messagebox.showwarning("Warning", "Please load an image first.")
            return False
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
            if show_messages:
                messagebox.showinfo("Auto-Calibration", "Found graph area. You can now drag the sides to fine-tune.")
            self.redraw_overlays();
            self.escape_mode()
            return True
        except Exception as e:
            if show_messages:
                messagebox.showerror("Auto-Calibration Error",
                                     f"Could not detect a clear graph area. Using margins as a fallback.\nError: {e}")
            w, h = self.image.size;
            margin = min(w, h) // 10
            self.calibration_points = [(margin, margin), (w - margin, margin), (w - margin, h - margin),
                                       (margin, h - margin)]
            self.redraw_overlays()
            return False

    def auto_calibrate_from_numbers(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Load an image before running Smart Auto Calibrate.")
            return
        if not hasattr(self, 'number_detector') or self.number_detector is None:
            messagebox.showerror("OCR Missing", "Smart calibration requires Tesseract OCR to be installed.")
            return
        if not getattr(self.number_detector, 'available', False):
            messagebox.showerror(
                "OCR Missing",
                "Tesseract OCR binary not found.\n"
                "Install Tesseract or set TESSERACT_PATH to its location."
            )
            return

        detections = self.number_detector.detect_numbers(self.image)
        x_pair = self.number_detector.pick_best_pair(detections.get('x', []), axis='x')
        y_pair = self.number_detector.pick_best_pair(detections.get('y', []), axis='y')

        if not x_pair or not y_pair:
            messagebox.showinfo("Smart Auto Calibrate", "Could not confidently read both X and Y axis numbers.")
            return

        for lbl in detections.get('x', []):
            self.create_canvas_highlight(lbl['bbox'], color='#2563eb', duration_ms=6000)
        for lbl in detections.get('y', []):
            self.create_canvas_highlight(lbl['bbox'], color='#16a34a', duration_ms=6000)

        x_sorted = sorted(x_pair, key=lambda e: e['value'])
        y_sorted = sorted(y_pair, key=lambda e: e['value'])
        x_low, x_high = x_sorted[0], x_sorted[1]
        self.create_canvas_highlight(x_low['bbox'], color='#f97316', width=3, duration_ms=6000)
        self.create_canvas_highlight(x_high['bbox'], color='#f97316', width=3, duration_ms=6000)
        y_low, y_high = y_sorted[0], y_sorted[1]
        self.create_canvas_highlight(y_low['bbox'], color='#f97316', width=3, duration_ms=6000)
        self.create_canvas_highlight(y_high['bbox'], color='#f97316', width=3, duration_ms=6000)

        used_contour = self.auto_calibrate(show_messages=False)
        if not used_contour:
            width, height = self.image.size

            def clamp(val, low, high):
                if high < low:
                    high = low
                return max(low, min(high, val))

            y_axis_right_edge = max(lbl['bbox'][0] + lbl['bbox'][2] for lbl in detections['y']) if detections['y'] else \
            x_low['bbox'][0]
            x_axis_top_edge = min(lbl['bbox'][1] for lbl in detections['x']) if detections['x'] else y_low['bbox'][1]
            x_axis_bottom_edge = max(lbl['bbox'][1] + lbl['bbox'][3] for lbl in detections['x']) if detections['x'] else \
            y_low['bbox'][1] + y_low['bbox'][3]

            right_x = int(clamp(max(x_high['cx'], y_axis_right_edge + 40), 40, width - 5))
            left_limit_high = max(10, right_x - 20)
            left_x = int(clamp(min(y_axis_right_edge, x_low['cx']) - 5, 5, left_limit_high))

            top_y_candidate = y_high['cy'] - y_high['bbox'][3]
            top_y = int(clamp(top_y_candidate - 10, 5, max(20, height - 60)))
            bottom_candidate = max(y_low['cy'] + y_low['bbox'][3] + 10, x_axis_bottom_edge + 5)
            bottom_y = int(clamp(bottom_candidate, top_y + 20, height - 5))

            self.calibration_points = [
                (left_x, top_y),
                (right_x, top_y),
                (right_x, bottom_y),
                (left_x, bottom_y)
            ]

        self.snap_calibration_box_to_gridlines(detections.get('x', []), detections.get('y', []))
        self.pull_calibration_box_inside_contour()
        self.x_min_var.set(f"{x_low['value']:.6g}")
        self.x_max_var.set(f"{x_high['value']:.6g}")
        self.y_min_var.set(f"{y_low['value']:.6g}")
        self.y_max_var.set(f"{y_high['value']:.6g}")

        self.redraw_overlays()
        self.escape_mode()
        messagebox.showinfo(
            "Smart Auto Calibrate",
            ("Detected axis numbers and updated calibration.\n"
             "Plot bounds were obtained via contour detection." if used_contour
             else "Detected axis numbers and approximated the plot area from their positions.") +
            "\nPlease verify the corners and adjust if needed."
        )

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
            self.method_results_label.config(text=f"Interpolated to {final_point_count} points.")
        else:
            messagebox.showinfo("Interpolation",
                                "No new points were added. The data is already dense enough or interpolated points fall outside the line mask.")

    def extract_data_method1(self):
        """Original method with color masking and weighted averaging."""
        try:
            if not self.selected_color and not self.line_masking.detected_lines:
                messagebox.showwarning("No Color", "Select a line color or detect lines first.")
                return []
            mask_from_line_detection = False
            if self.line_masking.detected_lines:
                mask = self.line_masking.get_current_line_mask()
                if mask is not None:
                    mask_from_line_detection = True
                    mask = self.apply_axis_guards(mask)
                else:
                    if not self.selected_color:
                        messagebox.showwarning("No Color", "Select a line color or detect lines first.")
                        return []
                    mask = self.enhance_line_detection(self.image_array, self.selected_color)
            else:
                if not self.selected_color:
                    messagebox.showwarning("No Color", "Select a line color or detect lines first.")
                    return []
                mask = self.enhance_line_detection(self.image_array, self.selected_color)
            gray_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)

            mask = self.focus_mask_on_selected_color(mask)
            if mask is None:
                if self.selected_color:
                    messagebox.showwarning(
                        "Color Filter Empty",
                        "No pixels matched the selected color even after auto-expanding the tolerance. "
                        "Try re-picking the color or increase the color tolerance."
                    )
                return []
            if mask_from_line_detection:
                mask = self.remove_outer_border_components(mask)

            x_min, x_max, y_min, y_max = self.compute_sampling_bounds(mask)
            if len(self.calibration_points) == 4:
                calib_x = [p[0] for p in self.calibration_points]
                calib_y = [p[1] for p in self.calibration_points]
                calib_x_min, calib_x_max = max(0, min(calib_x)), min(mask.shape[1] - 1, max(calib_x))
                calib_y_min, calib_y_max = max(0, min(calib_y)), min(mask.shape[0] - 1, max(calib_y))
                calib_width = max(1, calib_x_max - calib_x_min)
                calib_height = max(1, calib_y_max - calib_y_min)
                mask_width = max(0, x_max - x_min)
                mask_height = max(0, y_max - y_min)
                coverage_threshold = 0.55
                if mask_width < coverage_threshold * calib_width:
                    x_min, x_max = calib_x_min, calib_x_max
                if mask_height < coverage_threshold * calib_height:
                    y_min, y_max = calib_y_min, calib_y_max

            target_color_vec = np.array(self.selected_color, dtype=np.int16) if self.selected_color else None
            color_tol_sq = self.get_active_color_tolerance() ** 2

            all_points = []
            for x in range(x_min, x_max + 1):
                if x < 0 or x >= mask.shape[1]:
                    continue
                column_slice = mask[y_min:y_max + 1, x]
                y_indices = np.where(column_slice > 0)[0]

                if (
                        y_indices.size == 0
                        and target_color_vec is not None
                        and y_max >= y_min
                ):
                    column_rgb = self.image_array[y_min:y_max + 1, x].astype(np.int16)
                    if column_rgb.size > 0:
                        diff = column_rgb - target_color_vec
                        dist_sq = np.sum(diff * diff, axis=1)
                        min_idx = int(np.argmin(dist_sq))
                        if dist_sq[min_idx] <= color_tol_sq:
                            y_indices = np.array([min_idx], dtype=np.int32)

                if y_indices.size > 0:
                    y_coords_in_col = y_indices + y_min
                    grayscale_values = gray_image[y_coords_in_col, x]
                    weights = 255.0 - grayscale_values.astype(np.float32)

                    if np.sum(weights) > 0:
                        y_center = np.sum(y_coords_in_col * weights) / np.sum(weights)
                    else:
                        y_center = np.mean(y_coords_in_col)

                    if not self.is_in_exclusion_zone(x, int(y_center)):
                        all_points.append((x, int(y_center)))

            if not all_points:
                return []

            unique_points = sorted(list(dict.fromkeys(all_points)), key=lambda p: p[0])
            unique_points = self.refine_peak_and_valley_tips(mask, unique_points)
            unique_points = self.boost_missing_peaks(mask, unique_points)
            return unique_points

        except Exception as e:
            print(f"Method 1 failed: {e}")
            return []

    def find_line_y_in_column(self, column_data, last_y=None, max_thickness=15):
        """Find the most probable Y-coordinate of the line in a given column."""
        y_coords_detected = np.where(column_data > 0)[0]
        if y_coords_detected.size == 0:
            return None

        diffs = np.diff(y_coords_detected)
        segments = np.split(y_coords_detected, np.where(diffs > 1.5)[0] + 1)

        if not segments or segments[0].size == 0:
            return None

        valid_segments = [s for s in segments if s.size > 0 and s.size <= max_thickness]
        if not valid_segments:
            return None

        segment_centers = [int(np.mean(s)) for s in valid_segments]
        if last_y is None:
            return segment_centers[0]

        closest_y = -1
        min_dist = float('inf')
        for y_center in segment_centers:
            dist = abs(y_center - last_y)
            if dist < min_dist:
                min_dist = dist
                closest_y = y_center

        if min_dist > 50:
            return None

        return closest_y

    def refine_peak_and_valley_tips(self, mask, points, min_shift=2):
        """
        Snap local peaks/valleys back to the very top/bottom pixels of the mask so
        we don't lose sharp tips on small spectral features.
        """
        if mask is None or not points:
            return points

        refined = []
        total = len(points)
        width = mask.shape[1]

        for idx, (x, y) in enumerate(points):
            if x < 0 or x >= width:
                refined.append((x, y))
                continue

            column_hits = np.where(mask[:, x] > 0)[0]
            if column_hits.size == 0:
                refined.append((x, y))
                continue

            window_top = column_hits[:min(3, column_hits.size)]
            window_bottom = column_hits[max(0, column_hits.size - 3):]
            top_y = int(np.median(window_top))
            bottom_y = int(np.median(window_bottom))
            prev_y = points[idx - 1][1] if idx > 0 else y
            next_y = points[idx + 1][1] if idx < total - 1 else y

            is_peak = y <= prev_y and y <= next_y
            is_valley = y >= prev_y and y >= next_y

            if is_peak and (y - top_y) >= min_shift:
                refined.append((x, top_y))
            elif is_valley and (bottom_y - y) >= min_shift:
                refined.append((x, bottom_y))
            else:
                refined.append((x, y))

        return refined

    def boost_missing_peaks(self, mask, points, min_prominence=1.5):
        """
        Add extra samples at sharp peaks that might be skipped by uniform sampling.
        """
        if mask is None or not points:
            return points

        columns = np.where(np.any(mask > 0, axis=0))[0]
        if columns.size < 5:
            return points

        xs = []
        tops = []
        for x in columns:
            rows = np.where(mask[:, x] > 0)[0]
            if rows.size == 0:
                continue
            xs.append(x)
            tops.append(rows[0])

        if len(xs) < 5:
            return points

        xs = np.array(xs)
        tops = np.array(tops, dtype=np.float32)
        window = max(5, (len(tops) // 12) * 2 + 1)
        window = min(window, len(tops) if len(tops) % 2 == 1 else len(tops) - 1)
        if window >= 5 and window < len(tops):
            smooth = savgol_filter(tops, window_length=window, polyorder=2)
        else:
            smooth = tops

        ridge = np.max(smooth) - smooth
        spread = np.ptp(ridge)
        if spread == 0:
            return points

        distance = max(3, len(xs) // 100)
        prominence = max(min_prominence, spread * 0.1)
        peak_indices, _ = find_peaks(ridge, distance=distance, prominence=prominence)

        if not len(peak_indices):
            return points

        existing = points[:]

        def is_near_existing(px, py):
            for ex, ey in existing:
                if abs(ex - px) <= 2 and abs(ey - py) <= 2:
                    return True
            return False

        for idx in peak_indices:
            x_peak = int(xs[idx])
            y_peak = int(tops[idx])
            if not is_near_existing(x_peak, y_peak) and not self.is_in_exclusion_zone(x_peak, y_peak):
                existing.append((x_peak, y_peak))

        existing = sorted(list(dict.fromkeys(existing)), key=lambda p: p[0])
        return existing

    def extract_data_method2(self):
        """Advanced method with thickness filtering and line masking support."""
        try:
            mask_from_line_detection = False
            if self.line_masking.detected_lines:
                final_mask = self.line_masking.get_current_line_mask()
                if final_mask is None:
                    return []
                mask_from_line_detection = True
                final_mask = self.apply_axis_guards(final_mask)
            else:
                if not self.selected_color:
                    messagebox.showwarning("No Color", "Select a line color or detect lines first.")
                    return []
                q_color = self.selected_color
                tolerance = int(self.color_tolerance_var.get()) if hasattr(self, 'color_tolerance_var') else 40
                bgr_color = np.array([q_color[2], q_color[1], q_color[0]])
                lower_bound = np.clip(bgr_color - tolerance, 0, 255)
                upper_bound = np.clip(bgr_color + tolerance, 0, 255)

                bgr_image = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)
                color_mask = cv2.inRange(bgr_image, lower_bound, upper_bound)

                plot_mask = np.ones_like(color_mask, dtype=np.uint8) * 255
                if len(self.calibration_points) == 4 and getattr(self, 'limit_to_calibration', None):
                    if self.limit_to_calibration.get():
                        plot_mask.fill(0)
                        calib_pts_int = np.array(self.calibration_points, dtype=np.int32)
                        cv2.fillPoly(plot_mask, [calib_pts_int], 255)

                kernel = np.ones((3, 3), np.uint8)
                eroded_plot_mask = cv2.erode(plot_mask, kernel, iterations=2)
                final_mask = cv2.bitwise_and(color_mask, eroded_plot_mask)
            final_mask = self.focus_mask_on_selected_color(final_mask)
            if final_mask is None:
                if self.selected_color:
                    messagebox.showwarning(
                        "Color Filter Empty",
                        "No pixels matched the selected color even after auto-expanding the tolerance. "
                        "Try re-picking the color or increase the color tolerance."
                    )
                return []
            if mask_from_line_detection:
                final_mask = self.apply_axis_guards(final_mask)
                final_mask = self.remove_outer_border_components(final_mask)

            x_min, x_max, y_min, y_max = self.compute_sampling_bounds(final_mask)

            mode = self.mode_var.get()
            try:
                value = float(self.value_var.get())
            except ValueError:
                value = 10

            if mode == "Number of points":
                num_points = max(2, int(value))
                samples = np.linspace(x_min, x_max, num_points, dtype=int)
                x_values_to_sample = sorted(set(samples.tolist()))
            else:
                step = max(1, int(value))
                x_values_to_sample = list(range(x_min, x_max + 1, step))

            pixel_points = []
            last_found_y = None
            img_width = self.image_array.shape[1]

            for x in x_values_to_sample:
                if 0 <= x < img_width:
                    col_data = final_mask[:, x]
                    y = self.find_line_y_in_column(col_data, last_found_y)
                    if y is not None and not self.is_in_exclusion_zone(x, y):
                        pixel_points.append((x, y))
                        last_found_y = y

            scan_min_x = max(0, x_min)
            scan_max_x = min(final_mask.shape[1] - 1, x_max)
            scan_min_y = max(0, y_min)
            scan_max_y = min(final_mask.shape[0] - 1, y_max)

            max_point = None
            for y_scan in range(scan_min_y, scan_max_y + 1):
                row = final_mask[y_scan, scan_min_x:scan_max_x + 1]
                if np.any(row > 0):
                    x_idx = scan_min_x + int(np.argmax(row > 0))
                    max_point = (x_idx, y_scan)
                    break

            min_point = None
            for y_scan in range(scan_max_y, scan_min_y - 1, -1):
                row = final_mask[y_scan, scan_min_x:scan_max_x + 1]
                if np.any(row > 0):
                    x_idx = scan_min_x + int(np.argmax(row > 0))
                    min_point = (x_idx, y_scan)
                    break

            for special in (max_point, min_point):
                if special and not any(abs(p[0] - special[0]) < 5 and abs(p[1] - special[1]) < 5 for p in pixel_points):
                    if not self.is_in_exclusion_zone(special[0], special[1]):
                        pixel_points.append(special)

            pixel_points.sort(key=lambda p: p[0])
            pixel_points = self.refine_peak_and_valley_tips(final_mask, pixel_points)
            pixel_points = self.boost_missing_peaks(final_mask, pixel_points)
            pixel_points = self.smooth_non_peak_sections(pixel_points)
            return pixel_points

        except Exception as e:
            print(f"Method 2 failed: {e}")
            return []

    def extract_data_dual(self):
        """Run both detection methods and display based on selected method."""
        if not all([self.image, len(self.calibration_points) == 4]):
            missing = []
            if not self.image:
                missing.append("An image is loaded")
            if len(self.calibration_points) != 4:
                missing.append("4 calibration points are set")
            if not self.selected_color and not self.line_masking.detected_lines:
                missing.append("A line color is selected OR lines are detected")

            messagebox.showwarning("Prerequisites Missing", "Please ensure:\n" + "\n".join(
                [f"{i + 1}. {item}" for i, item in enumerate(missing)]))
            return

        self.clear_data()

        try:
            self.progress_bar.pack(fill=tk.X, pady=2, padx=5)
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = 100
            self.root.update_idletasks()

            self.progress_bar['value'] = 25
            self.root.update_idletasks()
            self.method1_points = self.extract_data_method1()

            self.progress_bar['value'] = 75
            self.root.update_idletasks()
            self.method2_points = self.extract_data_method2()

            self.progress_bar['value'] = 100
            self.root.update_idletasks()

            self.apply_selected_method()

        except Exception as e:
            messagebox.showerror("Extraction Error", f"Failed during data extraction.\nError: {e}")
        finally:
            self.progress_bar['value'] = 0
            self.progress_bar.pack_forget()

    def create_final_mask(self):
        """
        Creates the final binary mask for data extraction by intelligently combining
        line detection, color selection, and the calibrated plot area.
        """
        # 1. Create a base mask from the calibrated plot area
        plot_mask = np.ones(self.image_array.shape[:2], dtype=np.uint8) * 255
        limit_to_calibration = (
                len(self.calibration_points) == 4 and
                getattr(self, 'limit_to_calibration', None) and
                self.limit_to_calibration.get()
        )
        if limit_to_calibration:
            plot_mask.fill(0)
            calib_pts_int = np.array(self.calibration_points, dtype=np.int32)
            cv2.fillPoly(plot_mask, [calib_pts_int], 255)
            # Erode slightly to avoid capturing axis lines
            kernel = np.ones((3, 3), np.uint8)
            plot_mask = cv2.erode(plot_mask, kernel, iterations=2)

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

        source_mask = self.focus_mask_on_selected_color(source_mask)
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

        combined_mask = self.apply_axis_guards(combined_mask)
        return self.refine_line_mask(combined_mask)

    def refine_line_mask(self, mask):
        """
        Slim the mask to avoid bridges/pockets between nearby peaks.
        """
        if mask is None:
            return None
        refined = mask.copy()
        try:
            if min(refined.shape[:2]) >= 3:
                refined = cv2.medianBlur(refined, 3)
        except cv2.error:
            pass
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = remove_shallow_components(refined, min_height_ratio=0.015)
        distance = cv2.distanceTransform(refined, cv2.DIST_L2, 3)
        thickness_limit = 2.0
        if self.image is not None:
            thickness_limit = max(1.5, min(5.0, self.image.height / 450.0))
        slim = (distance <= thickness_limit).astype(np.uint8) * 255
        slim = remove_shallow_components(slim, min_height_ratio=0.015)
        return slim if cv2.countNonZero(slim) > 30 else refined

    def build_skeleton_from_mask(self, mask):
        """
        Lightweight morphological skeletonization used by the multiscale extractor.
        """
        skeleton = np.zeros_like(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = mask.copy()
        while True:
            eroded = cv2.erode(working, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(working, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded
            if cv2.countNonZero(working) == 0:
                break
        return skeleton

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

        if self.exclusion_zones:
            filtered_path = [p for p in path if not self.is_in_exclusion_zone(p[0], p[1])]
            if not filtered_path:
                return []
            path = filtered_path

        # --- NEW: Peak detection to preserve important features ---
        # We find peaks on the negated Y-axis because lower Y values are higher on the screen.
        y_values = -np.array([p[1] for p in path])

        # Much lower prominence to catch subtle features
        # Distance ensures we don't oversample very close peaks
        peak_indices, _ = find_peaks(y_values, prominence=2, width=1, distance=3)
        valley_indices, _ = find_peaks(-y_values, prominence=2, width=1, distance=3)

        feature_indices = set(peak_indices) | set(valley_indices)

        # Also detect inflection points (where curvature changes)
        if len(y_values) > 10:
            # Second derivative to find inflection points
            second_deriv = np.diff(y_values, n=2)
            # Sign changes indicate inflection
            inflection_indices = np.where(np.diff(np.sign(second_deriv)))[0] + 1
            feature_indices |= set(inflection_indices)

        # --- Original sampling logic ---
        mode = self.mode_var.get()
        try:
            value = float(self.value_var.get())
        except ValueError:
            value = 10.0
        value = max(1.0, abs(value))

        sampled_points = []
        if mode == "Number of points":
            num_points = max(2, int(round(value)))
            indices = np.linspace(0, len(path) - 1, num_points, dtype=int)

            # --- MODIFICATION: Add feature indices to the regularly sampled indices ---
            # This ensures that even if regular sampling misses a peak, it gets included.
            final_indices = sorted(list(set(indices) | feature_indices))
            sampled_points = [path[i] for i in final_indices]

        else:  # "Step between points"
            step = max(1, int(round(value)))
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

    def enhance_line_detection(self, image_array, target_color, tolerance=None):
        """
        Creates a binary mask for pixels near the target_color using the original main8 flow.
        """
        if target_color is None or image_array is None:
            return None

        if tolerance is None:
            tolerance = int(self.color_tolerance_var.get()) if hasattr(self, 'color_tolerance_var') else 40
        tolerance = max(0, int(tolerance))

        img_float = image_array.astype(np.float32)
        target_color_float = np.array(target_color, dtype=np.float32)
        dist_sq = np.sum((img_float - target_color_float) ** 2, axis=-1)
        mask = (dist_sq <= tolerance ** 2).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def build_color_mask(self, target_color=None, tolerance=None, image_array=None, clip_to_calibration=True):
        """
        Build a binary mask centered on the selected color using the exact main8 behavior.
        """
        if target_color is None:
            target_color = self.selected_color
        if target_color is None:
            return None
        if image_array is None:
            image_array = self.image_array
        if image_array is None:
            return None

        color_mask = self.enhance_line_detection(image_array, target_color, tolerance)
        if color_mask is None:
            return None

        plot_mask = np.ones_like(color_mask, dtype=np.uint8) * 255
        if len(self.calibration_points) == 4:
            plot_mask.fill(0)
            calib_pts_int = np.array(self.calibration_points, dtype=np.int32)
            cv2.fillPoly(plot_mask, [calib_pts_int], 255)
            kernel = np.ones((3, 3), np.uint8)
            plot_mask = cv2.erode(plot_mask, kernel, iterations=2)

        return cv2.bitwise_and(color_mask, plot_mask)

    def build_color_mask_with_auto_expand(self, target_color=None, tolerance=None, expansion_steps=(12, 25, 40, 60)):
        """
        Try progressively larger tolerances so we always obtain a usable color mask.
        Returns None only if no tolerance up to the largest step produces any pixels.
        """
        if target_color is None:
            target_color = self.selected_color
        if target_color is None or self.image_array is None:
            return None

        base_tolerance = tolerance
        if base_tolerance is None:
            base_tolerance = int(self.color_tolerance_var.get()) if hasattr(self, 'color_tolerance_var') else 40
        base_tolerance = max(0, int(base_tolerance))

        attempted = set()
        tolerances = [base_tolerance]
        for step in expansion_steps:
            candidate = min(255, base_tolerance + int(step))
            if candidate not in tolerances:
                tolerances.append(candidate)

        self._last_color_mask_tolerance = None

        for tol in tolerances:
            if tol in attempted:
                continue
            attempted.add(tol)
            color_mask = self.build_color_mask(target_color=target_color, tolerance=tol)
            if color_mask is not None and cv2.countNonZero(color_mask) > 0:
                self._last_color_mask_tolerance = tol
                return color_mask

        return None

    def get_active_color_tolerance(self):
        """
        Return the tolerance that most recently produced a usable color mask,
        falling back to the UI control's value if needed.
        """
        if getattr(self, '_last_color_mask_tolerance', None) is not None:
            return int(self._last_color_mask_tolerance)
        if hasattr(self, 'color_tolerance_var'):
            try:
                return max(0, int(self.color_tolerance_var.get()))
            except tk.TclError:
                return 40
        return 40

    def focus_mask_on_selected_color(self, mask):
        """
        Restrict an existing mask to the currently selected color; fall back to the pure
        color mask if the intersection is empty (main8 semantics).
        """
        if self.selected_color is None or self.image_array is None:
            return mask

        color_mask = self.build_color_mask_with_auto_expand()
        if color_mask is None or cv2.countNonZero(color_mask) == 0:
            return None

        if mask is None:
            return color_mask

        filtered = cv2.bitwise_and(mask, color_mask)
        if cv2.countNonZero(filtered) == 0:
            return color_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        return cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

    def clear_data(self):
        self.extracted_points, self.real_coordinates = [], []
        self.manual_points = []
        self.method1_points, self.method2_points = [], []
        self.method_results_label.config(text="No data extracted")
        self.canvas.delete("extracted", "extrema")

    def apply_axis_guards(self, mask):
        """Remove tight bands near the calibrated axes to prevent bridging."""
        if mask is None or len(self.calibration_points) != 4:
            return mask

        guard_enabled = (
                getattr(self, 'limit_to_calibration', None) and
                self.limit_to_calibration.get()
        )
        if not guard_enabled:
            return mask

        mask = mask.copy()
        xs = [p[0] for p in self.calibration_points]
        ys = [p[1] for p in self.calibration_points]
        x_min, x_max = int(max(0, min(xs))), int(min(mask.shape[1] - 1, max(xs)))
        y_min, y_max = int(max(0, min(ys))), int(min(mask.shape[0] - 1, max(ys)))
        if x_max <= x_min or y_max <= y_min:
            return mask
        guard_y = max(4, int(0.02 * (y_max - y_min)))
        guard_x = max(4, int(0.02 * (x_max - x_min)))
        top_end = min(y_min + guard_y, y_max + 1)
        bottom_start = max(y_max - guard_y, y_min)
        left_end = min(x_min + guard_x, x_max + 1)
        right_start = max(x_max - guard_x, x_min)
        mask[y_min:top_end, x_min:x_max + 1] = 0
        mask[bottom_start:y_max + 1, x_min:x_max + 1] = 0
        mask[y_min:y_max + 1, x_min:left_end] = 0
        mask[y_min:y_max + 1, right_start:x_max + 1] = 0
        return mask

    def remove_outer_border_components(self, mask, edge_margin=4, min_area_ratio=0.002):
        """Strip large components that hug the outer frame/axes."""
        if mask is None:
            return None
        cleaned = mask.copy()
        h, w = cleaned.shape[:2]
        total_area = h * w
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels <= 1:
            return cleaned
        for idx in range(1, num_labels):
            x, y, width, height, area = stats[idx]
            if area < total_area * min_area_ratio:
                continue
            touches_edge = (
                    x <= edge_margin or y <= edge_margin or
                    x + width >= w - edge_margin or
                    y + height >= h - edge_margin
            )
            if touches_edge:
                cleaned[labels == idx] = 0
        return cleaned if cv2.countNonZero(cleaned) else mask

    def smooth_non_peak_sections(self, points, window=5, limit=2.0):
        """Lightly smooth non-peak sections to remove jitter while keeping sharp peaks."""
        if not points or len(points) < 3:
            return points
        sorted_points = sorted(points, key=lambda p: p[0])
        xs = np.array([p[0] for p in sorted_points])
        ys = np.array([p[1] for p in sorted_points], dtype=np.float32)
        if window % 2 == 0:
            window += 1
        if window >= len(ys):
            window = len(ys) - 1 if len(ys) % 2 == 0 else len(ys)
        if window >= 5:
            smoothed = savgol_filter(ys, window_length=window, polyorder=2, mode='interp')
        else:
            smoothed = ys
        result = []
        for idx, (x, y) in enumerate(zip(xs, ys)):
            prev_y = ys[idx - 1] if idx > 0 else y
            next_y = ys[idx + 1] if idx < len(ys) - 1 else y
            is_peak = y <= prev_y and y <= next_y
            is_valley = y >= prev_y and y >= next_y
            if is_peak or is_valley:
                result.append((int(x), int(round(y))))
                continue
            target = smoothed[idx]
            if abs(target - y) > limit:
                target = y + np.clip(target - y, -limit, limit)
            result.append((int(x), int(round(target))))
        return result

    def compute_sampling_bounds(self, mask):
        """Return a sampling window that covers the mask extents plus calibration area."""
        if mask is None:
            height, width = self.image_array.shape[:2]
            return 0, width - 1, 0, height - 1

        height, width = mask.shape[:2]
        cols = np.where(np.any(mask > 0, axis=0))[0]
        rows = np.where(np.any(mask > 0, axis=1))[0]
        x_min = int(cols[0]) if cols.size else 0
        x_max = int(cols[-1]) if cols.size else width - 1
        y_min = int(rows[0]) if rows.size else 0
        y_max = int(rows[-1]) if rows.size else height - 1

        if len(self.calibration_points) == 4:
            calib_x = [p[0] for p in self.calibration_points]
            calib_y = [p[1] for p in self.calibration_points]
            if getattr(self, 'limit_to_calibration', None) and self.limit_to_calibration.get():
                x_min = max(0, min(calib_x))
                x_max = min(width - 1, max(calib_x))
                y_min = max(0, min(calib_y))
                y_max = min(height - 1, max(calib_y))
            else:
                x_min = max(0, min(x_min, min(calib_x)))
                x_max = min(width - 1, max(x_max, max(calib_x)))
                y_min = max(0, min(y_min, min(calib_y)))
                y_max = min(height - 1, max(y_max, max(calib_y)))

        return x_min, x_max, y_min, y_max

    def get_opposite_color(self, rgb):
        return f'#{255 - rgb[0]:02x}{255 - rgb[1]:02x}{255 - rgb[2]:02x}' if rgb else "#000000"

    def draw_extracted_points(self):
        self.canvas.delete("extracted")
        if not self.extracted_points: return
        point_color = self.get_opposite_color(self.selected_color)
        for x, y in self.extracted_points:
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=point_color, outline="", tags="extracted")

    # --- Smart Auto Calibration Helpers ---
    def snap_calibration_box_to_gridlines(self, x_labels, y_labels, threshold=25):
        """
        Try to align the calibration rectangle with actual axis gridlines or
        fall back to the numeric label bounding boxes when gridlines cannot be detected.
        """
        if len(self.calibration_points) != 4 or self.image is None:
            return
        xs = [p[0] for p in self.calibration_points]
        ys = [p[1] for p in self.calibration_points]
        left = min(xs);
        right = max(xs)
        top = min(ys);
        bottom = max(ys)

        left_snap = self.find_nearest_vertical_gridline(left, threshold)
        right_snap = self.find_nearest_vertical_gridline(right, threshold)
        top_snap = self.find_nearest_horizontal_gridline(top, threshold)
        bottom_snap = self.find_nearest_horizontal_gridline(bottom, threshold)

        if None in (left_snap, right_snap):
            if x_labels:
                left_snap = int(min(x_labels, key=lambda l: l['cx'])['cx'])
                right_snap = int(max(x_labels, key=lambda l: l['cx'])['cx'])
        if None in (top_snap, bottom_snap):
            if y_labels:
                bottom_snap = int(max(y_labels, key=lambda l: l['cy'])['cy'])
                top_snap = int(min(y_labels, key=lambda l: l['cy'])['cy'])

        if None in (left_snap, right_snap, top_snap, bottom_snap):
            return

        self.calibration_points = [
            (left_snap, top_snap),
            (right_snap, top_snap),
            (right_snap, bottom_snap),
            (left_snap, bottom_snap)
        ]

    def pull_calibration_box_inside_contour(self, inset=3):
        if len(self.calibration_points) != 4 or not hasattr(self, 'line_masking'):
            return
        h, w = self.image_array.shape[:2] if self.image_array is not None else (None, None)
        if not h or not w:
            return
        inset = max(0, inset)
        xs = [p[0] for p in self.calibration_points]
        ys = [p[1] for p in self.calibration_points]
        left = int(np.clip(min(xs) + inset, 0, w - 1))
        right = int(np.clip(max(xs) - inset, left + 1, w - 1))
        top = int(np.clip(min(ys) + inset, 0, h - 1))
        bottom = int(np.clip(max(ys) - inset, top + 1, h - 1))
        self.calibration_points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)
        ]

    def find_nearest_vertical_gridline(self, target_x, threshold=25):
        """
        Scan for vertical gridlines by looking for long dark segments in the
        canvas image that run roughly vertically near the target x coordinate.
        """
        if self.image_array is None:
            return None
        gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cols = np.sum(edges > 0, axis=0)
        candidate_indices = np.where(cols > edges.shape[0] * 0.15)[0]
        if candidate_indices.size == 0:
            return None
        nearest = min(candidate_indices, key=lambda idx: abs(idx - target_x))
        return nearest if abs(nearest - target_x) <= threshold else None

    def find_nearest_horizontal_gridline(self, target_y, threshold=25):
        """
        Same as above but for horizontal gridlines.
        """
        if self.image_array is None:
            return None
        gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        rows = np.sum(edges > 0, axis=1)
        candidate_indices = np.where(rows > edges.shape[1] * 0.15)[0]
        if candidate_indices.size == 0:
            return None
        nearest = min(candidate_indices, key=lambda idx: abs(idx - target_y))
        return nearest if abs(nearest - target_y) <= threshold else None

    def create_canvas_highlight(self, bbox, color='#f97316', width=2, duration_ms=5000):
        """Draw a temporary rectangle on the canvas to visualize OCR axis selections."""
        if not bbox or not hasattr(self, 'canvas'):
            return
        x, y, w, h = bbox
        rect_id = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            outline=color, width=width, dash=(4, 2), tags="ocr_highlight"
        )
        if duration_ms > 0:
            self.canvas.after(duration_ms, lambda rid=rect_id: self.canvas.delete(rid))

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
            if getattr(self, 'export_normalized_var', None) and self.export_normalized_var.get():
                y_vals = df['y'].astype(float)
                y_min, y_max = y_vals.min(), y_vals.max()
                if y_max - y_min == 0:
                    df['y_normalized'] = 0.0
                else:
                    df['y_normalized'] = (y_vals - y_min) / (y_max - y_min)

            # Export directly to CSV without any commented metadata headers
            df.to_csv(file_path, index=False)

            messagebox.showinfo("Success", f"Data successfully exported to\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data.\nError: {e}")

    def export_json(self):
        """Exports the extracted data to a JSON file."""
        if not self.real_coordinates:
            messagebox.showwarning("No Data", "No data has been extracted to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        try:
            x_vals = [p[0] for p in self.real_coordinates]
            y_vals = [p[1] for p in self.real_coordinates]
            data_to_export = {'x': x_vals, 'y': y_vals}
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


    def create_canvas_highlight(self, bbox, color='#f97316', width=2, duration_ms=5000):
        if not bbox or not hasattr(self, 'canvas'):
            return
        x, y, w, h = bbox
        rect_id = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            outline=color, width=width, dash=(4, 2),
            tags="ocr_highlight"
        )
        if duration_ms > 0:
            self.canvas.after(duration_ms, lambda rid=rect_id: self.canvas.delete(rid))
