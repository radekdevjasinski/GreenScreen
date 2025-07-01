import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk
import threading
import time


class GreenScreenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Green Screen Color Keying")
        self.root.geometry("1060x500")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Default color to key out (green)
        self.key_color = [0, 255, 0]  # BGR format
        self.tolerance = 50
        self.running = False
        self.current_frame = None  # Store current frame for color picking
        self.color_picking_mode = False  # Track if we're in color picking mode

        # Background image (white by default)
        self.background = np.ones((480, 640, 3), dtype=np.uint8) * 255

        self.setup_gui()
        self.start_camera()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left: Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 10))
        self.video_label.bind("<Button-1>", self.on_frame_click)

        # Right: Settings panel
        settings_frame = ttk.Frame(main_frame)
        settings_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))

        # Color selection frame
        color_frame = ttk.LabelFrame(settings_frame, text="Color Selection", padding="10")
        color_frame.pack(fill=tk.X, pady=(0, 10))

        self.color_button = tk.Button(color_frame, text="Select Key Color",
                                      command=self.select_color, width=15, height=2)
        self.color_button.grid(row=0, column=0, padx=(0, 10))

        self.pick_from_frame_button = tk.Button(color_frame, text="Pick from Camera",
                                                command=self.toggle_color_picking,
                                                width=15, height=2, bg='lightblue')
        self.pick_from_frame_button.grid(row=1, column=0, padx=(0, 10), pady=(5, 0))

        ttk.Label(color_frame, text="Current Color:").grid(row=0, column=1, padx=(0, 5))
        self.color_display = tk.Canvas(color_frame, width=50, height=30, bg='green')
        self.color_display.grid(row=0, column=2)

        self.instruction_label = ttk.Label(color_frame, text="", foreground='red')
        self.instruction_label.grid(row=1, column=1, columnspan=2, pady=(5, 0))

        self.update_color_button()

        # Tolerance adjustment frame
        tolerance_frame = ttk.LabelFrame(settings_frame, text="Tolerance Settings", padding="10")
        tolerance_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(tolerance_frame, text="Tolerance:").grid(row=0, column=0, padx=(0, 10))
        self.tolerance_var = tk.IntVar(value=self.tolerance)
        self.tolerance_slider = ttk.Scale(tolerance_frame, from_=0, to=200,
                                          variable=self.tolerance_var, orient=tk.HORIZONTAL,
                                          length=200, command=self.update_tolerance)
        self.tolerance_slider.grid(row=0, column=1, padx=(0, 10))
        self.tolerance_label = ttk.Label(tolerance_frame, text=str(self.tolerance))
        self.tolerance_label.grid(row=0, column=2)

        # Background options frame
        bg_frame = ttk.LabelFrame(settings_frame, text="Background Options", padding="10")
        bg_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(bg_frame, text="White Background",
                   command=self.set_white_background).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(bg_frame, text="Black Background",
                   command=self.set_black_background).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(bg_frame, text="Load Background Image",
                   command=self.load_background).grid(row=0, column=2)

        # Control buttons
        control_frame = ttk.Frame(settings_frame)
        control_frame.pack(pady=(10, 0))

        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="Reset", command=self.reset_settings).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="Exit", command=self.cleanup_and_exit).grid(row=0, column=2)

    def select_color(self):
        """Open color picker dialog"""
        color = colorchooser.askcolor(title="Select color to key out")
        if color[0]:  # If color was selected
            # Convert RGB to BGR for OpenCV
            rgb = color[0]
            self.key_color = [int(rgb[2]), int(rgb[1]), int(rgb[0])]
            self.update_color_button()

    def toggle_color_picking(self):
        """Toggle color picking mode"""
        self.color_picking_mode = not self.color_picking_mode
        if self.color_picking_mode:
            self.pick_from_frame_button.configure(text="Cancel Picking", bg='orange')
            self.instruction_label.configure(text="Click on camera frame to pick color")
        else:
            self.pick_from_frame_button.configure(text="Pick from Camera", bg='lightblue')
            self.instruction_label.configure(text="")

    def on_frame_click(self, event):
        """Handle click on camera frame to pick color"""
        if not self.color_picking_mode or self.current_frame is None:
            return

        # Get click coordinates
        x, y = event.x, event.y

        # Get frame dimensions and label dimensions
        if hasattr(self.video_label, 'image') and self.video_label.image:
            # Calculate scaling factor between displayed image and actual frame
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            frame_height, frame_width = self.current_frame.shape[:2]

            # Convert click coordinates to frame coordinates
            if label_width > 0 and label_height > 0:
                frame_x = int((x / label_width) * frame_width)
                frame_y = int((y / label_height) * frame_height)

                # Ensure coordinates are within frame bounds
                frame_x = max(0, min(frame_x, frame_width - 1))
                frame_y = max(0, min(frame_y, frame_height - 1))

                # Get BGR color at clicked position
                bgr_color = self.current_frame[frame_y, frame_x]
                self.key_color = [int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])]

                # Update UI
                self.update_color_button()
                self.toggle_color_picking()  # Exit color picking mode

    def update_color_button(self):
        """Update the color button and display"""
        # Convert BGR to RGB for display
        rgb_color = f"#{self.key_color[2]:02x}{self.key_color[1]:02x}{self.key_color[0]:02x}"
        self.color_button.configure(bg=rgb_color)
        self.color_display.configure(bg=rgb_color)

    def update_tolerance(self, value):
        """Update tolerance value from slider"""
        self.tolerance = int(float(value))
        self.tolerance_label.configure(text=str(self.tolerance))

    def set_white_background(self):
        """Set background to white"""
        height, width = 480, 640
        self.background = np.ones((height, width, 3), dtype=np.uint8) * 255

    def set_black_background(self):
        """Set background to black"""
        height, width = 480, 640
        self.background = np.zeros((height, width, 3), dtype=np.uint8)

    def load_background(self):
        """Load custom background image"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            bg_img = cv2.imread(file_path)
            if bg_img is not None:
                # Resize to match camera resolution
                self.background = cv2.resize(bg_img, (640, 480))

    def create_mask(self, frame):
        """Create improved mask for color keying"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        key_hsv = cv2.cvtColor(np.uint8([[self.key_color]]), cv2.COLOR_BGR2HSV)[0][0]

        h, s, v = int(key_hsv[0]), int(key_hsv[1]), int(key_hsv[2])
        tolerance = self.tolerance

        lower_bound = np.array([
            max(0, h - tolerance // 2),
            max(0, s - tolerance),
            max(0, v - tolerance)
        ], dtype=np.uint8)
        upper_bound = np.array([
            min(179, h + tolerance // 2),
            min(255, s + tolerance),
            min(255, v + tolerance)
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Morphological cleaning
        kernel_close = np.ones((5, 5), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Feather (strong blur to smooth edges)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        return mask

    def apply_green_screen(self, frame):
        """Apply green screen effect"""
        # Create mask
        mask = self.create_mask(frame)

        # Normalize mask to 0-1 range
        mask_norm = mask.astype(float) / 255

        # Create inverse mask
        mask_inv = 1.0 - mask_norm

        # Apply green screen effect
        result = np.zeros_like(frame)
        for i in range(3):  # For each color channel
            result[:, :, i] = (frame[:, :, i] * mask_inv +
                               self.background[:, :, i] * mask_norm)

        return result.astype(np.uint8)

    def update_frame(self):
        """Update video frame"""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Store current frame for color picking
            self.current_frame = frame.copy()

            # Apply green screen effect (only if not in color picking mode)
            if self.color_picking_mode:
                processed_frame = frame  # Show original frame for color picking
            else:
                processed_frame = self.apply_green_screen(frame)

            # Convert BGR to RGB for Tkinter
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)

            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference

        # Schedule next update
        self.root.after(30, self.update_frame)  # ~33 FPS

    def start_camera(self):
        """Start camera capture"""
        self.running = True
        self.update_frame()

    def stop_camera(self):
        """Stop camera capture"""
        self.running = False

    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def reset_settings(self):
        """Reset all settings to default"""
        self.key_color = [0, 255, 0]  # Green
        self.tolerance = 50
        self.tolerance_var.set(self.tolerance)
        self.tolerance_label.configure(text=str(self.tolerance))
        self.update_color_button()
        self.set_white_background()
        # Exit color picking mode if active
        if self.color_picking_mode:
            self.toggle_color_picking()

    def cleanup_and_exit(self):
        """Clean up resources and exit"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GreenScreenApp(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.cleanup_and_exit)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.cleanup_and_exit()


if __name__ == "__main__":
    main()