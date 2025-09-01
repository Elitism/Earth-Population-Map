import tkinter as tk
from tkinter import messagebox
import sys
import subprocess
import importlib.util
import math
import csv
from PIL import Image
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pyopengltk import OpenGLFrame
from scipy.spatial import cKDTree

# --- Dependency Check ---
def check_and_install_dependencies():
    required_packages = {
        'numpy': 'numpy',
        'PyOpenGL': 'OpenGL',
        'PyOpenGL_accelerate': 'OpenGL_accelerate',
        'pyopengltk': 'pyopengltk',
        'Pillow': 'PIL',
        'scipy': 'scipy'
    }
    missing_packages = []

    for package, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package)

    if missing_packages:
        root = tk.Tk()
        root.withdraw()
        package_list = "\n".join([f"- {pkg}" for pkg in missing_packages])
        response = messagebox.askyesno(
            "Missing Dependencies",
            "The following required libraries are not installed:\n\n"
            f"{package_list}\n\n"
            "Do you want to attempt to install them now? This requires an internet connection."
        )

        if response:
            for package in missing_packages:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except subprocess.CalledProcessError:
                    messagebox.showerror(
                        "Installation Failed",
                        f"Failed to install '{package}'.\nPlease try installing it manually:\n"
                        f"pip install {package}"
                    )
                    root.destroy()
                    sys.exit(f"Error: Failed to install '{package}'.")
            messagebox.showinfo(
                "Installation Complete",
                "All dependencies have been installed. Please restart the application."
            )
            root.destroy()
            sys.exit()
        else:
            messagebox.showwarning(
                "Installation Aborted",
                "Installation was cancelled. The application cannot run without its dependencies."
            )
            root.destroy()
            sys.exit("Aborted by user.")

check_and_install_dependencies()

# --- OpenGL Frame ---
class EarthViewerFrame(OpenGLFrame):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.points_vbo = None
        self.colors_vbo = None
        self.point_count = 0
        self.rotation_angle_x = 0  # Point cloud X
        self.rotation_angle_y = 0  # Point cloud Y
        self.sphere_rotation_x = 90  # Sphere X
        self.sphere_rotation_y = 0  # Sphere Y
        self.sphere_rotation_z = 270  # Sphere Z
        self.last_mouse_pos = {'x': 0, 'y': 0}
        self.is_dragging = False
        self.zoom = -8
        
        # Heatmap parameters
        self.color_scheme = 'plasma'  # plasma, viridis, hot, cool, rainbow
        self.use_logarithmic = True
        self.alpha_blending = True

        # --- Load Points ---
        try:
            points_list, populations_list = self.generate_points_from_csv("GeoNames_Cleaned.csv", radius=2.5)
            if points_list:
                self.points_vbo = np.array(points_list, dtype=np.float32)
                self.populations = np.array(populations_list, dtype=np.float32)
                self.point_count = len(self.points_vbo)
                # --- Compute colors from population data ---
                self.colors_vbo = self.compute_population_colors(self.populations)
        except FileNotFoundError:
            print("Error: 'GeoNames_Cleaned.csv' not found. Cannot display points.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # --- Mouse Events ---
        self.bind("<ButtonPress-1>", self.on_mouse_press)
        self.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Button-4>", self.on_mouse_wheel)
        self.bind("<Button-5>", self.on_mouse_wheel)

    # OpenGL initialization
    def initgl(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_POINT_SMOOTH)
        
        # Enable alpha blending for better heatmap visualization
        if self.alpha_blending:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glPointSize(4.0)  # Slightly larger points for better visibility

        # --- Texture ---
        self.texture_id = glGenTextures(1)
        self.load_texture("earth_texture.jpg")  # Replace with your texture

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.height > 0:
            gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def redraw(self):
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_angle_x, 1, 0, 0)  # Point cloud X
        glRotatef(self.rotation_angle_y, 0, 1, 0)  # Point cloud Y

        # --- Draw Textured Sphere (X,Y,Z rotation) ---
        glPushMatrix()
        glRotatef(self.sphere_rotation_x, 1, 0, 0)
        glRotatef(self.sphere_rotation_y, 0, 1, 0)
        glRotatef(self.sphere_rotation_z, 0, 0, 1)
        self.draw_textured_sphere(radius=2.5)
        glPopMatrix()

        # --- Draw Point Cloud with Enhanced Heatmap Colors ---
        if self.points_vbo is not None and self.point_count > 0:
            # Disable texture for points
            glDisable(GL_TEXTURE_2D)
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.points_vbo)
            glColorPointer(4, GL_FLOAT, 0, self.colors_vbo)  # 4 components for RGBA
            glDrawArrays(GL_POINTS, 0, self.point_count)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            
            # Re-enable texture
            glEnable(GL_TEXTURE_2D)

    def generate_points_from_csv(self, filepath, radius):
        points = []
        populations = []
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    # Try different possible population column names
                    pop = 0
                    for pop_col in ['Population', 'population', 'POPULATION', 'Pop', 'pop']:
                        if pop_col in row and row[pop_col]:
                            try:
                                pop = float(row[pop_col])
                                break
                            except ValueError:
                                continue
                    
                    theta = math.radians(90 - lat)
                    phi = math.radians(lon)
                    x = -radius * math.sin(theta) * math.cos(phi)  # Inverted X-axis
                    y = radius * math.cos(theta)
                    z = radius * math.sin(theta) * math.sin(phi)
                    points.append((x, y, z))
                    populations.append(pop)
                except (ValueError, KeyError):
                    pass
        return points, populations

    def compute_population_colors(self, populations):
        """Compute colors based on population data"""
        print(f"Computing population colors for {len(populations)} points...")
        
        # Handle zero populations and apply logarithmic scaling if enabled
        if self.use_logarithmic:
            # Add 1 to avoid log(0), then apply log
            pop_values = np.log1p(populations)  # log(1 + x)
        else:
            pop_values = populations.copy()
        
        # Filter out zero/negative values for better scaling
        valid_pops = pop_values[pop_values > 0]
        if len(valid_pops) == 0:
            # All populations are zero, use uniform color
            normalized_pops = np.zeros_like(pop_values)
        else:
            # Normalize to 0-1 range
            min_pop = valid_pops.min()
            max_pop = valid_pops.max()
            
            if max_pop > min_pop:
                # Scale all values, setting negatives/zeros to 0
                normalized_pops = np.maximum(0, (pop_values - min_pop) / (max_pop - min_pop))
            else:
                normalized_pops = np.zeros_like(pop_values)
        
        # Generate colors based on selected scheme
        colors = self.generate_color_scheme(normalized_pops)
        
        print(f"Population range: {populations.min():.0f} - {populations.max():.0f}")
        if self.use_logarithmic:
            print(f"Log-scaled range: {pop_values.min():.2f} - {pop_values.max():.2f}")
        
        return colors

    def generate_color_scheme(self, normalized_values):
        """Generate colors based on the selected color scheme"""
        n = len(normalized_values)
        colors = np.zeros((n, 4), dtype=np.float32)  # RGBA
        
        if self.color_scheme == 'plasma':
            # Plasma-like colormap: dark blue -> purple -> pink -> yellow
            for i, val in enumerate(normalized_values):
                if val < 0.25:
                    # Dark blue to purple
                    t = val * 4
                    colors[i] = [0.1 + 0.4*t, 0.0, 0.3 + 0.4*t, 0.8]
                elif val < 0.5:
                    # Purple to pink
                    t = (val - 0.25) * 4
                    colors[i] = [0.5 + 0.3*t, 0.1*t, 0.7 - 0.2*t, 0.9]
                elif val < 0.75:
                    # Pink to orange
                    t = (val - 0.5) * 4
                    colors[i] = [0.8 + 0.2*t, 0.1 + 0.4*t, 0.5 - 0.5*t, 1.0]
                else:
                    # Orange to yellow
                    t = (val - 0.75) * 4
                    colors[i] = [1.0, 0.5 + 0.5*t, 0.0, 1.0]
        
        elif self.color_scheme == 'viridis':
            # Viridis-like: dark purple -> blue -> green -> yellow
            for i, val in enumerate(normalized_values):
                if val < 0.25:
                    t = val * 4
                    colors[i] = [0.3*t, 0.0, 0.4 + 0.2*t, 0.8]
                elif val < 0.5:
                    t = (val - 0.25) * 4
                    colors[i] = [0.3 - 0.1*t, 0.2*t, 0.6 + 0.2*t, 0.9]
                elif val < 0.75:
                    t = (val - 0.5) * 4
                    colors[i] = [0.2*t, 0.2 + 0.6*t, 0.8 - 0.4*t, 1.0]
                else:
                    t = (val - 0.75) * 4
                    colors[i] = [0.2 + 0.8*t, 0.8 + 0.2*t, 0.4 - 0.4*t, 1.0]
        
        elif self.color_scheme == 'hot':
            # Hot colormap: black -> red -> yellow -> white
            for i, val in enumerate(normalized_values):
                if val < 0.33:
                    # Black to red
                    t = val * 3
                    colors[i] = [t, 0.0, 0.0, 0.7 + 0.3*t]
                elif val < 0.66:
                    # Red to yellow
                    t = (val - 0.33) * 3
                    colors[i] = [1.0, t, 0.0, 1.0]
                else:
                    # Yellow to white
                    t = (val - 0.66) * 3
                    colors[i] = [1.0, 1.0, t, 1.0]
        
        elif self.color_scheme == 'cool':
            # Cool colormap: cyan to magenta
            for i, val in enumerate(normalized_values):
                colors[i] = [val, 1.0 - val, 1.0, 0.8]
        
        elif self.color_scheme == 'rainbow':
            # Rainbow colormap
            for i, val in enumerate(normalized_values):
                hue = val * 300  # 0 to 300 degrees (red to magenta)
                rgb = self.hsv_to_rgb(hue, 1.0, 1.0)
                colors[i] = [rgb[0], rgb[1], rgb[2], 0.9]
        
        else:  # default green-red
            # Original green-red gradient
            for i, val in enumerate(normalized_values):
                colors[i] = [val, 1.0 - val, 0.0, 0.8]
        
        return colors

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)

    def update_heatmap_settings(self, scheme=None, logarithmic=None):
        """Update heatmap parameters and recompute colors"""
        if scheme is not None:
            self.color_scheme = scheme
        if logarithmic is not None:
            self.use_logarithmic = logarithmic
        
        if hasattr(self, 'populations') and self.populations is not None:
            self.colors_vbo = self.compute_population_colors(self.populations)
            self.redraw()

    def draw_textured_sphere(self, radius):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        quad = gluNewQuadric()
        gluQuadricTexture(quad, GL_TRUE)
        gluSphere(quad, radius, 36, 18)
        gluDeleteQuadric(quad)

    def load_texture(self, filepath):
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_data = np.array(list(img.getdata()), np.uint8)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, img_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        except Exception as e:
            print(f"Error loading texture '{filepath}': {e}")

    # --- Mouse handlers ---
    def on_mouse_press(self, event):
        self.is_dragging = True
        self.last_mouse_pos['x'] = event.x
        self.last_mouse_pos['y'] = event.y

    def on_mouse_release(self, event):
        self.is_dragging = False

    def on_mouse_drag(self, event):
        if self.is_dragging:
            dx = event.x - self.last_mouse_pos['x']
            dy = event.y - self.last_mouse_pos['y']
            self.rotation_angle_y += dx * 0.5
            self.rotation_angle_x += dy * 0.5
            self.last_mouse_pos['x'] = event.x
            self.last_mouse_pos['y'] = event.y

    def on_mouse_wheel(self, event):
        # More precise zoom increments
        zoom_increment = 0.1  # Much smaller increment for finer control
        
        if event.num == 5 or event.delta == -120:
            self.zoom -= zoom_increment  # Zoom out
        elif event.num == 4 or event.delta == 120:
            self.zoom += zoom_increment  # Zoom in
        
        # Tighter zoom limits for better control
        self.zoom = max(min(self.zoom, -1.0), -50.0)  # Closer minimum, further maximum

# --- Main Window ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("3D Earth Viewer with Enhanced Density Heatmap")
    root.geometry("900x800")

    # --- Create controls frame ---
    controls_frame = tk.Frame(root)
    controls_frame.pack(side='top', fill='x', padx=5, pady=5)

    # --- Rotation sliders ---
    rotation_frame = tk.Frame(controls_frame)
    rotation_frame.pack(side='top', fill='x', pady=2)

    x_slider = tk.Scale(rotation_frame, from_=0, to=360, resolution=0.01, 
                       label="Sphere Rotation X", orient='horizontal')
    x_slider.pack(side='left', fill='x', expand=True, padx=2)
    x_slider.set(90.00)
    
    y_slider = tk.Scale(rotation_frame, from_=0, to=360, resolution=0.01,
                       label="Sphere Rotation Y", orient='horizontal')
    y_slider.pack(side='left', fill='x', expand=True, padx=2)
    
    z_slider = tk.Scale(rotation_frame, from_=0, to=360, resolution=0.01,
                       label="Sphere Rotation Z", orient='horizontal')
    z_slider.pack(side='left', fill='x', expand=True, padx=2)
    z_slider.set(270.00)

    # --- Heatmap controls ---
    heatmap_frame = tk.Frame(controls_frame)
    heatmap_frame.pack(side='top', fill='x', pady=2)

    # Color scheme dropdown
    tk.Label(heatmap_frame, text="Color Scheme:").pack(side='left', padx=5)
    color_var = tk.StringVar(value='plasma')
    color_menu = tk.OptionMenu(heatmap_frame, color_var, 'plasma', 'viridis', 'hot', 'cool', 'rainbow', 'green-red')
    color_menu.pack(side='left', padx=2)

    # Logarithmic checkbox
    log_var = tk.BooleanVar(value=True)
    log_check = tk.Checkbutton(heatmap_frame, text="Logarithmic Scale", variable=log_var)
    log_check.pack(side='left', padx=5)

    # --- OpenGL Frame ---
    app = EarthViewerFrame(root, width=900, height=650)
    app.pack(fill="both", expand=True)

    # --- Connect controls ---
    def update_sphere_rotation(_=None):
        app.sphere_rotation_x = x_slider.get()
        app.sphere_rotation_y = y_slider.get()
        app.sphere_rotation_z = z_slider.get()
        app.redraw()

    def update_heatmap(_=None):
        app.update_heatmap_settings(
            scheme=color_var.get(),
            logarithmic=log_var.get()
        )

    x_slider.config(command=update_sphere_rotation)
    y_slider.config(command=update_sphere_rotation)
    z_slider.config(command=update_sphere_rotation)
    
    color_var.trace('w', lambda *args: update_heatmap())
    log_var.trace('w', lambda *args: update_heatmap())

    # Create legend
    legend_frame = tk.Frame(root)
    legend_frame.pack(side='bottom', fill='x', padx=5, pady=2)
    tk.Label(legend_frame, text="Heatmap: Low Population ← → High Population", 
             font=('Arial', 10)).pack()

    app.animate = 1
    root.mainloop()