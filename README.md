# Terrain Mapping and Contour Visualization using Python and MATLAB

Date created: April-20-2026

Last modified: May-15-2026

A complete dual-language analysis and visualization package designed to parse mathematical surfaces $f(x, y)$, calculate critical extrema, and interpret scalar functions as interactive geographical terrain models.

---

## Features

- **String-Based Function Parsing**: Directly input complex mathematical expressions like `sin(sqrt(x^2 + y^2))` or polynomials as readable text strings.
- **Topographical Feature Detection**:
  - **Local Maxima (Peaks)**: Identifies strict elevated local supports.
  - **Local Minima (Valleys)**: Detects deep local depressions and clearance zones.
  - **Saddle Points (Connecting Ridges)**: Isolates hyperbolic cross-passes connecting local peaks.
  - **Inconclusive Regions**: Flags flat local plateaus requiring boundary/higher-order checks.
- **Dual Visual Output**:
  - **2D Contour Map**: High-resolution isometric top-down map with customizable contour intervals ($k$).
  - **3D Surface Visualization**: Fully shaded interactive/rotatable mesh view embedded with explicit 3D coordinate annotations.

---

## Implementations Included

### 1. Python Implementation (`contour_map.py`)
Built using **NumPy** for optimized discrete numerical operations and **Plotly** for responsive, browser-ready interactive dashboards. Features automatic boundary isolation and interactive HTML control panels to toggle individual feature trace groups.

#### Installation & Setup
Create a virtual environment and install the required packages:
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Command-Line Usage
Run the script using flexible command-line flags:
```bash
python contour_map.py -f "<function_expression>" -x "<xmin>,<xmax>" -y "<ymin>,<ymax>" -s <step_size> -k <contour_levels>
```

**Available Flags**:
- `-f`, `--function`: Mathematical expression string (e.g., `"-0.1x^4 - 0.1y^4 + 0.8x^2 + 0.8y^2 + 1"`).
- `-x`, `--x_range`: Domain range for the X coordinate (default: `"-3,3"`).
- `-y`, `--y_range`: Domain range for the Y coordinate (default: `"-3,3"`).
- `-s`, `--step_size`: Discrete sampling step size (default: `0.05`).
- `-k`, `--contour_levels`: Target number of contour levels displayed (default: `30`).
- `-h`, `--help`: Display available CLI arguments and instructions.

**Example**:
```bash
python contour_map.py -f "-0.1x^4 - 0.1y^4 + 0.8x^2 + 0.8y^2 + 1" -x "-3,3" -y "-3,3" -s 0.05 -k 30
```

---

### 2. MATLAB Implementation (`contour_map.m`)
Leverages native MATLAB computing combined with the **Symbolic Math Toolbox** to execute analytic partial differentiation ($f_x, f_y$), exact algebraic isolation of real critical solutions via the Second Derivative Test discriminant ($D = f_{xx}f_{yy} - f_{xy}^2$), and produces highly polished dual-panel vector graphics.

#### Running the Script
1. Open `contour_map.m` inside your MATLAB IDE.
2. Modify the configuration parameters directly at the top of the file if desired:
   ```matlab
   f_str = '-0.1*x^4 - 0.1*y^4 + 0.8*x^2 + 0.8*y^2 + 1';
   k = 30; % Number of contour levels
   x_range = [-3, 3];
   y_range = [-3, 3];
   step_size = 0.05;
   ```
3. Click **Run**. The script will automatically compute the derivatives, print detailed classification logs to the MATLAB console, and render a complete native Figure Window incorporating customized plotting symbols and text labels.

---

## Output Structure

Both implementations return robust analytical summaries containing:
- Full coordinates of all interior extrema bounded within the analysis grid domain.
- Evaluated pad elevation/thickness ($z$) at each respective point.
- Extrema categorization cleanly structured into dedicated summary lists and formatted tabular reports.
