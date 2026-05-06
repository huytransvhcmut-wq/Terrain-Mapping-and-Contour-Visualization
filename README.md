# Terrain Mapping and Contour Visualization using Python 

Date created: April-20-2026

## Introduction

Build a program to generate contour maps of simple functions and interpret them as terrain. Identify peaks, valleys, and slopes.

## Inputs

- A function of two variables, f(x, y), which defines the surface we want to visualize. Default will be `f(x, y) = sin(sqrt(x^2 + y^2))`.
- The range of x and y values to view, which determines the area of the contour map. Default will be from -10 to 10 for both x and y.
- The step size for x and y values, which determines the resolution of the contour map. Default will be 0.1.

## Output Requirements

- A contour map that represents the levels of the function f(x, y) in a 2D plane.
- Peaks, valleys and slopes of the function can be identified from the contour map.
- The contour map will be displayed using Plotly, which allows for interactive visualization.
- The program will also print out the coordinates of the peaks and valleys identified in the contour map.
- The program will be implemented in Python and will use libraries such as NumPy for numerical operations and Plotly for visualization.
- The program will be able to handle complex functions and will be able to visualize them in a 2D plane.
- The program will able to convert the mathematical expression to a numpy expression and will be able to visualize it in a 2D plane (For example, if the function is f(x, y) = sin(sqrt(x^2 + y^2)), the program will be able to convert it to a numpy expression and will be able to visualize it in a 2D plane).


## Running the script 
- First, create a virtual environment and install the required libraries using the command:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment 
## For Windows
.venv\Scripts\activate
## For Unix or MacOS
source .venv/bin/activate
# Install the required libraries
pip install -r requirements.txt
```

- Then, run the script using the command:
```bash
python contour_map.py -f [your_function] -x [x range to view] -y [y range to view] -s [step size for x and y]
```

Flags: 
- `-f` or `--function`: The function of two variables to visualize, e.g., "sin(sqrt(x^2 + y^2))".
- `-x` or `--x_range`: The range of x values to view,e.g., "-10,10".
- `-y` or `--y_range`: The range of y values to view,e.g., "-10,10".
- `-s` or `--step_size`: The step size for x and y values, e.g., "0.1".
- `-h` or `--help`: Show the help message and exit.

For example, to visualize the function f(x, y) = sin(sqrt(x^2 + y^2)) over the range of -10 to 10 for both x and y with a step size of 0.1, you would run:

```bash
python contour_map.py -f "sin(sqrt(x^2 + y^2))" -x "-10,10" -y "-10,10" -s "0.1"
```

