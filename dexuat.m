%% Terrain Mapping and Contour Visualization Script
% Fulfills requirements:
% - Inputs: An f(x,y) function defined as a string/symbolic expression.
% - Outputs (Logs): Calculates saddle points, local maxima, and local minima,
%   and prints all the critical points to the console.
% - Outputs (Visual): Generates a 2D contour map with a configurable parameter 'k'
%   defined at the start, and a 3D visualization showing the surface along with
%   annotated coordinates of the maxima and minima.

clear; clc; close all;

%% 1. Input Definitions
% Define the function f(x,y) as a string.
% Examples to try:
%   'x^3 - 3*x + y^3 - 3*y'          (Has 1 min, 1 max, 2 saddle points)
%   'sin(sqrt(x^2 + y^2))'           (Circular waves / ripple pattern)
%   'x^2 + y^2'                      (Simple paraboloid bowl, 1 minimum)
%   'x^2 - y^2'                      (Standard hyperbolic saddle)
% f_str = 'x^3 - 3*x + y^3 - 3*y';

% f(x,y) = -0.1x^4 - 0.1y^4 + 0.8x^2 + 0.8y^2 + 1


f_str = '-0.1*x^4 - 0.1*y^4 + 0.8*x^2 + 0.8*y^2 + 1'; %

% 'k' determines the number of contour levels displayed in the contour map.
k = 30;

% Define the grid range and resolution for plotting the visualizations
x_range = [-3, 3];
y_range = [-3, 3];
step_size = 0.05;

%% 2. Symbolic Setup & Derivatives
disp('==================================================');
disp(['Analyzing Function: f(x,y) = ', f_str]);
disp('==================================================');

syms x y real
% Convert string representation to a symbolic expression compatibly
try
    f_sym = str2sym(f_str);
catch
    f_sym = sym(f_str);
end

% First partial derivatives (Gradient vector components)
fx = diff(f_sym, x);
fy = diff(f_sym, y);

% Second partial derivatives (Hessian matrix components)
fxx = diff(fx, x);
fyy = diff(fy, y);
fxy = diff(fx, y);

% Hessian Determinant / Discriminant: D = fxx*fyy - (fxy)^2
D_sym = fxx * fyy - fxy^2;

%% 3. Calculate Critical Points & Log Outputs
disp('Solving for critical points where gradient is zero...');

% Solve the system of equations fx == 0 and fy == 0
sol = solve([fx == 0, fy == 0], [x, y], 'Real', true);

% Safely extract numeric values for x and y coordinates
if isstruct(sol)
    xc = double(sol.x);
    yc = double(sol.y);
else
    xc = double(sol);
    yc = zeros(size(xc));
end

% Ensure all extracted coordinates are strictly real numbers
real_idx = (imag(xc) == 0) & (imag(yc) == 0);
xc = xc(real_idx);
yc = yc(real_idx);

% Initialize storage arrays for plotting: [x, y, z]
maxima_pts = [];
minima_pts = [];
saddle_pts = [];
inconc_pts = [];

disp(' ');
disp('--- Critical Points Log ---');
if isempty(xc)
    disp('No real critical points found for the given function.');
else
    for i = 1:length(xc)
        x0 = xc(i);
        y0 = yc(i);
        % Evaluate z-coordinate at the critical point
        z0 = double(subs(f_sym, [x, y], [x0, y0]));
        
        % Evaluate the Discriminant D and fxx at the critical point
        D_val = double(subs(D_sym, [x, y], [x0, y0]));
        fxx_val = double(subs(fxx, [x, y], [x0, y0]));
        
        % Classify using the Second Derivative Test
        if D_val > 0
            if fxx_val > 0
                pt_type = 'Local Minimum';
                minima_pts = [minima_pts; x0, y0, z0];
            else
                pt_type = 'Local Maximum';
                maxima_pts = [maxima_pts; x0, y0, z0];
            end
        elseif D_val < 0
            pt_type = 'Saddle Point';
            saddle_pts = [saddle_pts; x0, y0, z0];
        else
            pt_type = 'Inconclusive';
            inconc_pts = [inconc_pts; x0, y0, z0];
        end
        
        % Print detailed log for each critical point
        fprintf('Point %d: (x, y) = (%8.4f, %8.4f) | z = %8.4f | D = %8.4f | Type: %s\n', ...
            i, x0, y0, z0, D_val, pt_type);
    end
end
disp('==================================================');

%% 4. Prepare Evaluation Grid for Visualization
% Generate 2D meshgrid for X and Y domains
[X, Y] = meshgrid(x_range(1):step_size:x_range(2), y_range(1):step_size:y_range(2));

% Convert symbolic expression to an optimized MATLAB anonymous function
f_anon = matlabFunction(f_sym, 'Vars', [x, y]);

% Evaluate Z values over the grid
Z = f_anon(X, Y);

% Handle scalar expansion if the function evaluates to a constant
if isscalar(Z)
    Z = Z * ones(size(X));
end

%% 5. Visual Outputs (2D Contour Map & 3D Surface Visualization)
% Initialize main figure window with premium aesthetics
fig = figure('Name', 'Function Analysis & Visualization', 'Color', [1 1 1], ...
             'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.55]);

% ----------------------------------------------------
% Visual 1: 2D Contour Map using modified parameter k
% ----------------------------------------------------
subplot(1, 2, 1);
hold on;
% Generate contour plot with k levels
contour(X, Y, Z, k, 'LineWidth', 1.2);
colormap(gca, 'parula');
colorbar;

% Plot critical points overlay on the contour map cleanly
h_plots2d = [];
lbl_plots2d = {};

if ~isempty(maxima_pts)
    h_max = plot(maxima_pts(:,1), maxima_pts(:,2), '^r', 'MarkerFaceColor', 'r', ...
                 'MarkerSize', 8, 'LineWidth', 1);
    h_plots2d(end+1) = h_max;
    lbl_plots2d{end+1} = 'Local Maxima';
end
if ~isempty(minima_pts)
    h_min = plot(minima_pts(:,1), minima_pts(:,2), 'vb', 'MarkerFaceColor', 'b', ...
                 'MarkerSize', 8, 'LineWidth', 1);
    h_plots2d(end+1) = h_min;
    lbl_plots2d{end+1} = 'Local Minima';
end
if ~isempty(saddle_pts)
    h_sad = plot(saddle_pts(:,1), saddle_pts(:,2), 'og', 'MarkerFaceColor', 'g', ...
                 'MarkerSize', 8, 'LineWidth', 1);
    h_plots2d(end+1) = h_sad;
    lbl_plots2d{end+1} = 'Saddle Points';
end

% Enhance contour map aesthetics
title(['2D Contour Map (k = ', num2str(k), ' levels)'], 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X Axis', 'FontSize', 10);
ylabel('Y Axis', 'FontSize', 10);
grid on;
box on;
axis equal;
axis([x_range(1), x_range(2), y_range(1), y_range(2)]);

% Show legend cleanly if points exist
if ~isempty(h_plots2d)
    legend(h_plots2d, lbl_plots2d, 'Location', 'best');
end
hold off;

% ----------------------------------------------------
% Visual 2: 3D Surface Visualization with Annotations
% ----------------------------------------------------
subplot(1, 2, 2);
hold on;
% Render smooth 3D surface plot
surf(X, Y, Z, 'EdgeAlpha', 0.25, 'FaceAlpha', 0.85);
shading interp;
colormap(gca, 'parula');
colorbar;

h_plots3d = [];
lbl_plots3d = {};

% Plot and visually label coordinates of Local Maxima on 3D view
if ~isempty(maxima_pts)
    h_max3d = plot3(maxima_pts(:,1), maxima_pts(:,2), maxima_pts(:,3), '^r', ...
                    'MarkerFaceColor', 'r', 'MarkerSize', 10, 'LineWidth', 1.5);
    h_plots3d(end+1) = h_max3d;
    lbl_plots3d{end+1} = 'Local Maxima';
    
    for i = 1:size(maxima_pts, 1)
        coord_label = sprintf('  Max(%.2f, %.2f, %.2f)', maxima_pts(i,1), maxima_pts(i,2), maxima_pts(i,3));
        text(maxima_pts(i,1), maxima_pts(i,2), maxima_pts(i,3), coord_label, ...
             'Color', [0.8 0 0], 'FontWeight', 'bold', 'FontSize', 10);
    end
end

% Plot and visually label coordinates of Local Minima on 3D view
if ~isempty(minima_pts)
    h_min3d = plot3(minima_pts(:,1), minima_pts(:,2), minima_pts(:,3), 'vb', ...
                    'MarkerFaceColor', 'b', 'MarkerSize', 10, 'LineWidth', 1.5);
    h_plots3d(end+1) = h_min3d;
    lbl_plots3d{end+1} = 'Local Minima';
    
    for i = 1:size(minima_pts, 1)
        coord_label = sprintf('  Min(%.2f, %.2f, %.2f)', minima_pts(i,1), minima_pts(i,2), minima_pts(i,3));
        text(minima_pts(i,1), minima_pts(i,2), minima_pts(i,3), coord_label, ...
             'Color', [0 0 0.8], 'FontWeight', 'bold', 'FontSize', 10);
    end
end

% Plot Saddle Points on 3D view
if ~isempty(saddle_pts)
    h_sad3d = plot3(saddle_pts(:,1), saddle_pts(:,2), saddle_pts(:,3), 'og', ...
                    'MarkerFaceColor', 'g', 'MarkerSize', 8, 'LineWidth', 1.5);
    h_plots3d(end+1) = h_sad3d;
    lbl_plots3d{end+1} = 'Saddle Points';
end

% Enhance 3D plot aesthetics
title('3D Surface Visualization', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X Axis', 'FontSize', 10);
ylabel('Y Axis', 'FontSize', 10);
zlabel('Z Axis', 'FontSize', 10);
grid on;
view(-37.5, 30);

if ~isempty(h_plots3d)
    legend(h_plots3d, lbl_plots3d, 'Location', 'best');
end
hold off;

% Set master title for the entire figure window
try
    sgtitle(['Visualization & Critical Points of f(x,y) = ', f_str], 'FontSize', 14, 'FontWeight', 'bold');
catch
    % Fallback for older MATLAB versions lacking sgtitle
end


%% 6. DYNAMIC SLOPE DISPLAY ON MOUSE CLICK 

% Convert first-order partial derivative expressions (fx, fy) to Anonymous Functions
fx_anon = matlabFunction(fx, 'Vars', [x, y]);
fy_anon = matlabFunction(fy, 'Vars', [x, y]);

% Enable Data Cursor Mode for the figure and assign the custom interactive callback
dcm_obj = datacursormode(fig);
set(dcm_obj, 'Enable', 'on', 'UpdateFcn', @(obj, event) myDynamicDataTip(obj, event, f_anon, fx_anon, fy_anon));

% Local function to dynamically compute coordinates and the actual slope value upon mouse click
function txt = myDynamicDataTip(~, event_obj, f_anon, fx_anon, fy_anon)
    % Extract exact coordinates of the target cursor position
    pos = event_obj.Position;
    click_x = pos(1);
    click_y = pos(2);
    
    % Compute the real Z level and localized slope magnitude at the clicked point
    current_z = f_anon(click_x, click_y);
    gx = fx_anon(click_x, click_y);
    gy = fy_anon(click_x, click_y);
    current_slope = sqrt(gx^2 + gy^2);
    
    % Format text lines displayed inside the interactive cursor box
    txt = {['X: ', num2str(click_x, '%.4f')], ...
           ['Y: ', num2str(click_y, '%.4f')], ...
           ['Z (Level): ', num2str(current_z, '%.4f')], ...
           ['Slope: ', num2str(current_slope, '%.4f')]};
end