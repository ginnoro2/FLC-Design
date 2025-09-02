%% CEC'2005 Benchmark Functions Implementation
% This file implements selected CEC'2005 benchmark functions for optimization comparison
% Functions selected: F1 (Shifted Sphere) and F6 (Shifted Rosenbrock)
%
% Reference: Suganthan, P. N., et al. "Problem definitions and evaluation criteria 
% for the CEC 2005 special session on real-parameter optimization." (2005)

%% Function 1: Shifted Sphere Function (F1)
function f = cec2005_f1(x, shift_data)
    % F1: Shifted Sphere Function
    % Global minimum: f(x*) = -450, where x* = shift_data
    % Search domain: [-100, 100]^D
    
    if nargin < 2
        shift_data = zeros(size(x));
    end
    
    % Shift the variables
    z = x - shift_data;
    
    % Sphere function
    f = sum(z.^2) - 450;
end

%% Function 6: Shifted Rosenbrock Function (F6)
function f = cec2005_f6(x, shift_data)
    % F6: Shifted Rosenbrock Function  
    % Global minimum: f(x*) = 390, where x* = shift_data + 1
    % Search domain: [-100, 100]^D
    
    if nargin < 2
        shift_data = zeros(size(x));
    end
    
    % Shift the variables
    z = x - shift_data;
    
    % Rosenbrock function
    D = length(z);
    f = 0;
    for i = 1:(D-1)
        f = f + 100 * (z(i)^2 - z(i+1))^2 + (z(i) - 1)^2;
    end
    f = f + 390;
end

%% Generate Shift Data
function shift_data = generate_shift_data(func_num, dimension)
    % Generate pseudo-random shift data for CEC'2005 functions
    % This simulates the official shift data (normally provided as separate files)
    
    rng(func_num * 1000 + dimension); % Reproducible random seed
    
    if func_num == 1
        % F1: Shifted Sphere - shift within [-80, 80]
        shift_data = -80 + 160 * rand(1, dimension);
    elseif func_num == 6
        % F6: Shifted Rosenbrock - shift within [-80, 80] 
        shift_data = -80 + 160 * rand(1, dimension);
    else
        shift_data = zeros(1, dimension);
    end
end

%% Wrapper Functions for Different Dimensions
function f = cec2005_f1_wrapper(x, D)
    shift_data = generate_shift_data(1, D);
    f = cec2005_f1(x, shift_data);
end

function f = cec2005_f6_wrapper(x, D)
    shift_data = generate_shift_data(6, D);
    f = cec2005_f6(x, shift_data);
end

%% Test Functions
function test_cec2005_functions()
    fprintf('Testing CEC''2005 Benchmark Functions\n');
    fprintf('====================================\n');
    
    % Test dimensions
    dimensions = [2, 10];
    
    for D = dimensions
        fprintf('\nDimension D = %d:\n', D);
        fprintf('--------------\n');
        
        % Test F1 - Shifted Sphere
        shift1 = generate_shift_data(1, D);
        optimum1 = shift1; % Global optimum location
        f1_opt = cec2005_f1(optimum1, shift1);
        
        % Test point
        test_point = zeros(1, D);
        f1_test = cec2005_f1(test_point, shift1);
        
        fprintf('F1 (Shifted Sphere):\n');
        fprintf('  Optimum value: %.6f (should be -450)\n', f1_opt);
        fprintf('  Test point [0,...,0] value: %.6f\n', f1_test);
        
        % Test F6 - Shifted Rosenbrock  
        shift6 = generate_shift_data(6, D);
        optimum6 = shift6 + 1; % Global optimum location (shift + 1)
        f6_opt = cec2005_f6(optimum6, shift6);
        
        % Test point
        f6_test = cec2005_f6(test_point, shift6);
        
        fprintf('F6 (Shifted Rosenbrock):\n');
        fprintf('  Optimum value: %.6f (should be 390)\n', f6_opt);
        fprintf('  Test point [0,...,0] value: %.6f\n', f6_test);
    end
end

%% Visualization for 2D Functions
function visualize_2d_functions()
    fprintf('\nGenerating 2D visualizations...\n');
    
    % Create meshgrid for 2D visualization
    [X, Y] = meshgrid(-10:0.5:10, -10:0.5:10);
    
    % Get shift data
    shift1_2d = generate_shift_data(1, 2);
    shift6_2d = generate_shift_data(6, 2);
    
    % Evaluate functions
    Z1 = zeros(size(X));
    Z6 = zeros(size(X));
    
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            point = [X(i,j), Y(i,j)];
            Z1(i,j) = cec2005_f1(point, shift1_2d);
            Z6(i,j) = cec2005_f6(point, shift6_2d);
        end
    end
    
    % Plot F1
    figure('Position', [100, 100, 1200, 500]);
    
    subplot(1,2,1);
    contour(X, Y, Z1, 50);
    colorbar;
    title('F1: Shifted Sphere Function (2D)');
    xlabel('x_1');
    ylabel('x_2');
    hold on;
    plot(shift1_2d(1), shift1_2d(2), 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    legend('Contours', 'Global Optimum', 'Location', 'best');
    grid on;
    
    subplot(1,2,2);
    contour(X, Y, Z6, 50);
    colorbar;
    title('F6: Shifted Rosenbrock Function (2D)');
    xlabel('x_1');
    ylabel('x_2');
    hold on;
    plot(shift6_2d(1)+1, shift6_2d(2)+1, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
    legend('Contours', 'Global Optimum', 'Location', 'best');
    grid on;
    
    sgtitle('CEC''2005 Benchmark Functions Visualization', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, 'cec2005_functions_2d.png');
    
    % 3D surface plots
    figure('Position', [150, 150, 1200, 500]);
    
    subplot(1,2,1);
    surf(X, Y, Z1);
    title('F1: Shifted Sphere Function (3D)');
    xlabel('x_1');
    ylabel('x_2');
    zlabel('f(x_1, x_2)');
    colorbar;
    
    subplot(1,2,2);
    surf(X, Y, Z6);
    title('F6: Shifted Rosenbrock Function (3D)');
    xlabel('x_1');
    ylabel('x_2');
    zlabel('f(x_1, x_2)');
    colorbar;
    
    sgtitle('CEC''2005 Benchmark Functions 3D Surface', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, 'cec2005_functions_3d.png');
    
    fprintf('Visualizations saved as PNG files.\n');
end

%% Main Execution
if ~exist('OCTAVE_VERSION', 'builtin')
    % Only run if called directly (not when functions are loaded)
    if strcmp(get(0, 'DefaultFigureVisible'), 'on')
        test_cec2005_functions();
        visualize_2d_functions();
    end
end
