%% Smart Home FLC Analysis and Visualization
% This script provides comprehensive analysis of the FLC including:
% - Membership function plots
% - Control surface visualizations
% - Rule activation analysis
% - Performance evaluation
%
% Author: Priyanka Rai
% Course: Evolutionary and Fuzzy Systems

clear all; close all; clc;

%% Load the FIS
try
    fis = readfis('smart_home_flc.fis');
    fprintf('FIS loaded successfully!\n');
catch
    fprintf('FIS file not found. Running smart_home_flc.m first...\n');
    run('smart_home_flc.m');
    fis = readfis('smart_home_flc.fis');
end

%% 1. Membership Function Visualization
figure('Position', [100, 100, 1200, 800]);

% Input membership functions
subplot(3,3,1);
plotmf(fis, 'input', 1);
title('Temperature Membership Functions');
xlabel('Temperature (°C)');
ylabel('Membership Degree');
grid on;

subplot(3,3,2);
plotmf(fis, 'input', 2);
title('Light Level Membership Functions');
xlabel('Light Level (lux)');
ylabel('Membership Degree');
grid on;

subplot(3,3,3);
plotmf(fis, 'input', 3);
title('Time of Day Membership Functions');
xlabel('Time (hours)');
ylabel('Membership Degree');
grid on;

subplot(3,3,4);
plotmf(fis, 'input', 4);
title('Activity Level Membership Functions');
xlabel('Activity Level (%)');
ylabel('Membership Degree');
grid on;

subplot(3,3,5);
plotmf(fis, 'input', 5);
title('User Preference Membership Functions');
xlabel('User Preference (1-5)');
ylabel('Membership Degree');
grid on;

% Output membership functions
subplot(3,3,6);
plotmf(fis, 'output', 1);
title('HVAC Control Membership Functions');
xlabel('HVAC Control (%)');
ylabel('Membership Degree');
grid on;

subplot(3,3,7);
plotmf(fis, 'output', 2);
title('Lighting Control Membership Functions');
xlabel('Lighting Control (%)');
ylabel('Membership Degree');
grid on;

subplot(3,3,8);
plotmf(fis, 'output', 3);
title('Blind Position Membership Functions');
xlabel('Blind Position (%)');
ylabel('Membership Degree');
grid on;

sgtitle('Smart Home FLC - Membership Functions', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, 'membership_functions.png');

%% 2. Control Surface Analysis
% Temperature vs User Preference (fixing other inputs)
figure('Position', [150, 150, 1200, 400]);

% Fixed values for other inputs
light_fixed = 300;    % Dim light
time_fixed = 14;      % Afternoon
activity_fixed = 50;  % Moderate activity

subplot(1,3,1);
gensurf(fis, [1 5], [1], 4); % Temperature vs User Preference -> HVAC Control
title('HVAC Control Surface');
xlabel('Temperature (°C)');
ylabel('User Preference');
zlabel('HVAC Control (%)');
colorbar;

subplot(1,3,2);
gensurf(fis, [1 5], [2], 4); % Temperature vs User Preference -> Lighting Control
title('Lighting Control Surface');
xlabel('Temperature (°C)');
ylabel('User Preference');
zlabel('Lighting Control (%)');
colorbar;

subplot(1,3,3);
gensurf(fis, [1 5], [3], 4); % Temperature vs User Preference -> Blind Position
title('Blind Position Control Surface');
xlabel('Temperature (°C)');
ylabel('User Preference');
zlabel('Blind Position (%)');
colorbar;

sgtitle('Control Surfaces: Temperature vs User Preference', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'control_surfaces_temp_pref.png');

% Light Level vs Time of Day
figure('Position', [200, 200, 1200, 400]);

subplot(1,3,1);
gensurf(fis, [2 3], [1], 3); % Light Level vs Time of Day -> HVAC Control
title('HVAC Control Surface');
xlabel('Light Level (lux)');
ylabel('Time of Day (hours)');
zlabel('HVAC Control (%)');
colorbar;

subplot(1,3,2);
gensurf(fis, [2 3], [2], 3); % Light Level vs Time of Day -> Lighting Control
title('Lighting Control Surface');
xlabel('Light Level (lux)');
ylabel('Time of Day (hours)');
zlabel('Lighting Control (%)');
colorbar;

subplot(1,3,3);
gensurf(fis, [2 3], [3], 3); % Light Level vs Time of Day -> Blind Position
title('Blind Position Control Surface');
xlabel('Light Level (lux)');
ylabel('Time of Day (hours)');
zlabel('Blind Position (%)');
colorbar;

sgtitle('Control Surfaces: Light Level vs Time of Day', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'control_surfaces_light_time.png');

%% 3. Rule Activation Analysis
fprintf('\n=== Rule Activation Analysis ===\n');

% Test scenarios with detailed rule firing analysis
scenarios = [
    17, 100, 7, 15, 3;   % Cold morning
    30, 800, 14, 60, 2;  % Hot afternoon  
    22, 200, 20, 10, 3;  % Comfortable evening
    19, 50, 22, 5, 4;    % Cool night, resting
    28, 600, 12, 80, 1   % Warm day, active
];

scenario_names = {
    'Cold Morning (Low Activity)'
    'Hot Afternoon (Moderate Activity)'
    'Comfortable Evening (Resting)'
    'Cool Night (Resting)'
    'Warm Day (High Activity)'
};

figure('Position', [250, 250, 1000, 600]);

for i = 1:size(scenarios, 1)
    input_vals = scenarios(i, :);
    
    % Evaluate FIS
    output = evalfis(fis, input_vals);
    
    fprintf('\nScenario %d: %s\n', i, scenario_names{i});
    fprintf('Inputs: T=%.1f°C, L=%.0flux, Time=%.0fh, Act=%.0f%%, Pref=%.1f\n', ...
            input_vals);
    fprintf('Outputs: HVAC=%.1f%%, Light=%.1f%%, Blinds=%.1f%%\n', output);
    
    % Plot output values
    subplot(2, 3, i);
    bar(output);
    title(scenario_names{i});
    xlabel('Output Index');
    ylabel('Output Value');
    grid on;
    ylim([-100, 100]);
end

sgtitle('Output Analysis for Different Scenarios', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'output_analysis.png');

%% 4. System Response Analysis
fprintf('\n=== System Response Analysis ===\n');

% Temperature sweep analysis
temp_range = 15:0.5:35;
light_val = 300; time_val = 14; activity_val = 50; pref_val = 3;

hvac_response = zeros(size(temp_range));
light_response = zeros(size(temp_range));
blind_response = zeros(size(temp_range));

for i = 1:length(temp_range)
    output = evalfis(fis, [temp_range(i), light_val, time_val, activity_val, pref_val]);
    hvac_response(i) = output(1);
    light_response(i) = output(2);
    blind_response(i) = output(3);
end

figure('Position', [300, 300, 1200, 400]);

subplot(1,3,1);
plot(temp_range, hvac_response, 'b-', 'LineWidth', 2);
xlabel('Temperature (°C)');
ylabel('HVAC Control (%)');
title('HVAC Response to Temperature');
grid on;

subplot(1,3,2);
plot(temp_range, light_response, 'g-', 'LineWidth', 2);
xlabel('Temperature (°C)');
ylabel('Lighting Control (%)');
title('Lighting Response to Temperature');
grid on;

subplot(1,3,3);
plot(temp_range, blind_response, 'r-', 'LineWidth', 2);
xlabel('Temperature (°C)');
ylabel('Blind Position (%)');
title('Blind Response to Temperature');
grid on;

sgtitle('System Response to Temperature Variation', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'temperature_response_analysis.png');

%% 5. User Preference Impact Analysis
pref_range = 1:0.1:5;
temp_val = 22; % Comfortable temperature

hvac_pref_response = zeros(size(pref_range));
light_pref_response = zeros(size(pref_range));
blind_pref_response = zeros(size(pref_range));

for i = 1:length(pref_range)
    output = evalfis(fis, [temp_val, light_val, time_val, activity_val, pref_range(i)]);
    hvac_pref_response(i) = output(1);
    light_pref_response(i) = output(2);
    blind_pref_response(i) = output(3);
end

figure('Position', [350, 350, 1200, 400]);

subplot(1,3,1);
plot(pref_range, hvac_pref_response, 'b-', 'LineWidth', 2);
xlabel('User Preference (1=Cool, 5=Warm)');
ylabel('HVAC Control (%)');
title('HVAC Response to User Preference');
grid on;

subplot(1,3,2);
plot(pref_range, light_pref_response, 'g-', 'LineWidth', 2);
xlabel('User Preference (1=Cool, 5=Warm)');
ylabel('Lighting Control (%)');
title('Lighting Response to User Preference');
grid on;

subplot(1,3,3);
plot(pref_range, blind_pref_response, 'r-', 'LineWidth', 2);
xlabel('User Preference (1=Cool, 5=Warm)');
ylabel('Blind Position (%)');
title('Blind Response to User Preference');
grid on;

sgtitle('System Response to User Preference Variation', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'user_preference_analysis.png');

%% 6. Performance Metrics
fprintf('\n=== Performance Evaluation ===\n');

% Generate test dataset
n_tests = 100;
test_inputs = [
    15 + 20*rand(n_tests, 1), ... % Temperature: 15-35°C
    1000*rand(n_tests, 1), ...    % Light: 0-1000 lux
    24*rand(n_tests, 1), ...      % Time: 0-24 hours
    100*rand(n_tests, 1), ...     % Activity: 0-100%
    1 + 4*rand(n_tests, 1)        % Preference: 1-5
];

% Evaluate all test cases
test_outputs = zeros(n_tests, 3);
evaluation_time = zeros(n_tests, 1);

for i = 1:n_tests
    tic;
    test_outputs(i, :) = evalfis(fis, test_inputs(i, :));
    evaluation_time(i) = toc;
end

fprintf('Performance Statistics:\n');
fprintf('Average evaluation time: %.4f seconds\n', mean(evaluation_time));
fprintf('Max evaluation time: %.4f seconds\n', max(evaluation_time));
fprintf('Min evaluation time: %.4f seconds\n', min(evaluation_time));

% Output range analysis
fprintf('\nOutput Range Analysis:\n');
fprintf('HVAC Control: [%.1f, %.1f]%%\n', min(test_outputs(:,1)), max(test_outputs(:,1)));
fprintf('Lighting Control: [%.1f, %.1f]%%\n', min(test_outputs(:,2)), max(test_outputs(:,2)));
fprintf('Blind Position: [%.1f, %.1f]%%\n', min(test_outputs(:,3)), max(test_outputs(:,3)));

% Sensitivity analysis
figure('Position', [400, 400, 800, 600]);
histogram2(test_inputs(:,1), test_outputs(:,1), 20, 'DisplayStyle', 'tile');
xlabel('Temperature Input (°C)');
ylabel('HVAC Output (%)');
title('Input-Output Relationship: Temperature vs HVAC Control');
colorbar;
saveas(gcf, 'sensitivity_analysis.png');

fprintf('\nAnalysis completed! All plots saved as PNG files.\n');
fprintf('Files generated:\n');
fprintf('- membership_functions.png\n');
fprintf('- control_surfaces_temp_pref.png\n');
fprintf('- control_surfaces_light_time.png\n');
fprintf('- output_analysis.png\n');
fprintf('- temperature_response_analysis.png\n');
fprintf('- user_preference_analysis.png\n');
fprintf('- sensitivity_analysis.png\n');