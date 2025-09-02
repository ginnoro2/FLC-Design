%% Smart Home Fuzzy Logic Controller & Analysis
% Combined script: Design, Test, and Visualize FLC
%
% Author: Priyanka Rai
% Course: Evolutionary and Fuzzy Systems
% Date: 2024

%% === PART 1: DESIGN THE FLC ===
clear all; close all; clc;  % Only clear at the very beginning

%% Create Fuzzy Inference System
fis = mamfis('Name', 'SmartHomeFLC');

%% Define Input Variables

% Input 1: Room Temperature (°C)
fis = addInput(fis, [15 35], 'Name', 'Temperature');
fis = addMF(fis, 'Temperature', 'trapmf', [15 15 18 20], 'Name', 'Cold');
fis = addMF(fis, 'Temperature', 'trimf', [18 22 26], 'Name', 'Comfortable');
fis = addMF(fis, 'Temperature', 'trapmf', [24 27 35 35], 'Name', 'Hot');

% Input 2: Light Level (lux)
fis = addInput(fis, [0 1000], 'Name', 'LightLevel');
fis = addMF(fis, 'LightLevel', 'trapmf', [0 0 50 150], 'Name', 'Dark');
fis = addMF(fis, 'LightLevel', 'trimf', [100 300 500], 'Name', 'Dim');
fis = addMF(fis, 'LightLevel', 'trapmf', [400 600 1000 1000], 'Name', 'Bright');

% Input 3: Time of Day (hours)
fis = addInput(fis, [0 24], 'Name', 'TimeOfDay');
fis = addMF(fis, 'TimeOfDay', 'trapmf', [0 0 6 8], 'Name', 'Night');
fis = addMF(fis, 'TimeOfDay', 'trimf', [7 12 17], 'Name', 'Day');
fis = addMF(fis, 'TimeOfDay', 'trapmf', [16 20 24 24], 'Name', 'Evening');

% Input 4: User Activity Level (%)
fis = addInput(fis, [0 100], 'Name', 'ActivityLevel');
fis = addMF(fis, 'ActivityLevel', 'trapmf', [0 0 10 30], 'Name', 'Resting');
fis = addMF(fis, 'ActivityLevel', 'trimf', [20 50 80], 'Name', 'Moderate');
fis = addMF(fis, 'ActivityLevel', 'trapmf', [70 90 100 100], 'Name', 'Active');

% Input 5: User Preference (1-5 scale)
fis = addInput(fis, [1 5], 'Name', 'UserPreference');
fis = addMF(fis, 'UserPreference', 'trimf', [1 1 2.5], 'Name', 'Cool');
fis = addMF(fis, 'UserPreference', 'trimf', [2 3 4], 'Name', 'Neutral');
fis = addMF(fis, 'UserPreference', 'trimf', [3.5 5 5], 'Name', 'Warm');

%% Define Output Variables

% Output 1: HVAC Control (-100 to 100%)
fis = addOutput(fis, [-100 100], 'Name', 'HVACControl');
fis = addMF(fis, 'HVACControl', 'trapmf', [-100 -100 -60 -20], 'Name', 'StrongCooling');
fis = addMF(fis, 'HVACControl', 'trimf', [-40 -10 10], 'Name', 'LightCooling');
fis = addMF(fis, 'HVACControl', 'trimf', [-15 0 15], 'Name', 'Off');
fis = addMF(fis, 'HVACControl', 'trimf', [-10 10 40], 'Name', 'LightHeating');
fis = addMF(fis, 'HVACControl', 'trapmf', [20 60 100 100], 'Name', 'StrongHeating');

% Output 2: Lighting Control (0-100%)
fis = addOutput(fis, [0 100], 'Name', 'LightingControl');
fis = addMF(fis, 'LightingControl', 'trapmf', [0 0 5 15], 'Name', 'Off');
fis = addMF(fis, 'LightingControl', 'trimf', [10 30 50], 'Name', 'Dim');
fis = addMF(fis, 'LightingControl', 'trimf', [40 70 90], 'Name', 'Medium');
fis = addMF(fis, 'LightingControl', 'trapmf', [80 95 100 100], 'Name', 'Bright');

% Output 3: Blind Position (0-100%)
fis = addOutput(fis, [0 100], 'Name', 'BlindPosition');
fis = addMF(fis, 'BlindPosition', 'trapmf', [0 0 10 25], 'Name', 'Closed');
fis = addMF(fis, 'BlindPosition', 'trimf', [15 40 65], 'Name', 'PartiallyOpen');
fis = addMF(fis, 'BlindPosition', 'trapmf', [60 85 100 100], 'Name', 'FullyOpen');

%% Define Fuzzy Rules
rules = [
    "Temperature==Cold & UserPreference==Cool => HVACControl=LightHeating, LightingControl=Medium, BlindPosition=PartiallyOpen (1)"
    "Temperature==Cold & UserPreference==Neutral => HVACControl=StrongHeating, LightingControl=Medium, BlindPosition=PartiallyOpen (1)"
    "Temperature==Cold & UserPreference==Warm => HVACControl=StrongHeating, LightingControl=Medium, BlindPosition=Closed (1)"
    
    "Temperature==Hot & UserPreference==Cool => HVACControl=StrongCooling, LightingControl=Dim, BlindPosition=Closed (1)"
    "Temperature==Hot & UserPreference==Neutral => HVACControl=LightCooling, LightingControl=Dim, BlindPosition=PartiallyOpen (1)"
    "Temperature==Hot & UserPreference==Warm => HVACControl=Off, LightingControl=Medium, BlindPosition=FullyOpen (1)"
    
    "Temperature==Comfortable & UserPreference==Cool => HVACControl=LightCooling, LightingControl=Medium, BlindPosition=FullyOpen (1)"
    "Temperature==Comfortable & UserPreference==Neutral => HVACControl=Off, LightingControl=Medium, BlindPosition=PartiallyOpen (1)"
    "Temperature==Comfortable & UserPreference==Warm => HVACControl=LightHeating, LightingControl=Medium, BlindPosition=PartiallyOpen (1)"
    
    "LightLevel==Dark & TimeOfDay==Day => LightingControl=Bright, BlindPosition=FullyOpen (1)"
    "LightLevel==Dark & TimeOfDay==Evening => LightingControl=Medium, BlindPosition=Closed (1)"
    "LightLevel==Dark & TimeOfDay==Night => LightingControl=Dim, BlindPosition=Closed (1)"
    
    "LightLevel==Bright & TimeOfDay==Day => LightingControl=Off, BlindPosition=PartiallyOpen (1)"
    "LightLevel==Bright & TimeOfDay==Evening => LightingControl=Dim, BlindPosition=PartiallyOpen (1)"
    "LightLevel==Bright & TimeOfDay==Night => LightingControl=Off, BlindPosition=Closed (1)"
    
    "ActivityLevel==Resting & TimeOfDay==Night => HVACControl=Off, LightingControl=Off, BlindPosition=Closed (1)"
    "ActivityLevel==Active & TimeOfDay==Day => HVACControl=LightCooling, LightingControl=Bright, BlindPosition=FullyOpen (1)"
    "ActivityLevel==Moderate => LightingControl=Medium (0.7)"
    
    "Temperature==Hot & LightLevel==Bright => HVACControl=StrongCooling, BlindPosition=Closed (1)"
    "Temperature==Cold & LightLevel==Dark => HVACControl=StrongHeating, LightingControl=Bright (1)"
    
    "UserPreference==Cool & ActivityLevel==Active => HVACControl=LightCooling (0.8)"
    "UserPreference==Warm & ActivityLevel==Resting => HVACControl=LightHeating (0.8)"
];

fis = addRule(fis, rules);

%% Display FIS Info
fprintf('Smart Home FLC System Created Successfully!\n');
fprintf('Number of inputs: %d\n', length(fis.Inputs));
fprintf('Number of outputs: %d\n', length(fis.Outputs));
fprintf('Number of rules: %d\n', length(fis.Rules));

%% Save FIS
writeFIS(fis, 'smart_home_flc.fis');
fprintf('FIS saved as smart_home_flc.fis\n');

%% Test Scenarios
fprintf('\n=== Testing Controller with Sample Scenarios ===\n');

% Scenario 1: Cold morning
[temp1, light1, time1, activity1, preference1] = deal(17, 100, 7, 15, 3);
output1 = evalfis(fis, [temp1, light1, time1, activity1, preference1]);
fprintf('Scenario 1 - Cold Morning:\n');
fprintf('  Inputs: Temp=%.1f°C, Light=%.0f lux, Time=%.0f h, Activity=%.0f%%, Pref=%.1f\n', ...
        temp1, light1, time1, activity1, preference1);
fprintf('  Outputs: HVAC=%.1f%%, Lighting=%.1f%%, Blinds=%.1f%%\n\n', output1(1), output1(2), output1(3));

% Scenario 2: Hot afternoon
[temp2, light2, time2, activity2, preference2] = deal(30, 800, 14, 60, 2);
output2 = evalfis(fis, [temp2, light2, time2, activity2, preference2]);
fprintf('Scenario 2 - Hot Afternoon:\n');
fprintf('  Inputs: Temp=%.1f°C, Light=%.0f lux, Time=%.0f h, Activity=%.0f%%, Pref=%.1f\n', ...
        temp2, light2, time2, activity2, preference2);
fprintf('  Outputs: HVAC=%.1f%%, Lighting=%.1f%%, Blinds=%.1f%%\n\n', output2(1), output2(2), output2(3));

% Scenario 3: Comfortable evening
[temp3, light3, time3, activity3, preference3] = deal(22, 200, 20, 10, 3);
output3 = evalfis(fis, [temp3, light3, time3, activity3, preference3]);
fprintf('Scenario 3 - Comfortable Evening:\n');
fprintf('  Inputs: Temp=%.1f°C, Light=%.0f lux, Time=%.0f h, Activity=%.0f%%, Pref=%.1f\n', ...
        temp3, light3, time3, activity3, preference3);
fprintf('  Outputs: HVAC=%.1f%%, Lighting=%.1f%%, Blinds=%.1f%%\n\n', output3(1), output3(2), output3(3));

%% === PART 2: ANALYSIS & VISUALIZATION ===

%% 1. Membership Function Plots
figure('Name', 'Membership Functions', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
title('Fuzzy Membership Functions', 'FontSize', 14, 'FontWeight', 'bold');
current_subplot = 1;

for i = 1:length(fis.Inputs)
    subplot(3, 3, current_subplot);
    plotmf(fis, 'input', i);
    title(fis.Inputs(i).Name, 'Interpreter', 'none');
    grid on;
    current_subplot = current_subplot + 1;
end

for i = 1:length(fis.Outputs)
    subplot(3, 3, current_subplot);
    plotmf(fis, 'output', i);
    title(fis.Outputs(i).Name, 'Interpreter', 'none');
    grid on;
    current_subplot = current_subplot + 1;
end
sgtitle('Input and Output Membership Functions', 'FontSize', 12);

%% 2. Control Surface Plots
io_pairs = {
    {'Temperature', 'UserPreference', 'HVACControl'}
    {'LightLevel', 'TimeOfDay', 'LightingControl'}
    {'Temperature', 'LightLevel', 'BlindPosition'}
    {'TimeOfDay', 'ActivityLevel', 'HVACControl'}
    {'ActivityLevel', 'UserPreference', 'LightingControl'}
    {'TimeOfDay', 'LightLevel', 'BlindPosition'}
};

figure('Name', 'Control Surfaces', 'NumberTitle', 'off');
for i = 1:size(io_pairs, 1)
    subplot(2, 3, i);
    try
        in1 = find([fis.Inputs.Name] == io_pairs{i}{1});
        in2 = find([fis.Inputs.Name] == io_pairs{i}{2});
        out = find([fis.Outputs.Name] == io_pairs{i}{3});
        gensurf(fis, [in1, in2], out);
        title(sprintf('%s vs %s → %s', io_pairs{i}{1}, io_pairs{i}{2}, io_pairs{i}{3}), 'FontSize', 8);
    catch
        title('Plot Error', 'Color', 'r');
    end
end
sgtitle('Control Surface Plots', 'FontSize', 12);

%% 3. Rule Activation for Scenario 1
figure;
output = evalfis(fis, [17, 100, 7, 15, 3]);
bar(output);
title('Output Values (Cold Morning)');
xlabel('Output Index'); ylabel('Output Value');
grid on; xticks(1:length(output));

%% 4. Daily Simulation
hours = 0:0.5:24;
n = length(hours);

% Simulate daily patterns
temp_data = 20 + 8*sin(pi*(hours-6)/12) + 2*randn(size(hours));
light_data = max(0, 800*(1 - abs(sin(pi*hours/12)))) + 50*randn(size(hours));
activity_data = zeros(size(hours));
activity_data(hours>=7 & hours<=9) = 70;
activity_data(hours>=12 & hours<=14) = 60;
activity_data(hours>=18 & hours<=22) = 80;
activity_data((hours>=23) | (hours<=6)) = 10;
preference_data = 3 * ones(size(hours));

% Evaluate over time
hvac_out = zeros(size(hours));
light_out = zeros(size(hours));
blind_out = zeros(size(hours));

for i = 1:n
    inputs = [temp_data(i), light_data(i), hours(i), activity_data(i), preference_data(i)];
    y = evalfis(fis, inputs);
    hvac_out(i) = y(1);
    light_out(i) = y(2);
    blind_out(i) = y(3);
end

% Plot time-series
figure('Position', [100, 100, 1000, 800]);
subplot(4,1,1);
plot(hours, temp_data, 'r'); hold on; yyaxis right; plot(hours, hvac_out, 'b--');
title('Temperature & HVAC'); ylabel('Temp (°C)'); yyaxis right; ylabel('HVAC (%)');
legend('Temp', 'HVAC', 'Location', 'best'); grid on; xlim([0 24]);

subplot(4,1,2);
plot(hours, light_data, 'm'); hold on; yyaxis right; plot(hours, light_out, 'g--');
title('Light Level & Lighting'); ylabel('Light (lux)'); yyaxis right; ylabel('Lighting (%)');
legend('Ambient', 'Lighting', 'Location', 'best'); grid on; xlim([0 24]);

subplot(4,1,3);
plot(hours, activity_data, 'c.'); hold on; yyaxis right; plot(hours, blind_out, 'k-');
title('Activity & Blinds'); ylabel('Activity (%)'); yyaxis right; ylabel('Blinds (%)');
legend('Activity', 'Blinds', 'Location', 'best'); grid on; xlim([0 24]);

subplot(4,1,4);
area(hours, blind_out, 'FaceAlpha', 0.5); title('Blind Position Over 24h');
xlabel('Time (h)'); ylabel('Blind Position (%)'); grid on; xlim([0 24]);
sgtitle('Daily Smart Home Simulation', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Analysis completed! All plots generated.\n');