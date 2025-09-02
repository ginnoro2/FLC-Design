%% Sugeno vs Mamdani Model Comparison for GA Optimization
% This script demonstrates how the GA optimization would change 
% when using a Sugeno (TSK) model instead of a Mamdani model
%
% Author: Priyanka Rai
% Course: Evolutionary and Fuzzy Systems

clear all; close all; clc;

%% Create Sugeno FIS Model
fprintf('Creating Sugeno (TSK) FIS Model for Smart Home Control...\n');

% Create Sugeno FIS
sugeno_fis = sugfis('Name', 'SmartHomeSugenoFLC');

%% Define Input Variables (Same as Mamdani)
% Input 1: Room Temperature (°C)
sugeno_fis = addInput(sugeno_fis, [15 35], 'Name', 'Temperature');
sugeno_fis = addMF(sugeno_fis, 'Temperature', 'trapmf', [15 15 18 20], 'Name', 'Cold');
sugeno_fis = addMF(sugeno_fis, 'Temperature', 'trimf', [18 22 26], 'Name', 'Comfortable');
sugeno_fis = addMF(sugeno_fis, 'Temperature', 'trapmf', [24 27 35 35], 'Name', 'Hot');

% Input 2: Light Level (lux)
sugeno_fis = addInput(sugeno_fis, [0 1000], 'Name', 'LightLevel');
sugeno_fis = addMF(sugeno_fis, 'LightLevel', 'trapmf', [0 0 50 150], 'Name', 'Dark');
sugeno_fis = addMF(sugeno_fis, 'LightLevel', 'trimf', [100 300 500], 'Name', 'Dim');
sugeno_fis = addMF(sugeno_fis, 'LightLevel', 'trapmf', [400 600 1000 1000], 'Name', 'Bright');

% Input 3: Time of Day (hours)
sugeno_fis = addInput(sugeno_fis, [0 24], 'Name', 'TimeOfDay');
sugeno_fis = addMF(sugeno_fis, 'TimeOfDay', 'trapmf', [0 0 6 8], 'Name', 'Night');
sugeno_fis = addMF(sugeno_fis, 'TimeOfDay', 'trimf', [7 12 17], 'Name', 'Day');
sugeno_fis = addMF(sugeno_fis, 'TimeOfDay', 'trapmf', [16 20 24 24], 'Name', 'Evening');

% Input 4: User Activity Level (%)
sugeno_fis = addInput(sugeno_fis, [0 100], 'Name', 'ActivityLevel');
sugeno_fis = addMF(sugeno_fis, 'ActivityLevel', 'trapmf', [0 0 10 30], 'Name', 'Resting');
sugeno_fis = addMF(sugeno_fis, 'ActivityLevel', 'trimf', [20 50 80], 'Name', 'Moderate');
sugeno_fis = addMF(sugeno_fis, 'ActivityLevel', 'trapmf', [70 90 100 100], 'Name', 'Active');

% Input 5: User Preference (1-5 scale)
sugeno_fis = addInput(sugeno_fis, [1 5], 'Name', 'UserPreference');
sugeno_fis = addMF(sugeno_fis, 'UserPreference', 'trimf', [1 1 2.5], 'Name', 'Cool');
sugeno_fis = addMF(sugeno_fis, 'UserPreference', 'trimf', [2 3 4], 'Name', 'Neutral');
sugeno_fis = addMF(sugeno_fis, 'UserPreference', 'trimf', [3.5 5 5], 'Name', 'Warm');

%% Define Output Variables (Sugeno - Linear/Constant functions)
% For Sugeno model, outputs are mathematical functions of inputs

% Output 1: HVAC Control
sugeno_fis = addOutput(sugeno_fis, [-100 100], 'Name', 'HVACControl');

% Output 2: Lighting Control
sugeno_fis = addOutput(sugeno_fis, [0 100], 'Name', 'LightingControl');

% Output 3: Blind Position
sugeno_fis = addOutput(sugeno_fis, [0 100], 'Name', 'BlindPosition');

%% Define Sugeno Rules with Linear Output Functions
% In Sugeno model, consequents are mathematical functions

% For demonstration, we'll use linear functions of inputs
% HVAC = a1*Temp + a2*Light + a3*Time + a4*Activity + a5*Preference + a0
% Lighting = b1*Temp + b2*Light + b3*Time + b4*Activity + b5*Preference + b0
% Blinds = c1*Temp + c2*Light + c3*Time + c4*Activity + c5*Preference + c0

sugeno_rules = [
    % Rule format: [input1_mf input2_mf input3_mf input4_mf input5_mf output1_params output2_params output3_params weight connection]
    % Temperature-based rules
    1 0 0 0 1 [-3 0 0 0 5 60] [0 -0.05 0 0 0 30] [0 0 0 0 0 20] 1 1;  % Cold & Cool preference
    1 0 0 0 2 [-3 0 0 0 2 80] [0 -0.05 0 0 0 40] [0 0 0 0 0 30] 1 1;  % Cold & Neutral preference  
    1 0 0 0 3 [-3 0 0 0 0 90] [0 -0.05 0 0 0 50] [0 0 0 0 0 15] 1 1;  % Cold & Warm preference
    
    3 0 0 0 1 [2 0 0 -0.5 5 -80] [0 -0.1 0 0 0 20] [0 0 0 0 0 10] 1 1; % Hot & Cool preference
    3 0 0 0 2 [2 0 0 -0.3 2 -40] [0 -0.1 0 0 0 30] [0 0 0 0 0 40] 1 1; % Hot & Neutral preference
    3 0 0 0 3 [2 0 0 0 0 -10] [0 -0.1 0 0 0 50] [0 0 0 0 0 70] 1 1;    % Hot & Warm preference
    
    2 0 0 0 1 [0 0 0 -0.2 2 -20] [0 -0.05 0 0 0 60] [0 0 0 0 0 80] 1 1; % Comfortable & Cool
    2 0 0 0 2 [0 0 0 0 0 0] [0 -0.05 0 0 0 50] [0 0 0 0 0 60] 1 1;      % Comfortable & Neutral
    2 0 0 0 3 [0 0 0 0 -2 20] [0 -0.05 0 0 0 40] [0 0 0 0 0 50] 1 1;    % Comfortable & Warm
    
    % Light and time-based rules
    0 1 2 0 0 [0 0 0 0 0 0] [0 0 3 0 0 80] [0 0 0 0 0 90] 1 1;         % Dark & Day
    0 1 3 0 0 [0 0 0 0 0 0] [0 0 2 0 0 60] [0 0 0 0 0 30] 1 1;         % Dark & Evening  
    0 1 1 0 0 [0 0 0 0 0 0] [0 0 1 0 0 30] [0 0 0 0 0 10] 1 1;         % Dark & Night
    
    0 3 2 0 0 [0 0 0 0 0 0] [0 0 0 0 0 10] [0 0 0 0 0 50] 1 1;         % Bright & Day
    0 3 3 0 0 [0 0 0 0 0 0] [0 0 0 0 0 20] [0 0 0 0 0 40] 1 1;         % Bright & Evening
    0 3 1 0 0 [0 0 0 0 0 0] [0 0 0 0 0 5] [0 0 0 0 0 10] 1 1;          % Bright & Night
    
    % Activity-based rules
    0 0 1 1 0 [0 0 0 0 0 0] [0 0 0 0 0 5] [0 0 0 0 0 10] 1 1;          % Night & Resting
    0 0 2 3 0 [-1 0 0 0 0 -20] [0 0 0 2 0 70] [0 0 0 0 0 80] 1 1;      % Day & Active
    0 0 0 2 0 [0 0 0 0 0 0] [0 0 0 1 0 50] [0 0 0 0 0 60] 1 1;         % Moderate activity
];

% Add rules to Sugeno FIS
for i = 1:size(sugeno_rules, 1)
    rule_antecedent = sugeno_rules(i, 1:5);
    hvac_params = sugeno_rules(i, 6:11);
    lighting_params = sugeno_rules(i, 12:17);
    blind_params = sugeno_rules(i, 18:23);
    weight = sugeno_rules(i, 24);
    
    % Add output membership functions (linear)
    hvac_mf_name = sprintf('hvac_rule_%d', i);
    lighting_mf_name = sprintf('lighting_rule_%d', i);
    blind_mf_name = sprintf('blind_rule_%d', i);
    
    sugeno_fis = addMF(sugeno_fis, 'HVACControl', 'linear', hvac_params, 'Name', hvac_mf_name);
    sugeno_fis = addMF(sugeno_fis, 'LightingControl', 'linear', lighting_params, 'Name', lighting_mf_name);
    sugeno_fis = addMF(sugeno_fis, 'BlindPosition', 'linear', blind_params, 'Name', blind_mf_name);
    
    % Create rule string
    rule_str = sprintf('%d %d %d %d %d => HVACControl=%s, LightingControl=%s, BlindPosition=%s (%.1f)', ...
                       rule_antecedent(1), rule_antecedent(2), rule_antecedent(3), ...
                       rule_antecedent(4), rule_antecedent(5), ...
                       hvac_mf_name, lighting_mf_name, blind_mf_name, weight);
    
    sugeno_fis = addRule(sugeno_fis, rule_str);
end

% Save Sugeno FIS
writeFIS(sugeno_fis, 'smart_home_sugeno_flc.fis');
fprintf('Sugeno FIS created and saved!\n');

%% Comparison Analysis: Mamdani vs Sugeno for GA Optimization

fprintf('\n=== Mamdani vs Sugeno GA Optimization Comparison ===\n');

%% Chromosome Structure Differences

fprintf('\n1. CHROMOSOME STRUCTURE DIFFERENCES:\n');
fprintf('-----------------------------------\n');

% Mamdani chromosome (from main GA script)
mamdani_chromosome_length = 27; % MF center parameters only
fprintf('Mamdani Model Chromosome Length: %d parameters\n', mamdani_chromosome_length);
fprintf('  - Input MF centers: 15 parameters (3 MFs × 5 inputs)\n');
fprintf('  - Output MF centers: 12 parameters (5+4+3 MFs for 3 outputs)\n');
fprintf('  - Total parameters optimized: 27\n\n');

% Sugeno chromosome
% For Sugeno, we optimize the linear function coefficients in consequents
n_rules = size(sugeno_rules, 1);
n_inputs = 5;
n_outputs = 3;
sugeno_chromosome_length = n_rules * n_outputs * (n_inputs + 1); % +1 for constant term

fprintf('Sugeno Model Chromosome Length: %d parameters\n', sugeno_chromosome_length);
fprintf('  - Number of rules: %d\n', n_rules);
fprintf('  - Coefficients per rule per output: %d (5 inputs + 1 constant)\n', n_inputs + 1);
fprintf('  - Total coefficient parameters: %d × %d × %d = %d\n', n_rules, n_outputs, n_inputs + 1, sugeno_chromosome_length);
fprintf('  - Input MF parameters: 15 (same as Mamdani)\n');
fprintf('  - Total parameters optimized: %d\n\n', sugeno_chromosome_length + 15);

%% Genetic Algorithm Modifications for Sugeno

fprintf('2. GA MODIFICATIONS FOR SUGENO MODEL:\n');
fprintf('------------------------------------\n');

fprintf('a) Chromosome Encoding:\n');
fprintf('   Mamdani: [input_MF_centers, output_MF_centers]\n');
fprintf('   Sugeno:  [input_MF_centers, linear_function_coefficients]\n\n');

fprintf('b) Parameter Bounds:\n');
fprintf('   Mamdani: MF centers bounded by input/output ranges\n');
fprintf('   Sugeno:  Coefficients bounded by expected function ranges\n');
fprintf('           (e.g., HVAC coefficients: -5 to +5)\n\n');

fprintf('c) Fitness Evaluation:\n');
fprintf('   Both models: Same fitness function based on output error\n');
fprintf('   Difference: Sugeno typically more computationally efficient\n\n');

fprintf('d) Defuzzification:\n');
fprintf('   Mamdani: Centroid/other geometric methods\n');
fprintf('   Sugeno:  Weighted average (built-in, no optimization needed)\n\n');

%% Parameter Bounds for Sugeno GA
sugeno_param_bounds = [
    % Input MF centers (same as Mamdani)
    16, 19;   % Temperature Cold
    20, 24;   % Temperature Comfortable  
    26, 32;   % Temperature Hot
    50, 100;  % Light Dark
    250, 400; % Light Dim
    600, 800; % Light Bright
    2, 5;     % Time Night
    10, 14;   % Time Day
    18, 22;   % Time Evening
    5, 20;    % Activity Resting
    40, 60;   % Activity Moderate
    80, 95;   % Activity Active
    1.2, 2.0; % Preference Cool
    2.5, 3.5; % Preference Neutral
    4.0, 4.8; % Preference Warm
];

% Add coefficient bounds for each rule's linear functions
for rule = 1:n_rules
    for output = 1:n_outputs
        % Coefficient bounds depend on expected output contribution
        if output == 1 % HVAC Control (-100 to 100)
            coeff_bounds = [-5, 5; -0.2, 0.2; -2, 2; -1, 1; -10, 10; -100, 100]; % [temp, light, time, activity, pref, constant]
        elseif output == 2 % Lighting Control (0 to 100)
            coeff_bounds = [-1, 1; -0.15, 0.15; -2, 2; -1, 1; -5, 5; 0, 100];
        else % Blind Position (0 to 100)
            coeff_bounds = [-1, 1; -0.1, 0.1; -2, 2; -0.5, 0.5; -5, 5; 0, 100];
        end
        sugeno_param_bounds = [sugeno_param_bounds; coeff_bounds];
    end
end

fprintf('3. SUGENO PARAMETER BOUNDS:\n');
fprintf('--------------------------\n');
fprintf('Total Sugeno parameters: %d\n', size(sugeno_param_bounds, 1));
fprintf('Input MF parameters: 15\n');
fprintf('Linear function coefficients: %d\n', size(sugeno_param_bounds, 1) - 15);

%% Performance Comparison Example
fprintf('\n4. PERFORMANCE COMPARISON:\n');
fprintf('-------------------------\n');

% Test both models on same inputs
test_input = [22, 300, 14, 50, 3]; % Comfortable conditions

% Load Mamdani model for comparison
try
    mamdani_fis = readfis('../Part1_FLC_Design/smart_home_flc.fis');
    mamdani_output = evalfis(mamdani_fis, test_input);
    fprintf('Mamdani Output: HVAC=%.2f, Light=%.2f, Blinds=%.2f\n', ...
            mamdani_output(1), mamdani_output(2), mamdani_output(3));
catch
    fprintf('Mamdani FIS not found for comparison\n');
end

sugeno_output = evalfis(sugeno_fis, test_input);
fprintf('Sugeno Output:  HVAC=%.2f, Light=%.2f, Blinds=%.2f\n', ...
        sugeno_output(1), sugeno_output(2), sugeno_output(3));

%% Advantages and Disadvantages Analysis

fprintf('\n5. MAMDANI vs SUGENO FOR GA OPTIMIZATION:\n');
fprintf('----------------------------------------\n');

fprintf('MAMDANI ADVANTAGES:\n');
fprintf('+ Intuitive output membership functions\n');
fprintf('+ Easier to interpret and validate rules\n');
fprintf('+ Better for linguistic modeling\n');
fprintf('+ Fewer parameters to optimize (27 vs %d+)\n', sugeno_chromosome_length);
fprintf('+ More robust to parameter variations\n\n');

fprintf('MAMDANI DISADVANTAGES:\n');
fprintf('- Slower inference (defuzzification required)\n');
fprintf('- Less precise numerical modeling\n');
fprintf('- Fixed output shapes\n\n');

fprintf('SUGENO ADVANTAGES:\n');
fprintf('+ Faster inference (weighted average)\n');
fprintf('+ More precise numerical control\n');
fprintf('+ Better for adaptive control systems\n');
fprintf('+ Linear consequents enable analytical solutions\n');
fprintf('+ More suitable for optimization algorithms\n\n');

fprintf('SUGENO DISADVANTAGES:\n');
fprintf('- Many more parameters to optimize (%d+ vs 27)\n', sugeno_chromosome_length);
fprintf('- Less intuitive consequent functions\n');
fprintf('- Higher risk of overfitting\n');
fprintf('- More complex chromosome encoding\n');
fprintf('- Larger search space\n\n');

%% Recommendations

fprintf('6. RECOMMENDATIONS:\n');
fprintf('------------------\n');
fprintf('For THIS Smart Home Application:\n\n');

fprintf('USE MAMDANI IF:\n');
fprintf('- Interpretability is critical\n');
fprintf('- Expert knowledge is available for rule definition\n');
fprintf('- System requirements allow moderate precision\n');
fprintf('- Computational resources are limited\n');
fprintf('- Robustness is more important than precision\n\n');

fprintf('USE SUGENO IF:\n');
fprintf('- High precision control is required\n');
fprintf('- Real-time performance is critical\n');
fprintf('- System will be frequently re-optimized\n');
fprintf('- Mathematical modeling is preferred\n');
fprintf('- Large training datasets are available\n\n');

fprintf('FOR GA OPTIMIZATION:\n');
fprintf('- Mamdani: Faster convergence, smaller search space\n');
fprintf('- Sugeno: Higher precision, but requires more iterations\n');
fprintf('- Mamdani: Better for initial prototype development\n');
fprintf('- Sugeno: Better for final deployment optimization\n\n');

%% Save Analysis Results
analysis_results = struct();
analysis_results.mamdani_chromosome_length = mamdani_chromosome_length;
analysis_results.sugeno_chromosome_length = sugeno_chromosome_length + 15;
analysis_results.sugeno_param_bounds = sugeno_param_bounds;
analysis_results.test_input = test_input;
try
    analysis_results.mamdani_output = mamdani_output;
catch
    analysis_results.mamdani_output = [NaN, NaN, NaN];
end
analysis_results.sugeno_output = sugeno_output;

save('mamdani_sugeno_comparison.mat', 'analysis_results');
fprintf('Comparison analysis saved to mamdani_sugeno_comparison.mat\n');

fprintf('\nSugeno model analysis completed!\n');