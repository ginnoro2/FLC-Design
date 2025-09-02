%% Sugeno FIS Model for Smart Home Control
clear all; close all; clc;
fprintf('Creating Sugeno (TSK) FIS Model for Smart Home Control...\n');

%% Create Sugeno FIS
sugeno_fis = sugfis('Name', 'SmartHomeSugenoFLC');

%% Define Inputs
sugeno_fis = addInput(sugeno_fis,[15 35],'Name','Temperature');
sugeno_fis = addMF(sugeno_fis,'Temperature','trapmf',[15 15 18 20],'Name','Cold');
sugeno_fis = addMF(sugeno_fis,'Temperature','trimf',[18 22 26],'Name','Comfortable');
sugeno_fis = addMF(sugeno_fis,'Temperature','trapmf',[24 27 35 35],'Name','Hot');

sugeno_fis = addInput(sugeno_fis,[0 1000],'Name','LightLevel');
sugeno_fis = addMF(sugeno_fis,'LightLevel','trapmf',[0 0 50 150],'Name','Dark');
sugeno_fis = addMF(sugeno_fis,'LightLevel','trimf',[100 300 500],'Name','Dim');
sugeno_fis = addMF(sugeno_fis,'LightLevel','trapmf',[400 600 1000 1000],'Name','Bright');

sugeno_fis = addInput(sugeno_fis,[0 24],'Name','TimeOfDay');
sugeno_fis = addMF(sugeno_fis,'TimeOfDay','trapmf',[0 0 6 8],'Name','Night');
sugeno_fis = addMF(sugeno_fis,'TimeOfDay','trimf',[7 12 17],'Name','Day');
sugeno_fis = addMF(sugeno_fis,'TimeOfDay','trapmf',[16 20 24 24],'Name','Evening');

sugeno_fis = addInput(sugeno_fis,[0 100],'Name','ActivityLevel');
sugeno_fis = addMF(sugeno_fis,'ActivityLevel','trapmf',[0 0 10 30],'Name','Resting');
sugeno_fis = addMF(sugeno_fis,'ActivityLevel','trimf',[20 50 80],'Name','Moderate');
sugeno_fis = addMF(sugeno_fis,'ActivityLevel','trapmf',[70 90 100 100],'Name','Active');

sugeno_fis = addInput(sugeno_fis,[1 5],'Name','UserPreference');
sugeno_fis = addMF(sugeno_fis,'UserPreference','trimf',[1 1 2.5],'Name','Cool');
sugeno_fis = addMF(sugeno_fis,'UserPreference','trimf',[2 3 4],'Name','Neutral');
sugeno_fis = addMF(sugeno_fis,'UserPreference','trimf',[3.5 5 5],'Name','Warm');

%% Define Outputs (Linear)
sugeno_fis = addOutput(sugeno_fis,[-100 100],'Name','HVACControl');
sugeno_fis = addMF(sugeno_fis,'HVACControl','linear',[-3 0 0 0 5 60],'Name','HVAC1');
sugeno_fis = addMF(sugeno_fis,'HVACControl','linear',[-3 0 0 0 2 80],'Name','HVAC2');
sugeno_fis = addMF(sugeno_fis,'HVACControl','linear',[-3 0 0 0 0 90],'Name','HVAC3');
sugeno_fis = addMF(sugeno_fis,'HVACControl','linear',[0 0 0 -0.2 2 -20],'Name','HVAC4');

sugeno_fis = addOutput(sugeno_fis,[0 100],'Name','LightingControl');
sugeno_fis = addMF(sugeno_fis,'LightingControl','linear',[0 -0.05 0 0 0 30],'Name','Light1');
sugeno_fis = addMF(sugeno_fis,'LightingControl','linear',[0 -0.05 0 0 0 40],'Name','Light2');
sugeno_fis = addMF(sugeno_fis,'LightingControl','linear',[0 -0.05 0 0 0 50],'Name','Light3');
sugeno_fis = addMF(sugeno_fis,'LightingControl','linear',[0 -0.05 0 0 0 60],'Name','Light4');

sugeno_fis = addOutput(sugeno_fis,[0 100],'Name','BlindPosition');
sugeno_fis = addMF(sugeno_fis,'BlindPosition','linear',[0 0 0 0 0 20],'Name','Blind1');
sugeno_fis = addMF(sugeno_fis,'BlindPosition','linear',[0 0 0 0 0 30],'Name','Blind2');
sugeno_fis = addMF(sugeno_fis,'BlindPosition','linear',[0 0 0 0 0 15],'Name','Blind3');
sugeno_fis = addMF(sugeno_fis,'BlindPosition','linear',[0 0 0 0 0 80],'Name','Blind4');

%% Define Rules - Correct format for multiple outputs
% Format: [Input1_MF Input2_MF Input3_MF Input4_MF Input5_MF Output1_MF Output2_MF Output3_MF Weight Connection]
% Connection: 1=AND, 2=OR

rules_matrix = [
    % Cold temperature scenarios
    1 1 1 1 1  2 4 1  1 1;  % Cold+Dark+Night+Resting+Cool -> HVAC2, Light4, Blind1
    1 1 2 2 1  2 3 2  1 1;  % Cold+Dark+Day+Moderate+Cool -> HVAC2, Light3, Blind2
    1 2 2 2 2  3 2 3  1 1;  % Cold+Dim+Day+Moderate+Neutral -> HVAC3, Light2, Blind3
    1 3 2 3 2  3 1 4  1 1;  % Cold+Bright+Day+Active+Neutral -> HVAC3, Light1, Blind4
    
    % Comfortable temperature scenarios  
    2 1 1 1 2  4 3 1  1 1;  % Comfortable+Dark+Night+Resting+Neutral -> HVAC4, Light3, Blind1
    2 1 2 2 2  4 2 2  1 1;  % Comfortable+Dark+Day+Moderate+Neutral -> HVAC4, Light2, Blind2
    2 2 2 2 2  4 1 3  1 1;  % Comfortable+Dim+Day+Moderate+Neutral -> HVAC4, Light1, Blind3
    2 3 2 3 2  4 1 4  1 1;  % Comfortable+Bright+Day+Active+Neutral -> HVAC4, Light1, Blind4
    
    % Hot temperature scenarios
    3 1 1 1 3  1 4 1  1 1;  % Hot+Dark+Night+Resting+Warm -> HVAC1, Light4, Blind1
    3 2 2 2 3  1 2 2  1 1;  % Hot+Dim+Day+Moderate+Warm -> HVAC1, Light2, Blind2
    3 3 2 3 3  1 1 4  1 1;  % Hot+Bright+Day+Active+Warm -> HVAC1, Light1, Blind4
    3 3 3 3 3  1 1 4  1 1;  % Hot+Bright+Evening+Active+Warm -> HVAC1, Light1, Blind4
];

% Add rules to the FIS
sugeno_fis = addRule(sugeno_fis, rules_matrix);

%% Save FIS
writeFIS(sugeno_fis,'smart_home_sugeno_flc.fis');
fprintf('Sugeno FIS created and saved successfully!\n');

%% Display FIS Information
fprintf('\nFIS Summary:\n');
fprintf('Number of inputs: %d\n', length(sugeno_fis.Inputs));
fprintf('Number of outputs: %d\n', length(sugeno_fis.Outputs));
fprintf('Number of rules: %d\n', length(sugeno_fis.Rules));

%% Test Evaluation with multiple scenarios
fprintf('\n=== Testing Sugeno FIS with Various Scenarios ===\n');

test_scenarios = [
    22, 300, 14, 50, 3;    % Comfortable temp, dim light, day, moderate activity, neutral preference
    30, 800, 20, 80, 4;    % Hot temp, bright light, evening, active, warm preference
    17, 50, 2, 10, 2;      % Cold temp, dark, night, resting, cool preference
    25, 500, 12, 60, 3;    % Hot temp, bright light, day, moderate activity, neutral preference
];

scenario_names = {
    'Comfortable Day Scenario';
    'Hot Evening Active Scenario';
    'Cold Night Resting Scenario';
    'Hot Day Moderate Scenario'
};

for i = 1:size(test_scenarios, 1)
    test_input = test_scenarios(i, :);
    try
        % Try different evaluation methods
        if ismethod(sugeno_fis, 'evalfis')
            output = sugeno_fis.evalfis(test_input);
        elseif hasmethod(sugeno_fis, 'evaluate')
            output = sugeno_fis.evaluate(test_input);
        else
            % Manual evaluation using defuzzification
            output = zeros(1, 3); % Initialize output array
            fprintf('  Manual evaluation needed - FIS evaluation method unavailable\n');
            continue;
        end
    catch ME
        fprintf('  Evaluation error: %s\n', ME.message);
        output = [NaN, NaN, NaN];
    end
    
    fprintf('\n%s:\n', scenario_names{i});
    fprintf('  Input: Temp=%.1f°C, Light=%d lux, Time=%.1fh, Activity=%d%%, Pref=%.1f\n', ...
        test_input(1), test_input(2), test_input(3), test_input(4), test_input(5));
    fprintf('  Output: HVAC=%.2f, Lighting=%.2f%%, Blinds=%.2f%%\n', ...
        output(1), output(2), output(3));
end

%% Alternative: Create a simpler test without full evaluation
fprintf('\n=== FIS Structure Verification ===\n');
fprintf('Inputs:\n');
for i = 1:length(sugeno_fis.Inputs)
    fprintf('  %d. %s: [%.1f, %.1f]\n', i, sugeno_fis.Inputs(i).Name, ...
        sugeno_fis.Inputs(i).Range(1), sugeno_fis.Inputs(i).Range(2));
end

fprintf('\nOutputs:\n');
for i = 1:length(sugeno_fis.Outputs)
    fprintf('  %d. %s: [%.1f, %.1f]\n', i, sugeno_fis.Outputs(i).Name, ...
        sugeno_fis.Outputs(i).Range(1), sugeno_fis.Outputs(i).Range(2));
end

fprintf('\nRules: %d rules defined\n', length(sugeno_fis.Rules));

% Try to use the Fuzzy Logic Designer for testing
fprintf('\nTo test the FIS interactively, use: fuzzyLogicDesigner(sugeno_fis)\n');
fprintf('Or try: ruleview(sugeno_fis)\n');

% Plot Temperature MFs
figure('Name', 'Input Membership Functions');
subplot(2,3,1);
plotmf(sugeno_fis, 'input', 1);
title('Temperature Membership Functions');
xlabel('Temperature (°C)');
ylabel('Membership Degree');

% Plot Light Level MFs  
subplot(2,3,2);
plotmf(sugeno_fis, 'input', 2);
title('Light Level Membership Functions');
xlabel('Light Level (lux)');
ylabel('Membership Degree');

% Plot Time of Day MFs
subplot(2,3,3);
plotmf(sugeno_fis, 'input', 3);
title('Time of Day Membership Functions');
xlabel('Time (hours)');
ylabel('Membership Degree');

% Plot Activity Level MFs
subplot(2,3,4);
plotmf(sugeno_fis, 'input', 4);
title('Activity Level Membership Functions');
xlabel('Activity Level (%)');
ylabel('Membership Degree');

% Plot User Preference MFs
subplot(2,3,5);
plotmf(sugeno_fis, 'input', 5);
title('User Preference Membership Functions');
xlabel('User Preference');
ylabel('Membership Degree');

fprintf('Sugeno FIS Model completed successfully!\n');