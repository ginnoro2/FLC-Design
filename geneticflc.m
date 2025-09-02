%% Genetic Algorithm Optimization of Smart Home FLC
% This script optimizes the membership function parameters of the FLC
% using genetic algorithm to minimize control error and energy consumption
%
% Author: Priyanka Rai
% Course: Evolutionary and Fuzzy Systems

clear all; close all; clc;

%% Configuration and Base FIS Creation
fprintf('=== GA Optimization of Smart Home FLC ===\n');

% Create a simple base FIS for testing (avoiding file path dependencies)
base_fis = create_base_fis();
fprintf('Base FIS created successfully!\n');

%% Generate Training Dataset
% Create realistic training data based on typical smart home scenarios
n_samples = 200; % Reduced for faster processing

% Generate input scenarios
temperature = [
    normrnd(18, 2, round(n_samples*0.3), 1);  % Cold scenarios
    normrnd(22, 1, round(n_samples*0.4), 1);  % Comfortable scenarios
    normrnd(28, 3, round(n_samples*0.3), 1)   % Hot scenarios
];
temperature = max(15, min(35, temperature)); % Constrain to valid range

light_level = [
    unifrnd(0, 150, round(n_samples*0.3), 1);    % Dark scenarios
    unifrnd(200, 600, round(n_samples*0.4), 1);  % Moderate light
    unifrnd(700, 1000, round(n_samples*0.3), 1)  % Bright scenarios
];

time_of_day = [
    unifrnd(22, 24, round(n_samples*0.2), 1);    % Night
    unifrnd(0, 6, round(n_samples*0.1), 1);      % Early morning
    unifrnd(6, 18, round(n_samples*0.5), 1);     % Day
    unifrnd(18, 22, round(n_samples*0.2), 1)     % Evening
];
time_of_day = mod(time_of_day, 24); % Wrap around 24 hours

activity_level = [
    unifrnd(0, 20, round(n_samples*0.3), 1);     % Resting
    unifrnd(30, 70, round(n_samples*0.5), 1);    % Moderate activity
    unifrnd(80, 100, round(n_samples*0.2), 1)    % High activity
];

user_preference = [
    unifrnd(1, 2.5, round(n_samples*0.3), 1);    % Cool preference
    unifrnd(2.5, 3.5, round(n_samples*0.4), 1);  % Neutral preference
    unifrnd(3.5, 5, round(n_samples*0.3), 1)     % Warm preference
];

% Ensure all vectors are same length
min_length = min([length(temperature), length(light_level), length(time_of_day), ...
                  length(activity_level), length(user_preference)]);
training_inputs = [temperature(1:min_length), light_level(1:min_length), ...
                  time_of_day(1:min_length), activity_level(1:min_length), ...
                  user_preference(1:min_length)];

% Generate desired outputs using expert knowledge and comfort models
desired_outputs = zeros(min_length, 3);
for i = 1:min_length
    temp = training_inputs(i, 1);
    light = training_inputs(i, 2);
    time = training_inputs(i, 3);
    activity = training_inputs(i, 4);
    pref = training_inputs(i, 5);

    % HVAC Control (desired comfort temperature based on preference)
    comfort_temp = 20 + (pref - 1) * 2; % Range: 20-28°C based on preference
    temp_error = temp - comfort_temp;
    desired_outputs(i, 1) = -temp_error * 10; % Proportional control
    desired_outputs(i, 1) = max(-100, min(100, desired_outputs(i, 1))); % Saturate

    % Lighting Control (based on ambient light and time)
    if time >= 6 && time <= 18 % Day time
        desired_light = max(0, 60 - light/10); % Less artificial light when bright
    else % Night/Evening
        desired_light = max(20, 80 - light/15); % More artificial light needed
    end
    if activity > 70, desired_light = min(100, desired_light + 20); end
    desired_outputs(i, 2) = desired_light;

    % Blind Position (privacy and light control)
    if time >= 22 || time <= 6
        desired_blind = 10; % Mostly closed
    elseif light > 700
        desired_blind = 40; % Very bright - partial closure
    else
        desired_blind = 70; % Mostly open
    end
    desired_outputs(i, 3) = desired_blind;
end

fprintf('Training dataset generated: %d samples\n', min_length);

%% Chromosome Definition
chromosome_length = 15; % Simplified: only input MF centers
param_bounds = [
    % Temperature MF centers [Cold, Comfortable, Hot]
    16, 19; 20, 24; 26, 32;
    % Light Level MF centers [Dark, Dim, Bright]
    50, 100; 250, 400; 600, 800;
    % Time of Day MF centers [Night, Day, Evening]
    2, 5; 10, 14; 18, 22;
    % Activity Level MF centers [Resting, Moderate, Active]
    5, 20; 40, 60; 80, 95;
    % User Preference MF centers [Cool, Neutral, Warm]
    1.2, 2.0; 2.5, 3.5; 4.0, 4.8;
];

%% GA Parameters
ga_params = struct();
ga_params.pop_size = 30; % Reduced for faster testing
ga_params.max_generations = 50; % Reduced for faster testing
ga_params.crossover_rate = 0.8;
ga_params.mutation_rate = 0.1;
ga_params.elitism_count = 2;
ga_params.tournament_size = 3;

fprintf('\nGenetic Algorithm Parameters:\n');
fprintf('Population Size: %d\n', ga_params.pop_size);
fprintf('Max Generations: %d\n', ga_params.max_generations);
fprintf('Crossover Rate: %.2f\n', ga_params.crossover_rate);
fprintf('Mutation Rate: %.2f\n', ga_params.mutation_rate);

%% Initialize Population
population = initialize_population(ga_params.pop_size, chromosome_length, param_bounds);
fprintf('Initial population created\n');

%% Evolution Loop
best_fitness_history = zeros(ga_params.max_generations, 1);
mean_fitness_history = zeros(ga_params.max_generations, 1);
best_chromosome = [];
best_fitness = -inf;

fprintf('\nStarting Evolution...\n');
fprintf('Generation | Best Fitness | Mean Fitness | Std Fitness\n');
fprintf('-----------|--------------|--------------|------------\n');

for generation = 1:ga_params.max_generations
    % Evaluate fitness for all individuals
    fitness_values = zeros(ga_params.pop_size, 1);
    for i = 1:ga_params.pop_size
        fitness_values(i) = evaluate_fitness(population(i, :), training_inputs, ...
            desired_outputs, base_fis, param_bounds);
    end

    % Track statistics
    [current_best_fitness, best_idx] = max(fitness_values);
    mean_fitness = mean(fitness_values);
    std_fitness = std(fitness_values);

    best_fitness_history(generation) = current_best_fitness;
    mean_fitness_history(generation) = mean_fitness;

    % Update global best
    if current_best_fitness > best_fitness
        best_fitness = current_best_fitness;
        best_chromosome = population(best_idx, :);
    end

    fprintf('%10d | %12.4f | %12.4f | %11.4f\n', ...
        generation, current_best_fitness, mean_fitness, std_fitness);

    % Selection, Crossover, and Mutation
    if generation < ga_params.max_generations
        new_population = evolve_population(population, fitness_values, ga_params, param_bounds);
        population = new_population;
    end
end

%% Results Analysis
fprintf('\n=== Optimization Results ===\n');
fprintf('Best Fitness: %.6f\n', best_fitness);
fprintf('Final Mean Fitness: %.6f\n', mean_fitness_history(end));

% Create optimized FIS
optimized_fis = create_optimized_fis(best_chromosome, base_fis, param_bounds);
writeFIS(optimized_fis, 'optimized_smart_home_flc.fis');
fprintf('Optimized FIS saved as optimized_smart_home_flc.fis\n');

%% Performance Comparison
fprintf('\n=== Performance Comparison ===\n');

original_outputs = zeros(size(desired_outputs));
optimized_outputs = zeros(size(desired_outputs));
for i = 1:size(training_inputs, 1)
    original_outputs(i, :) = safe_evalfis(base_fis, training_inputs(i, :));
    optimized_outputs(i, :) = safe_evalfis(optimized_fis, training_inputs(i, :));
end

% Calculate errors
original_error = sqrt(mean(sum((original_outputs - desired_outputs).^2, 2)));
optimized_error = sqrt(mean(sum((optimized_outputs - desired_outputs).^2, 2)));
fprintf('Original FIS Error (RMSE): %.4f\n', original_error);
fprintf('Optimized FIS Error (RMSE): %.4f\n', optimized_error);
if original_error > 0
    fprintf('Improvement: %.2f%%\n', (original_error - optimized_error) / original_error * 100);
end

%% Visualization
figure('Position', [100, 100, 1200, 800]);

subplot(2,3,1);
plot(1:ga_params.max_generations, best_fitness_history, 'b-', 'LineWidth', 2); hold on;
plot(1:ga_params.max_generations, mean_fitness_history, 'r--', 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Fitness'); title('GA Convergence');
legend('Best Fitness', 'Mean Fitness', 'Location', 'best'); grid on;

subplot(2,3,2);
scatter(desired_outputs(:,1), original_outputs(:,1), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6); hold on;
scatter(desired_outputs(:,1), optimized_outputs(:,1), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
plot([-100, 100], [-100, 100], 'k--', 'LineWidth', 1);
xlabel('Desired HVAC Control'); ylabel('Actual HVAC Control');
title('HVAC Control Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

subplot(2,3,3);
scatter(desired_outputs(:,2), original_outputs(:,2), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6); hold on;
scatter(desired_outputs(:,2), optimized_outputs(:,2), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
plot([0, 100], [0, 100], 'k--', 'LineWidth', 1);
xlabel('Desired Lighting Control'); ylabel('Actual Lighting Control');
title('Lighting Control Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

subplot(2,3,4);
scatter(desired_outputs(:,3), original_outputs(:,3), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6); hold on;
scatter(desired_outputs(:,3), optimized_outputs(:,3), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
plot([0, 100], [0, 100], 'k--', 'LineWidth', 1);
xlabel('Desired Blind Position'); ylabel('Actual Blind Position');
title('Blind Position Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

% Error distribution comparison
subplot(2,3,5);
original_errors = sqrt(sum((original_outputs - desired_outputs).^2, 2));
optimized_errors = sqrt(sum((optimized_outputs - desired_outputs).^2, 2));
histogram(original_errors, 20, 'FaceAlpha', 0.6, 'DisplayName', 'Original'); hold on;
histogram(optimized_errors, 20, 'FaceAlpha', 0.6, 'DisplayName', 'Optimized');
xlabel('Error Magnitude'); ylabel('Frequency'); title('Error Distribution');
legend('Location', 'best'); grid on;

% Best chromosome visualization
subplot(2,3,6);
bar(best_chromosome);
xlabel('Parameter Index'); ylabel('Parameter Value'); title('Optimized Parameters');
grid on;

sgtitle('GA Optimization Results for Smart Home FLC', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'ga_optimization_results.png');

fprintf('\nOptimization completed! Results saved.\n');

%% =================== Supporting Functions ===================

function fis = create_base_fis()
    % Create a simple Mamdani FIS for smart home control
    fis = mamfis('Name', 'SmartHomeFLC');
    
    % Inputs
    fis = addInput(fis, [15 35], 'Name', 'Temperature');
    fis = addMF(fis, 'Temperature', 'trimf', [15 17.5 20], 'Name', 'Cold');
    fis = addMF(fis, 'Temperature', 'trimf', [18 22 26], 'Name', 'Comfortable');
    fis = addMF(fis, 'Temperature', 'trimf', [24 28 35], 'Name', 'Hot');
    
    fis = addInput(fis, [0 1000], 'Name', 'LightLevel');
    fis = addMF(fis, 'LightLevel', 'trimf', [0 75 150], 'Name', 'Dark');
    fis = addMF(fis, 'LightLevel', 'trimf', [100 350 600], 'Name', 'Dim');
    fis = addMF(fis, 'LightLevel', 'trimf', [500 750 1000], 'Name', 'Bright');
    
    fis = addInput(fis, [0 24], 'Name', 'TimeOfDay');
    fis = addMF(fis, 'TimeOfDay', 'trimf', [0 3 6], 'Name', 'Night');
    fis = addMF(fis, 'TimeOfDay', 'trimf', [6 12 18], 'Name', 'Day');
    fis = addMF(fis, 'TimeOfDay', 'trimf', [16 20 24], 'Name', 'Evening');
    
    fis = addInput(fis, [0 100], 'Name', 'ActivityLevel');
    fis = addMF(fis, 'ActivityLevel', 'trimf', [0 10 30], 'Name', 'Resting');
    fis = addMF(fis, 'ActivityLevel', 'trimf', [20 50 80], 'Name', 'Moderate');
    fis = addMF(fis, 'ActivityLevel', 'trimf', [70 90 100], 'Name', 'Active');
    
    fis = addInput(fis, [1 5], 'Name', 'UserPreference');
    fis = addMF(fis, 'UserPreference', 'trimf', [1 1.5 2.5], 'Name', 'Cool');
    fis = addMF(fis, 'UserPreference', 'trimf', [2 3 4], 'Name', 'Neutral');
    fis = addMF(fis, 'UserPreference', 'trimf', [3.5 4.5 5], 'Name', 'Warm');
    
    % Outputs
    fis = addOutput(fis, [-100 100], 'Name', 'HVACControl');
    fis = addMF(fis, 'HVACControl', 'trimf', [-100 -70 -40], 'Name', 'StrongCooling');
    fis = addMF(fis, 'HVACControl', 'trimf', [-30 -15 0], 'Name', 'LightCooling');
    fis = addMF(fis, 'HVACControl', 'trimf', [-5 0 5], 'Name', 'Off');
    fis = addMF(fis, 'HVACControl', 'trimf', [0 15 30], 'Name', 'LightHeating');
    fis = addMF(fis, 'HVACControl', 'trimf', [40 70 100], 'Name', 'StrongHeating');
    
    fis = addOutput(fis, [0 100], 'Name', 'LightingControl');
    fis = addMF(fis, 'LightingControl', 'trimf', [0 5 15], 'Name', 'Off');
    fis = addMF(fis, 'LightingControl', 'trimf', [10 30 50], 'Name', 'Dim');
    fis = addMF(fis, 'LightingControl', 'trimf', [40 65 85], 'Name', 'Medium');
    fis = addMF(fis, 'LightingControl', 'trimf', [80 95 100], 'Name', 'Bright');
    
    fis = addOutput(fis, [0 100], 'Name', 'BlindPosition');
    fis = addMF(fis, 'BlindPosition', 'trimf', [0 10 20], 'Name', 'Closed');
    fis = addMF(fis, 'BlindPosition', 'trimf', [15 40 65], 'Name', 'PartiallyOpen');
    fis = addMF(fis, 'BlindPosition', 'trimf', [60 85 100], 'Name', 'FullyOpen');
    
    % Add some basic rules
    rules = [
        1 1 1 1 1 2 4 1 1 1;  % Cold+Dark+Night+Resting+Cool -> LightCooling, Bright, Closed
        1 2 2 2 2 3 3 2 1 1;  % Cold+Dim+Day+Moderate+Neutral -> Off, Medium, PartiallyOpen
        2 3 2 3 2 1 2 3 1 1;  % Comfortable+Bright+Day+Active+Neutral -> StrongCooling, Dim, FullyOpen
        3 1 3 1 3 1 4 1 1 1;  % Hot+Dark+Evening+Resting+Warm -> StrongCooling, Bright, Closed
        3 3 2 3 3 1 1 2 1 1;  % Hot+Bright+Day+Active+Warm -> StrongCooling, Off, PartiallyOpen
    ];
    
    fis = addRule(fis, rules);
end

function population = initialize_population(pop_size, chromosome_length, param_bounds)
    population = zeros(pop_size, chromosome_length);
    for i = 1:pop_size
        for j = 1:chromosome_length
            min_val = param_bounds(j, 1);
            max_val = param_bounds(j, 2);
            population(i, j) = min_val + (max_val - min_val) * rand();
        end
    end
end

function fitness = evaluate_fitness(chromosome, inputs, desired_outputs, base_fis, param_bounds)
    try
        % Create FIS with modified parameters
        modified_fis = create_optimized_fis(chromosome, base_fis, param_bounds);

        % Evaluate on all training samples
        predicted_outputs = zeros(size(desired_outputs));
        for i = 1:size(inputs, 1)
            predicted_outputs(i, :) = safe_evalfis(modified_fis, inputs(i, :));
        end

        % Calculate fitness (negative RMSE with weights)
        weights = [0.4, 0.3, 0.3]; % Weight importance of outputs
        weighted_error = 0;
        for j = 1:3
            mse = mean((predicted_outputs(:, j) - desired_outputs(:, j)).^2);
            weighted_error = weighted_error + weights(j) * mse;
        end
        rmse = sqrt(weighted_error);
        fitness = -rmse; % GA maximizes fitness

        % Add penalty for invalid MF ordering
        penalty = calculate_mf_penalty(chromosome);
        fitness = fitness - penalty;

    catch ME
        % Severe penalty for invalid FIS or runtime errors
        fitness = -1000;
        fprintf('Fitness evaluation error: %s\n', ME.message);
    end
end

function y = safe_evalfis(fis, x_input)
    % Safe wrapper around evalfis to handle errors
    try
        % Try different evaluation methods
        if exist('evalfis', 'file') == 2
            y = evalfis(fis, x_input);
        else
            % Alternative method - use defuzzification manually
            y = manual_fis_eval(fis, x_input);
        end
        
        % Check for valid output
        if isempty(y) || any(~isfinite(y))
            y = get_default_outputs(fis);
        end
    catch
        y = get_default_outputs(fis);
    end
end

function y = manual_fis_eval(fis, x_input)
    % Manual FIS evaluation when evalfis is not available
    num_outputs = length(fis.Outputs);
    y = zeros(1, num_outputs);
    
    % Simple centroid defuzzification for each output
    for out_idx = 1:num_outputs
        output_range = fis.Outputs(out_idx).Range;
        y_values = linspace(output_range(1), output_range(2), 100);
        membership_sum = 0;
        value_sum = 0;
        
        for y_val = y_values
            % Calculate overall membership for this output value
            rule_activations = [];
            for rule_idx = 1:length(fis.Rules)
                rule = fis.Rules(rule_idx);
                if rule.Consequent(out_idx) > 0
                    % Calculate rule activation
                    activation = 1;
                    for in_idx = 1:length(fis.Inputs)
                        if rule.Antecedent(in_idx) > 0
                            mf = fis.Inputs(in_idx).MembershipFunctions(rule.Antecedent(in_idx));
                            activation = activation * evaluate_mf(mf, x_input(in_idx));
                        end
                    end
                    
                    % Get consequent membership
                    out_mf = fis.Outputs(out_idx).MembershipFunctions(rule.Consequent(out_idx));
                    consequent_membership = evaluate_mf(out_mf, y_val);
                    
                    rule_activations(end+1) = min(activation, consequent_membership);
                end
            end
            
            if ~isempty(rule_activations)
                overall_membership = max(rule_activations);
                membership_sum = membership_sum + overall_membership;
                value_sum = value_sum + y_val * overall_membership;
            end
        end
        
        if membership_sum > 0
            y(out_idx) = value_sum / membership_sum;
        else
            y(out_idx) = mean(output_range);
        end
    end
end

function membership = evaluate_mf(mf, x)
    % Evaluate membership function at point x
    params = mf.Parameters;
    switch mf.Type
        case 'trimf'
            if length(params) >= 3
                a = params(1); b = params(2); c = params(3);
                if x <= a || x >= c
                    membership = 0;
                elseif x <= b
                    membership = (x - a) / (b - a);
                else
                    membership = (c - x) / (c - b);
                end
            else
                membership = 0;
            end
        case 'trapmf'
            if length(params) >= 4
                a = params(1); b = params(2); c = params(3); d = params(4);
                if x <= a || x >= d
                    membership = 0;
                elseif x <= b
                    membership = (x - a) / (b - a);
                elseif x <= c
                    membership = 1;
                else
                    membership = (d - x) / (d - c);
                end
            else
                membership = 0;
            end
        otherwise
            membership = 0;
    end
    
    % Ensure valid membership value
    membership = max(0, min(1, membership));
end

function y = get_default_outputs(fis)
    % Return default output values (center of ranges)
    num_outputs = length(fis.Outputs);
    y = zeros(1, num_outputs);
    for i = 1:num_outputs
        range_vals = fis.Outputs(i).Range;
        y(i) = mean(range_vals);
    end
end

function penalty = calculate_mf_penalty(chromosome)
    % Penalty for invalid MF ordering
    penalty = 0;
    mf_groups = {
        [1, 2, 3];      % Temperature
        [4, 5, 6];      % Light Level
        [7, 8, 9];      % Time of Day
        [10, 11, 12];   % Activity Level
        [13, 14, 15];   % User Preference
    };
    
    for group_idx = 1:length(mf_groups)
        group = mf_groups{group_idx};
        centers = chromosome(group);
        % Check if centers are in ascending order
        if any(diff(centers) <= 0)
            penalty = penalty + 50;
        end
    end
end

function new_population = evolve_population(population, fitness_values, ga_params, param_bounds)
    [pop_size, chromosome_length] = size(population);
    new_population = zeros(pop_size, chromosome_length);

    % Elitism - keep best individuals
    [~, idx] = sort(fitness_values, 'descend');
    for i = 1:ga_params.elitism_count
        new_population(i, :) = population(idx(i), :);
    end

    % Generate rest via selection/crossover/mutation
    for i = (ga_params.elitism_count + 1):pop_size
        parent1 = tournament_selection(population, fitness_values, ga_params.tournament_size);
        parent2 = tournament_selection(population, fitness_values, ga_params.tournament_size);

        % Crossover
        if rand() < ga_params.crossover_rate
            [child1, ~] = single_point_crossover(parent1, parent2);
            child = child1;
        else
            child = parent1;
        end

        % Mutation
        if rand() < ga_params.mutation_rate
            child = mutate_chromosome(child, param_bounds);
        end

        new_population(i, :) = child;
    end
end

function parent = tournament_selection(population, fitness_values, tournament_size)
    pop_size = size(population, 1);
    tournament_indices = randperm(pop_size, tournament_size);
    [~, winner_idx] = max(fitness_values(tournament_indices));
    parent = population(tournament_indices(winner_idx), :);
end

function [child1, child2] = single_point_crossover(parent1, parent2)
    chromosome_length = length(parent1);
    crossover_point = randi(chromosome_length - 1);
    child1 = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
    child2 = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
end

function mutated = mutate_chromosome(chromosome, param_bounds)
    mutated = chromosome;
    chromosome_length = length(chromosome);
    
    for i = 1:chromosome_length
        if rand() < 0.1 % 10% chance for each gene
            % Gaussian mutation
            sigma = 0.1 * (param_bounds(i, 2) - param_bounds(i, 1));
            mutated(i) = mutated(i) + normrnd(0, sigma);
            % Ensure bounds
            mutated(i) = max(param_bounds(i, 1), min(param_bounds(i, 2), mutated(i)));
        end
    end
end

function optimized_fis = create_optimized_fis(chromosome, base_fis, param_bounds)
    % Create optimized FIS by modifying input MF centers
    optimized_fis = base_fis;
    
    try
        % Update Temperature MFs
        optimized_fis.Inputs(1).MembershipFunctions(1).Parameters(2) = clamp_value(chromosome(1), param_bounds(1, :));
        optimized_fis.Inputs(1).MembershipFunctions(2).Parameters(2) = clamp_value(chromosome(2), param_bounds(2, :));
        optimized_fis.Inputs(1).MembershipFunctions(3).Parameters(2) = clamp_value(chromosome(3), param_bounds(3, :));
        
        % Update Light Level MFs
        optimized_fis.Inputs(2).MembershipFunctions(1).Parameters(2) = clamp_value(chromosome(4), param_bounds(4, :));
        optimized_fis.Inputs(2).MembershipFunctions(2).Parameters(2) = clamp_value(chromosome(5), param_bounds(5, :));
        optimized_fis.Inputs(2).MembershipFunctions(3).Parameters(2) = clamp_value(chromosome(6), param_bounds(6, :));
        
        % Update Time of Day MFs
        optimized_fis.Inputs(3).MembershipFunctions(1).Parameters(2) = clamp_value(chromosome(7), param_bounds(7, :));
        optimized_fis.Inputs(3).MembershipFunctions(2).Parameters(2) = clamp_value(chromosome(8), param_bounds(8, :));
        optimized_fis.Inputs(3).MembershipFunctions(3).Parameters(2) = clamp_value(chromosome(9), param_bounds(9, :));
        
        % Update Activity Level MFs
        optimized_fis.Inputs(4).MembershipFunctions(1).Parameters(2) = clamp_value(chromosome(10), param_bounds(10, :));
        optimized_fis.Inputs(4).MembershipFunctions(2).Parameters(2) = clamp_value(chromosome(11), param_bounds(11, :));
        optimized_fis.Inputs(4).MembershipFunctions(3).Parameters(2) = clamp_value(chromosome(12), param_bounds(12, :));
        
        % Update User Preference MFs
        optimized_fis.Inputs(5).MembershipFunctions(1).Parameters(2) = clamp_value(chromosome(13), param_bounds(13, :));
        optimized_fis.Inputs(5).MembershipFunctions(2).Parameters(2) = clamp_value(chromosome(14), param_bounds(14, :));
        optimized_fis.Inputs(5).MembershipFunctions(3).Parameters(2) = clamp_value(chromosome(15), param_bounds(15, :));
        
        % Update triangular MF bounds to maintain proper shape
        for input_idx = 1:length(optimized_fis.Inputs)
            for mf_idx = 1:length(optimized_fis.Inputs(input_idx).MembershipFunctions)
                mf = optimized_fis.Inputs(input_idx).MembershipFunctions(mf_idx);
                center = mf.Parameters(2);
                input_range = optimized_fis.Inputs(input_idx).Range;
                
                % Adjust left and right bounds based on center
                if mf_idx == 1
                    % First MF: left bound at range start
                    left_bound = input_range(1);
                    right_bound = min(center + (center - left_bound), input_range(2));
                elseif mf_idx == length(optimized_fis.Inputs(input_idx).MembershipFunctions)
                    % Last MF: right bound at range end
                    right_bound = input_range(2);
                    left_bound = max(center - (right_bound - center), input_range(1));
                else
                    % Middle MFs: symmetric around center
                    spread = (input_range(2) - input_range(1)) / 6;
                    left_bound = max(center - spread, input_range(1));
                    right_bound = min(center + spread, input_range(2));
                end
                
                optimized_fis.Inputs(input_idx).MembershipFunctions(mf_idx).Parameters = ...
                    [left_bound, center, right_bound];
            end
        end
        
    catch ME
        fprintf('Error creating optimized FIS: %s\n', ME.message);
        optimized_fis = base_fis;
    end
end

function clamped_value = clamp_value(value, bounds)
    % Clamp value within specified bounds
    clamped_value = max(bounds(1), min(bounds(2), value));
end

function performance_metrics = analyze_performance(fis, test_inputs, test_outputs)
    % Comprehensive performance analysis
    predicted = zeros(size(test_outputs));
    for i = 1:size(test_inputs, 1)
        predicted(i, :) = safe_evalfis(fis, test_inputs(i, :));
    end
    
    % Calculate various metrics
    performance_metrics = struct();
    for output_idx = 1:size(test_outputs, 2)
        actual = test_outputs(:, output_idx);
        pred = predicted(:, output_idx);
        
        % Root Mean Square Error
        rmse = sqrt(mean((actual - pred).^2));
        
        % Mean Absolute Error
        mae = mean(abs(actual - pred));
        
        % Correlation coefficient
        if std(actual) > 0 && std(pred) > 0
            correlation = corrcoef(actual, pred);
            r_squared = correlation(1,2)^2;
        else
            r_squared = 0;
        end
        
        % Store metrics
        performance_metrics.(sprintf('output_%d', output_idx)) = struct(...
            'RMSE', rmse, 'MAE', mae, 'R_squared', r_squared);
    end
    
    % Overall performance
    overall_rmse = sqrt(mean(sum((test_outputs - predicted).^2, 2)));
    performance_metrics.overall_RMSE = overall_rmse;
end

function save_optimization_report(best_chromosome, best_fitness, fitness_history, ...
                                performance_metrics, ga_params)
    % Save detailed optimization report
    filename = sprintf('ga_optimization_report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
    fid = fopen(filename, 'w');
    
    fprintf(fid, '=== GA Optimization Report ===\n\n');
    fprintf(fid, 'Date: %s\n', datestr(now));
    fprintf(fid, 'Algorithm: Genetic Algorithm\n');
    fprintf(fid, 'Application: Smart Home Fuzzy Logic Controller Optimization\n\n');
    
    fprintf(fid, '--- GA Parameters ---\n');
    fprintf(fid, 'Population Size: %d\n', ga_params.pop_size);
    fprintf(fid, 'Generations: %d\n', ga_params.max_generations);
    fprintf(fid, 'Crossover Rate: %.2f\n', ga_params.crossover_rate);
    fprintf(fid, 'Mutation Rate: %.2f\n', ga_params.mutation_rate);
    fprintf(fid, 'Elitism Count: %d\n', ga_params.elitism_count);
    fprintf(fid, 'Tournament Size: %d\n\n', ga_params.tournament_size);
    
    fprintf(fid, '--- Best Solution ---\n');
    fprintf(fid, 'Best Fitness: %.6f\n', best_fitness);
    fprintf(fid, 'Best Chromosome: ');
    fprintf(fid, '%.4f ', best_chromosome);
    fprintf(fid, '\n\n');
    
    fprintf(fid, '--- Convergence Statistics ---\n');
    fprintf(fid, 'Initial Best Fitness: %.6f\n', fitness_history(1));
    fprintf(fid, 'Final Best Fitness: %.6f\n', fitness_history(end));
    fprintf(fid, 'Improvement: %.6f\n', fitness_history(end) - fitness_history(1));
    
    fclose(fid);
    fprintf('Optimization report saved as: %s\n', filename);
end