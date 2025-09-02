%% Genetic Algorithm Optimization of Smart Home FLC
% This script optimizes the membership function parameters of the FLC
% using genetic algorithm to minimize control error and energy consumption
%
% Author: Rupak Rajbanshi

clear all; close all; clc;

%% Load Base FIS and Generate Training Data
addpath('../Part1_FLC_Design');
try
    base_fis = readfis('../Part1_FLC_Design/smart_home_flc.fis');
    fprintf('Base FIS loaded successfully!\n');
catch
    fprintf('Base FIS not found. Creating it first...\n');
    run('../Part1_FLC_Design/smart_home_flc.m');
    base_fis = readfis('../Part1_FLC_Design/smart_home_flc.fis');
end

%% Generate Training Dataset
% Create realistic training data based on typical smart home scenarios
n_samples = 500;

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
    if activity > 70, desired_light = min(100, desired_light + 20); end % More light for activity
    desired_outputs(i, 2) = desired_light;
    
    % Blind Position (privacy and light control)
    if time >= 22 || time <= 6 % Night privacy
        desired_blind = 10; % Mostly closed
    elseif light > 700 % Very bright - partial closure
        desired_blind = 40;
    else % Normal day operation
        desired_blind = 70; % Mostly open
    end
    desired_outputs(i, 3) = desired_blind;
end

fprintf('Training dataset generated: %d samples\n', min_length);

%% Define Chromosome Structure
% For Mamdani FIS, we optimize membership function parameters
% Each triangular/trapezoidal MF has 3-4 parameters
% We'll optimize only the peak/center positions to maintain shape consistency

% Input MF parameters to optimize (centers/peaks only for simplicity)
% Temperature: 3 MFs -> 3 parameters (centers)
% Light Level: 3 MFs -> 3 parameters  
% Time of Day: 3 MFs -> 3 parameters
% Activity Level: 3 MFs -> 3 parameters
% User Preference: 3 MFs -> 3 parameters
% Total Input parameters: 15

% Output MF parameters:
% HVAC Control: 5 MFs -> 5 parameters
% Lighting Control: 4 MFs -> 4 parameters  
% Blind Position: 3 MFs -> 3 parameters
% Total Output parameters: 12

% Total chromosome length: 27 parameters
chromosome_length = 27;

% Define parameter bounds [min, max] for each gene
param_bounds = [
    % Temperature MF centers [Cold, Comfortable, Hot]
    16, 19;   % Cold center
    20, 24;   % Comfortable center  
    26, 32;   % Hot center
    
    % Light Level MF centers [Dark, Dim, Bright]
    50, 100;   % Dark center
    250, 400;  % Dim center
    600, 800;  % Bright center
    
    % Time of Day MF centers [Night, Day, Evening]
    2, 5;      % Night center
    10, 14;    % Day center
    18, 22;    % Evening center
    
    % Activity Level MF centers [Resting, Moderate, Active]
    5, 20;     % Resting center
    40, 60;    % Moderate center
    80, 95;    % Active center
    
    % User Preference MF centers [Cool, Neutral, Warm]
    1.2, 2.0;  % Cool center
    2.5, 3.5;  % Neutral center
    4.0, 4.8;  % Warm center
    
    % HVAC Control MF centers [StrongCooling, LightCooling, Off, LightHeating, StrongHeating]
    -80, -60;  % StrongCooling center
    -25, -5;   % LightCooling center
    -5, 5;     % Off center
    5, 25;     % LightHeating center
    60, 80;    % StrongHeating center
    
    % Lighting Control MF centers [Off, Dim, Medium, Bright]
    2, 10;     % Off center
    20, 40;    % Dim center
    55, 75;    % Medium center
    90, 98;    % Bright center
    
    % Blind Position MF centers [Closed, PartiallyOpen, FullyOpen]
    5, 15;     % Closed center
    30, 50;    % PartiallyOpen center
    75, 90     % FullyOpen center
];

%% Genetic Algorithm Parameters
ga_params = struct();
ga_params.pop_size = 50;
ga_params.max_generations = 100;
ga_params.crossover_rate = 0.8;
ga_params.mutation_rate = 0.1;
ga_params.elitism_count = 2;
ga_params.tournament_size = 3;

fprintf('\nGenetic Algorithm Parameters:\n');
fprintf('Population Size: %d\n', ga_params.pop_size);
fprintf('Max Generations: %d\n', ga_params.max_generations);
fprintf('Crossover Rate: %.2f\n', ga_params.crossover_rate);
fprintf('Mutation Rate: %.2f\n', ga_params.mutation_rate);

%% Fitness Function
fitness_function = @(chromosome) evaluate_fitness(chromosome, training_inputs, ...
                                                 desired_outputs, base_fis, param_bounds);

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
        fitness_values(i) = fitness_function(population(i, :));
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

% Test on training data
original_outputs = zeros(size(desired_outputs));
optimized_outputs = zeros(size(desired_outputs));

for i = 1:size(training_inputs, 1)
    original_outputs(i, :) = evalfis(base_fis, training_inputs(i, :));
    optimized_outputs(i, :) = evalfis(optimized_fis, training_inputs(i, :));
end

% Calculate errors
original_error = mean(sqrt(sum((original_outputs - desired_outputs).^2, 2)));
optimized_error = mean(sqrt(sum((optimized_outputs - desired_outputs).^2, 2)));

fprintf('Original FIS Error (RMSE): %.4f\n', original_error);
fprintf('Optimized FIS Error (RMSE): %.4f\n', optimized_error);
fprintf('Improvement: %.2f%%\n', (original_error - optimized_error) / original_error * 100);

%% Visualization
% Plot convergence
figure('Position', [100, 100, 800, 600]);

subplot(2,2,1);
plot(1:ga_params.max_generations, best_fitness_history, 'b-', 'LineWidth', 2);
hold on;
plot(1:ga_params.max_generations, mean_fitness_history, 'r--', 'LineWidth', 1.5);
xlabel('Generation');
ylabel('Fitness');
title('GA Convergence');
legend('Best Fitness', 'Mean Fitness', 'Location', 'best');
grid on;

% Compare outputs
subplot(2,2,2);
scatter(desired_outputs(:,1), original_outputs(:,1), 'b', 'filled', 'Alpha', 0.6);
hold on;
scatter(desired_outputs(:,1), optimized_outputs(:,1), 'r', 'filled', 'Alpha', 0.6);
plot([-100, 100], [-100, 100], 'k--');
xlabel('Desired HVAC Control (%)');
ylabel('Actual HVAC Control (%)');
title('HVAC Control Comparison');
legend('Original', 'Optimized', 'Perfect', 'Location', 'best');
grid on;

subplot(2,2,3);
scatter(desired_outputs(:,2), original_outputs(:,2), 'b', 'filled', 'Alpha', 0.6);
hold on;
scatter(desired_outputs(:,2), optimized_outputs(:,2), 'r', 'filled', 'Alpha', 0.6);
plot([0, 100], [0, 100], 'k--');
xlabel('Desired Lighting Control (%)');
ylabel('Actual Lighting Control (%)');
title('Lighting Control Comparison');
legend('Original', 'Optimized', 'Perfect', 'Location', 'best');
grid on;

subplot(2,2,4);
scatter(desired_outputs(:,3), original_outputs(:,3), 'b', 'filled', 'Alpha', 0.6);
hold on;
scatter(desired_outputs(:,3), optimized_outputs(:,3), 'r', 'filled', 'Alpha', 0.6);
plot([0, 100], [0, 100], 'k--');
xlabel('Desired Blind Position (%)');
ylabel('Actual Blind Position (%)');
title('Blind Position Comparison');
legend('Original', 'Optimized', 'Perfect', 'Location', 'best');
grid on;

sgtitle('GA Optimization Results', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'ga_optimization_results.png');

fprintf('\nOptimization completed! Results saved.\n');

%% Supporting Functions

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
            predicted_outputs(i, :) = evalfis(modified_fis, inputs(i, :));
        end
        
        % Calculate fitness (negative RMSE with weights)
        weights = [0.4, 0.3, 0.3]; % Weight importance of outputs
        weighted_error = 0;
        for j = 1:3
            mse = mean((predicted_outputs(:, j) - desired_outputs(:, j)).^2);
            weighted_error = weighted_error + weights(j) * mse;
        end
        
        rmse = sqrt(weighted_error);
        fitness = -rmse; % Negative because GA maximizes fitness
        
        % Add penalty for invalid MF ordering
        penalty = calculate_mf_penalty(chromosome, param_bounds);
        fitness = fitness - penalty;
        
    catch
        fitness = -1000; % Severe penalty for invalid FIS
    end
end

function penalty = calculate_mf_penalty(chromosome, param_bounds)
    penalty = 0;
    
    % Check MF ordering for each input/output
    mf_groups = {
        [1, 2, 3];      % Temperature
        [4, 5, 6];      % Light Level  
        [7, 8, 9];      % Time of Day
        [10, 11, 12];   % Activity Level
        [13, 14, 15];   % User Preference
        [16, 17, 18, 19, 20]; % HVAC Control
        [21, 22, 23, 24];     % Lighting Control
        [25, 26, 27]          % Blind Position
    };
    
    for group = mf_groups
        indices = group{1};
        values = chromosome(indices);
        
        % Check if MF centers are properly ordered
        if any(diff(values) <= 0)
            penalty = penalty + 50; % Large penalty for ordering violation
        end
    end
end

function new_population = evolve_population(population, fitness_values, ga_params, param_bounds)
    [pop_size, chromosome_length] = size(population);
    new_population = zeros(pop_size, chromosome_length);
    
    % Elitism - keep best individuals
    [~, sorted_indices] = sort(fitness_values, 'descend');
    for i = 1:ga_params.elitism_count
        new_population(i, :) = population(sorted_indices(i), :);
    end
    
    % Generate rest of population through crossover and mutation
    for i = (ga_params.elitism_count + 1):pop_size
        % Tournament selection
        parent1 = tournament_selection(population, fitness_values, ga_params.tournament_size);
        parent2 = tournament_selection(population, fitness_values, ga_params.tournament_size);
        
        % Crossover
        if rand() < ga_params.crossover_rate
            [child1, child2] = single_point_crossover(parent1, parent2);
            child = child1; % Use first child
        else
            child = parent1; % No crossover
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
    tournament_fitness = fitness_values(tournament_indices);
    [~, winner_idx] = max(tournament_fitness);
    parent = population(tournament_indices(winner_idx), :);
end

function [child1, child2] = single_point_crossover(parent1, parent2)
    chromosome_length = length(parent1);
    crossover_point = randi(chromosome_length - 1);
    
    child1 = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
    child2 = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
end

function mutated_chromosome = mutate_chromosome(chromosome, param_bounds)
    mutated_chromosome = chromosome;
    chromosome_length = length(chromosome);
    
    % Gaussian mutation with boundary constraints
    for i = 1:chromosome_length
        if rand() < 0.1 % Per-gene mutation probability
            sigma = (param_bounds(i, 2) - param_bounds(i, 1)) * 0.1; % 10% of range
            mutation = normrnd(0, sigma);
            mutated_chromosome(i) = mutated_chromosome(i) + mutation;
            
            % Enforce bounds
            mutated_chromosome(i) = max(param_bounds(i, 1), ...
                                       min(param_bounds(i, 2), mutated_chromosome(i)));
        end
    end
end

function optimized_fis = create_optimized_fis(chromosome, base_fis, param_bounds)
    optimized_fis = base_fis;
    
    % Update input membership functions
    % Temperature (3 MFs)
    optimized_fis.Inputs(1).MembershipFunctions(1).Parameters(2) = chromosome(1); % Cold center
    optimized_fis.Inputs(1).MembershipFunctions(2).Parameters(2) = chromosome(2); % Comfortable center
    optimized_fis.Inputs(1).MembershipFunctions(3).Parameters(2) = chromosome(3); % Hot center
    
    % Light Level (3 MFs)
    optimized_fis.Inputs(2).MembershipFunctions(1).Parameters(2) = chromosome(4); % Dark center
    optimized_fis.Inputs(2).MembershipFunctions(2).Parameters(2) = chromosome(5); % Dim center
    optimized_fis.Inputs(2).MembershipFunctions(3).Parameters(2) = chromosome(6); % Bright center
    
    % Time of Day (3 MFs)
    optimized_fis.Inputs(3).MembershipFunctions(1).Parameters(2) = chromosome(7); % Night center
    optimized_fis.Inputs(3).MembershipFunctions(2).Parameters(2) = chromosome(8); % Day center
    optimized_fis.Inputs(3).MembershipFunctions(3).Parameters(2) = chromosome(9); % Evening center
    
    % Activity Level (3 MFs)
    optimized_fis.Inputs(4).MembershipFunctions(1).Parameters(2) = chromosome(10); % Resting center
    optimized_fis.Inputs(4).MembershipFunctions(2).Parameters(2) = chromosome(11); % Moderate center
    optimized_fis.Inputs(4).MembershipFunctions(3).Parameters(2) = chromosome(12); % Active center
    
    % User Preference (3 MFs)
    optimized_fis.Inputs(5).MembershipFunctions(1).Parameters(2) = chromosome(13); % Cool center
    optimized_fis.Inputs(5).MembershipFunctions(2).Parameters(2) = chromosome(14); % Neutral center
    optimized_fis.Inputs(5).MembershipFunctions(3).Parameters(2) = chromosome(15); % Warm center
    
    % Update output membership functions
    % HVAC Control (5 MFs)
    optimized_fis.Outputs(1).MembershipFunctions(1).Parameters(2) = chromosome(16); % StrongCooling center
    optimized_fis.Outputs(1).MembershipFunctions(2).Parameters(2) = chromosome(17); % LightCooling center
    optimized_fis.Outputs(1).MembershipFunctions(3).Parameters(2) = chromosome(18); % Off center
    optimized_fis.Outputs(1).MembershipFunctions(4).Parameters(2) = chromosome(19); % LightHeating center
    optimized_fis.Outputs(1).MembershipFunctions(5).Parameters(2) = chromosome(20); % StrongHeating center
    
    % Lighting Control (4 MFs)
    optimized_fis.Outputs(2).MembershipFunctions(1).Parameters(2) = chromosome(21); % Off center
    optimized_fis.Outputs(2).MembershipFunctions(2).Parameters(2) = chromosome(22); % Dim center
    optimized_fis.Outputs(2).MembershipFunctions(3).Parameters(2) = chromosome(23); % Medium center
    optimized_fis.Outputs(2).MembershipFunctions(4).Parameters(2) = chromosome(24); % Bright center
    
    % Blind Position (3 MFs)
    optimized_fis.Outputs(3).MembershipFunctions(1).Parameters(2) = chromosome(25); % Closed center
    optimized_fis.Outputs(3).MembershipFunctions(2).Parameters(2) = chromosome(26); % PartiallyOpen center
    optimized_fis.Outputs(3).MembershipFunctions(3).Parameters(2) = chromosome(27); % FullyOpen center
end
