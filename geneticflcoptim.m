%% Genetic Algorithm Optimization of Smart Home FLC
% This script optimizes the membership function parameters of the FLC
% using genetic algorithm to minimize control error and energy consumption
%
% Author: Priyanka Rai
% Course: Evolutionary and Fuzzy Systems

clear all; close all; clc;

%% Load Base FIS and Generate Training Data
addpath('/Users/priyankarai/Desktop/Msc/AML_Task2/Part1_FLC_Design');

try
    base_fis = readfis('/Users/priyankarai/Desktop/Msc/AML_Task2/Part1_FLC_Design/smart_home_flc.fis');
    fprintf('Base FIS loaded successfully!\n');
catch
    fprintf('Base FIS not found. Creating it first...\n');
    run('/Users/priyankarai/Desktop/Msc/AML_Task2/Part1_FLC_Design/smart_home_flc.m');
    base_fis = readfis('/Users/priyankarai/Desktop/Msc/AML_Task2/Part1_FLC_Design/smart_home_flc.fis');
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

%% Chromosome / Bounds (unchanged)
chromosome_length = 27;
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
    % HVAC Control MF centers [StrongCooling, LightCooling, Off, LightHeating, StrongHeating]
    -80, -60; -25, -5; -5, 5; 5, 25; 60, 80;
    % Lighting Control MF centers [Off, Dim, Medium, Bright]
    2, 10; 20, 40; 55, 75; 90, 98;
    % Blind Position MF centers [Closed, PartiallyOpen, FullyOpen]
    5, 15; 30, 50; 75, 90
];

%% GA Parameters
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

%% Fitness Function (uses safe eval + MF validity enforcement)
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
    parfor i = 1:ga_params.pop_size   % use parfor if you have Parallel Toolbox, else change to for
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
% Save inside your project folder
writeFIS(optimized_fis, '/Users/priyankarai/Desktop/Msc/AML_Task2/Part1_FLC_Design/optimized_smart_home_flc.fis');
fprintf('Optimized FIS saved as optimized_smart_home_flc.fis\n');

%% Performance Comparison (safe evaluation)
fprintf('\n=== Performance Comparison ===\n');

original_outputs = zeros(size(desired_outputs));
optimized_outputs = zeros(size(desired_outputs));
for i = 1:size(training_inputs, 1)
    original_outputs(i, :) = safe_evalfis(base_fis, training_inputs(i, :));
    optimized_outputs(i, :) = safe_evalfis(optimized_fis, training_inputs(i, :));
end

% Calculate errors
original_error = mean(sqrt(sum((original_outputs - desired_outputs).^2, 2)));
optimized_error = mean(sqrt(sum((optimized_outputs - desired_outputs).^2, 2)));
fprintf('Original FIS Error (RMSE): %.4f\n', original_error);
fprintf('Optimized FIS Error (RMSE): %.4f\n', optimized_error);
fprintf('Improvement: %.2f%%\n', (original_error - optimized_error) / original_error * 100);

%% Visualization
figure('Position', [100, 100, 800, 600]);

subplot(2,2,1);
plot(1:ga_params.max_generations, best_fitness_history, 'b-', 'LineWidth', 2); hold on;
plot(1:ga_params.max_generations, mean_fitness_history, 'r--', 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Fitness'); title('GA Convergence');
legend('Best Fitness', 'Mean Fitness', 'Location', 'best'); grid on;

subplot(2,2,2);
scatter(desired_outputs(:,1), original_outputs(:,1), 'b', 'filled'); hold on;
scatter(desired_outputs(:,1), optimized_outputs(:,1), 'r', 'filled');
plot([-100, 100], [-100, 100], 'k--');
xlabel('Desired HVAC Control (%)'); ylabel('Actual HVAC Control (%)');
title('HVAC Control Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

subplot(2,2,3);
scatter(desired_outputs(:,2), original_outputs(:,2), 'b', 'filled'); hold on;
scatter(desired_outputs(:,2), optimized_outputs(:,2), 'r', 'filled');
plot([0, 100], [0, 100], 'k--');
xlabel('Desired Lighting Control (%)'); ylabel('Actual Lighting Control (%)');
title('Lighting Control Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

subplot(2,2,4);
scatter(desired_outputs(:,3), original_outputs(:,3), 'b', 'filled'); hold on;
scatter(desired_outputs(:,3), optimized_outputs(:,3), 'r', 'filled');
plot([0, 100], [0, 100], 'k--');
xlabel('Desired Blind Position (%)'); ylabel('Actual Blind Position (%)');
title('Blind Position Comparison'); legend('Original', 'Optimized', 'Perfect', 'Location', 'best'); grid on;

sgtitle('GA Optimization Results', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'ga_optimization_results.png');

fprintf('\nOptimization completed! Results saved.\n');

%% =================== Supporting Functions ===================

function population = initialize_population(pop_size, chromosome_length, param_bounds)
    population = zeros(pop_size, chromosome_length);
    for i = 1:pop_size
        for j = 1:chromosome_length
            min_val = param_bounds(j, 1);
            max_val = param_bounds(j, 2);
            % init near mid to reduce rule-drop at start
            population(i, j) = 0.5*(min_val + max_val) + 0.25*(max_val-min_val)*(2*rand()-1);
            % clamp
            population(i, j) = max(min_val, min(max_val, population(i, j)));
        end
    end
end

function fitness = evaluate_fitness(chromosome, inputs, desired_outputs, base_fis, param_bounds)
    try
        % Create FIS with modified parameters (enforce valid MFs + safe defuzz)
        modified_fis = create_optimized_fis(chromosome, base_fis, param_bounds);

        % Evaluate on all training samples with safe evaluator
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

        % Add penalty for invalid ordering among centers (kept for extra safety)
        penalty = calculate_mf_penalty(chromosome, param_bounds);
        fitness = fitness - penalty;

    catch
        % Severe penalty for invalid FIS or runtime errors
        fitness = -1e3;
    end
end

function y = safe_evalfis(fis, xrow)
    % Robust wrapper around evalfis:
    % - forces finite output
    % - handles empty/NaN by using mid-range fallback
    try
        y = evalfis(fis, xrow);
        if isempty(y) || any(~isfinite(y))
            y = output_mid_ranges(fis);
        end
    catch
        y = output_mid_ranges(fis);
    end
end

function mid = output_mid_ranges(fis)
    nout = numel(fis.Outputs);
    mid = zeros(1, nout);
    for k = 1:nout
        r = fis.Outputs(k).Range;
        mid(k) = mean(r);
    end
end

function penalty = calculate_mf_penalty(chromosome, ~)
    % Keep your original group-order penalty to discourage crossing
    penalty = 0;
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
        v = chromosome(group{1});
        if any(diff(v) <= 0)
            penalty = penalty + 50;
        end
    end
end

function new_population = evolve_population(population, fitness_values, ga_params, param_bounds)
    [pop_size, chromosome_length] = size(population);
    new_population = zeros(pop_size, chromosome_length);

    % Elitism
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
            [child1, child2] = single_point_crossover(parent1, parent2);
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
    t_idx = randperm(pop_size, tournament_size);
    [~, w] = max(fitness_values(t_idx));
    parent = population(t_idx(w), :);
end

function [child1, child2] = single_point_crossover(parent1, parent2)
    L = length(parent1);
    cp = randi(L - 1);
    child1 = [parent1(1:cp), parent2(cp+1:end)];
    child2 = [parent2(1:cp), parent1(cp+1:end)];
end

function mutated = mutate_chromosome(chromosome, param_bounds)
    mutated = chromosome;
    L = length(chromosome);
    for i = 1:L
        if rand() < 0.1
            sigma = 0.1 * (param_bounds(i, 2) - param_bounds(i, 1));
            mutated(i) = mutated(i) + normrnd(0, sigma);
            mutated(i) = max(param_bounds(i,1), min(param_bounds(i,2), mutated(i)));
        end
    end
end

function optimized_fis = create_optimized_fis(chromosome, base_fis, param_bounds)
    % Copy base and use safer defuzz during GA to avoid centroid-on-empty warnings
    optimized_fis = base_fis;
    try
        optimized_fis.DefuzzificationMethod = 'mom'; % safer than 'centroid' for sparse activation
    catch
        % older releases store per-output; ignore if not available
    end

    % Helper to clamp center b between its MF's a and c
    function set_center(fis_io, io_idx, mf_idx, new_center)
        params = fis_io(io_idx).MembershipFunctions(mf_idx).Parameters;
        % Assume triangular/trapezoidal with center at index 2 (as in your code)
        % Get left/right support; if trapezoid [a b c d], center ~ (b+c)/2 – we still clamp into [a,c] or [b,c]
        a = params(1);
        c = params(end-1); % works for tri [a b c] and trap [a b c d] (c is end-1)
        epsv = 1e-3 * max(1, abs(c - a));
        % Clamp to (a+eps, c-eps)
        b = max(a + epsv, min(c - epsv, new_center));
        params(2) = b;
        fis_io(io_idx).MembershipFunctions(mf_idx).Parameters = params;
    end

    % Update input membership functions (centers)
    set_center(optimized_fis.Inputs, 1, 1, clamp(chromosome(1),  param_bounds(1,:)));
    set_center(optimized_fis.Inputs, 1, 2, clamp(chromosome(2),  param_bounds(2,:)));
    set_center(optimized_fis.Inputs, 1, 3, clamp(chromosome(3),  param_bounds(3,:)));

    set_center(optimized_fis.Inputs, 2, 1, clamp(chromosome(4),  param_bounds(4,:)));
    set_center(optimized_fis.Inputs, 2, 2, clamp(chromosome(5),  param_bounds(5,:)));
    set_center(optimized_fis.Inputs, 2, 3, clamp(chromosome(6),  param_bounds(6,:)));

    set_center(optimized_fis.Inputs, 3, 1, clamp(chromosome(7),  param_bounds(7,:)));
    set_center(optimized_fis.Inputs, 3, 2, clamp(chromosome(8),  param_bounds(8,:)));
    set_center(optimized_fis.Inputs, 3, 3, clamp(chromosome(9),  param_bounds(9,:)));

    set_center(optimized_fis.Inputs, 4, 1, clamp(chromosome(10), param_bounds(10,:)));
    set_center(optimized_fis.Inputs, 4, 2, clamp(chromosome(11), param_bounds(11,:)));
    set_center(optimized_fis.Inputs, 4, 3, clamp(chromosome(12), param_bounds(12,:)));

    set_center(optimized_fis.Inputs, 5, 1, clamp(chromosome(13), param_bounds(13,:)));
    set_center(optimized_fis.Inputs, 5, 2, clamp(chromosome(14), param_bounds(14,:)));
    set_center(optimized_fis.Inputs, 5, 3, clamp(chromosome(15), param_bounds(15,:)));

    % Outputs
    set_center(optimized_fis.Outputs, 1, 1, clamp(chromosome(16), param_bounds(16,:)));
    set_center(optimized_fis.Outputs, 1, 2, clamp(chromosome(17), param_bounds(17,:)));
    set_center(optimized_fis.Outputs, 1, 3, clamp(chromosome(18), param_bounds(18,:)));
    set_center(optimized_fis.Outputs, 1, 4, clamp(chromosome(19), param_bounds(19,:)));
    set_center(optimized_fis.Outputs, 1, 5, clamp(chromosome(20), param_bounds(20,:)));

    set_center(optimized_fis.Outputs, 2, 1, clamp(chromosome(21), param_bounds(21,:)));
    set_center(optimized_fis.Outputs, 2, 2, clamp(chromosome(22), param_bounds(22,:)));
    set_center(optimized_fis.Outputs, 2, 3, clamp(chromosome(23), param_bounds(23,:)));
    set_center(optimized_fis.Outputs, 2, 4, clamp(chromosome(24), param_bounds(24,:)));

    set_center(optimized_fis.Outputs, 3, 1, clamp(chromosome(25), param_bounds(25,:)));
    set_center(optimized_fis.Outputs, 3, 2, clamp(chromosome(26), param_bounds(26,:)));
    set_center(optimized_fis.Outputs, 3, 3, clamp(chromosome(27), param_bounds(27,:)));
end

function v = clamp(x, bounds)
    v = max(bounds(1), min(bounds(2), x));
end
