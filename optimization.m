%% Optimization Algorithms Comparison on CEC'2005 Functions
% This script compares Genetic Algorithm (GA), Particle Swarm Optimization (PSO),
% and Simulated Annealing (SA) on CEC'2005 benchmark functions F1 and F6
% for dimensions D=2 and D=10

clear all; close all; clc;

%% Configuration
config = struct();
config.functions = {'F1', 'F6'};
config.function_names = {'Shifted Sphere', 'Shifted Rosenbrock'};
config.dimensions = [2, 10];
config.num_runs = 15;
config.max_evaluations = 10000; % Budget per run
config.search_range = [-100, 100];

% Algorithm-specific parameters
config.ga_params = struct('PopulationSize', 50, 'CrossoverFraction', 0.8);
config.pso_params = struct('SwarmSize', 50, 'SelfAdjustmentWeight', 1.49, ...
                          'SocialAdjustmentWeight', 1.49, 'InertiaRange', [0.1, 1.1]);
config.sa_params = struct('InitialTemperature', 100);

fprintf('=== CEC''2005 Optimization Algorithms Comparison ===\n');
fprintf('Functions: %s\n', strjoin(config.function_names, ', '));
fprintf('Dimensions: %s\n', mat2str(config.dimensions));
fprintf('Runs per algorithm: %d\n', config.num_runs);
fprintf('Maximum evaluations: %d\n', config.max_evaluations);

%% Initialize Results Storage
results = struct();
for func_idx = 1:length(config.functions)
    func_name = config.functions{func_idx};
    results.(func_name) = struct();
    
    for dim = config.dimensions
        dim_str = sprintf('D%d', dim);
        results.(func_name).(dim_str) = struct();
        
        % Initialize arrays for each algorithm
        algorithms = {'GA', 'PSO', 'SA'};
        for alg_idx = 1:length(algorithms)
            alg = algorithms{alg_idx};
            results.(func_name).(dim_str).(alg) = struct();
            results.(func_name).(dim_str).(alg).best_fitness = zeros(config.num_runs, 1);
            results.(func_name).(dim_str).(alg).convergence = cell(config.num_runs, 1);
            results.(func_name).(dim_str).(alg).evaluations = zeros(config.num_runs, 1);
            results.(func_name).(dim_str).(alg).computation_time = zeros(config.num_runs, 1);
        end
    end
end

%% CEC2005 Function Implementations
% Global variables for tracking convergence
global convergence_data;
global current_run;
global current_algorithm;
global current_function;
global current_dimension;

%% Run Optimization Experiments
fprintf('\nStarting optimization experiments...\n');

for func_idx = 1:length(config.functions)
    func_name = config.functions{func_idx};
    fprintf('\n--- Function %s (%s) ---\n', func_name, config.function_names{func_idx});
    
    for dim_idx = 1:length(config.dimensions)
        dim = config.dimensions(dim_idx);
        dim_str = sprintf('D%d', dim);
        fprintf('\nDimension: %d\n', dim);
        
        % Define objective function with proper CEC2005 implementation
        if strcmp(func_name, 'F1')
            % F1: Shifted Sphere Function
            o1 = rand(1, dim) * 180 - 90; % Random shift vector
            obj_func = @(x) cec2005_f1(x, o1);
            global_optimum = -450;
        else % F6
            % F6: Shifted Rosenbrock Function  
            o6 = rand(1, dim) * 180 - 90; % Random shift vector
            obj_func = @(x) cec2005_f6(x, o6);
            global_optimum = 390;
        end
        
        % Set bounds
        lb = config.search_range(1) * ones(1, dim);
        ub = config.search_range(2) * ones(1, dim);
        
        %% Run Genetic Algorithm
        fprintf('  Running GA... ');
        current_algorithm = 'GA';
        current_function = func_name;
        current_dimension = dim_str;
        
        for run = 1:config.num_runs
            current_run = run;
            convergence_data = [];
            
            tic;
            
            % Create wrapper function for convergence tracking
            obj_func_tracked = @(x) track_convergence(obj_func(x));
            
            % GA options
            ga_options = optimoptions('ga', ...
                'PopulationSize', config.ga_params.PopulationSize, ...
                'CrossoverFraction', config.ga_params.CrossoverFraction, ...
                'MaxGenerations', ceil(config.max_evaluations / config.ga_params.PopulationSize), ...
                'FunctionTolerance', 1e-10, ...
                'Display', 'off', ...
                'PlotFcn', []);
            
            [x_best, f_best, exitflag, output] = ga(obj_func_tracked, dim, [], [], [], [], lb, ub, [], ga_options);
            
            results.(func_name).(dim_str).GA.best_fitness(run) = f_best;
            results.(func_name).(dim_str).GA.evaluations(run) = output.funccount;
            results.(func_name).(dim_str).GA.computation_time(run) = toc;
            results.(func_name).(dim_str).GA.convergence{run} = convergence_data;
        end
        fprintf('Completed\n');
        
        %% Run Particle Swarm Optimization
        fprintf('  Running PSO... ');
        current_algorithm = 'PSO';
        
        for run = 1:config.num_runs
            current_run = run;
            convergence_data = [];
            
            tic;
            
            % Create wrapper function for convergence tracking
            obj_func_tracked = @(x) track_convergence(obj_func(x));
            
            % PSO options
            pso_options = optimoptions('particleswarm', ...
                'SwarmSize', config.pso_params.SwarmSize, ...
                'SelfAdjustmentWeight', config.pso_params.SelfAdjustmentWeight, ...
                'SocialAdjustmentWeight', config.pso_params.SocialAdjustmentWeight, ...
                'InertiaRange', config.pso_params.InertiaRange, ...
                'MaxIterations', ceil(config.max_evaluations / config.pso_params.SwarmSize), ...
                'FunctionTolerance', 1e-10, ...
                'Display', 'off', ...
                'PlotFcn', []);
            
            [x_best, f_best, exitflag, output] = particleswarm(obj_func_tracked, dim, lb, ub, pso_options);
            
            results.(func_name).(dim_str).PSO.best_fitness(run) = f_best;
            results.(func_name).(dim_str).PSO.evaluations(run) = output.funccount;
            results.(func_name).(dim_str).PSO.computation_time(run) = toc;
            results.(func_name).(dim_str).PSO.convergence{run} = convergence_data;
        end
        fprintf('Completed\n');
        
        %% Run Simulated Annealing
        fprintf('  Running SA... ');
        current_algorithm = 'SA';
        
        for run = 1:config.num_runs
            current_run = run;
            convergence_data = [];
            
            tic;
            
            % Create wrapper function for convergence tracking
            obj_func_tracked = @(x) track_convergence(obj_func(x));
            
            % SA options
            sa_options = optimoptions('simulannealbnd', ...
                'InitialTemperature', config.sa_params.InitialTemperature, ...
                'MaxFunctionEvaluations', config.max_evaluations, ...
                'FunctionTolerance', 1e-10, ...
                'Display', 'off', ...
                'PlotFcn', []);
            
            % Random initial point
            x0 = lb + (ub - lb) .* rand(1, dim);
            [x_best, f_best, exitflag, output] = simulannealbnd(obj_func_tracked, x0, lb, ub, sa_options);
            
            results.(func_name).(dim_str).SA.best_fitness(run) = f_best;
            results.(func_name).(dim_str).SA.evaluations(run) = output.funccount;
            results.(func_name).(dim_str).SA.computation_time(run) = toc;
            results.(func_name).(dim_str).SA.convergence{run} = convergence_data;
        end
        fprintf('Completed\n');
    end
end

%% Statistical Analysis
fprintf('\n=== Statistical Analysis ===\n');

algorithms = {'GA', 'PSO', 'SA'};
summary_table = table();
row_idx = 1;

for func_idx = 1:length(config.functions)
    func_name = config.functions{func_idx};
    
    for dim_idx = 1:length(config.dimensions)
        dim = config.dimensions(dim_idx);
        dim_str = sprintf('D%d', dim);
        
        fprintf('\n%s - %s (D=%d):\n', func_name, config.function_names{func_idx}, dim);
        fprintf('Algorithm | Best      | Worst     | Mean      | Std Dev   | Success Rate\n');
        fprintf('----------|-----------|-----------|-----------|-----------|-------------\n');
        
        for alg_idx = 1:length(algorithms)
            alg = algorithms{alg_idx};
            fitness_values = results.(func_name).(dim_str).(alg).best_fitness;
            
            best_val = min(fitness_values);
            worst_val = max(fitness_values);
            mean_val = mean(fitness_values);
            std_val = std(fitness_values);
            
            % Success rate (within 1% of global optimum)
            if strcmp(func_name, 'F1')
                tolerance = abs(-450 * 0.01);
                success_count = sum(fitness_values <= (-450 + tolerance));
            else % F6
                tolerance = abs(390 * 0.01);
                success_count = sum(fitness_values <= (390 + tolerance));
            end
            success_rate = success_count / config.num_runs * 100;
            
            fprintf('%-9s | %9.2e | %9.2e | %9.2e | %9.2e | %10.1f%%\n', ...
                    alg, best_val, worst_val, mean_val, std_val, success_rate);
            
            % Store in summary table
            summary_table.Function{row_idx} = func_name;
            summary_table.Dimension(row_idx) = dim;
            summary_table.Algorithm{row_idx} = alg;
            summary_table.Best(row_idx) = best_val;
            summary_table.Worst(row_idx) = worst_val;
            summary_table.Mean(row_idx) = mean_val;
            summary_table.StdDev(row_idx) = std_val;
            summary_table.SuccessRate(row_idx) = success_rate;
            summary_table.AvgTime(row_idx) = mean(results.(func_name).(dim_str).(alg).computation_time);
            row_idx = row_idx + 1;
        end
    end
end

% Save summary table
writetable(summary_table, 'optimization_results_summary.csv');
fprintf('\nSummary table saved to optimization_results_summary.csv\n');

%% Generate Convergence Plots
fprintf('\nGenerating convergence plots...\n');

for func_idx = 1:length(config.functions)
    func_name = config.functions{func_idx};
    
    for dim_idx = 1:length(config.dimensions)
        dim = config.dimensions(dim_idx);
        dim_str = sprintf('D%d', dim);
        
        figure('Position', [100 + func_idx*50 + dim_idx*25, 100 + func_idx*50 + dim_idx*25, 800, 600]);
        
        colors = {'b', 'r', 'g'};
        line_styles = {'-', '--', ':'};
        
        hold on;
        
        for alg_idx = 1:length(algorithms)
            alg = algorithms{alg_idx};
            
            % Get convergence data (average over all runs)
            convergence_data_all = results.(func_name).(dim_str).(alg).convergence;
            
            % Find the run with median performance for plotting
            fitness_values = results.(func_name).(dim_str).(alg).best_fitness;
            [~, median_idx] = min(abs(fitness_values - median(fitness_values)));
            
            if ~isempty(convergence_data_all{median_idx})
                conv_data = convergence_data_all{median_idx};
                evaluations = 1:length(conv_data);
                
                plot(evaluations, conv_data, 'Color', colors{alg_idx}, ...
                     'LineStyle', line_styles{alg_idx}, 'LineWidth', 2, 'DisplayName', alg);
            end
        end
        
        xlabel('Function Evaluations');
        ylabel('Best Fitness Value');
        title(sprintf('%s - %s (D=%d) Convergence', func_name, config.function_names{func_idx}, dim));
        legend('Location', 'best');
        grid on;
        set(gca, 'YScale', 'log');
        
        % Save plot
        filename = sprintf('convergence_%s_D%d.png', func_name, dim);
        saveas(gcf, filename);
    end
end

%% Generate Box Plots for Statistical Comparison
fprintf('Generating box plots...\n');

for func_idx = 1:length(config.functions)
    func_name = config.functions{func_idx};
    
    figure('Position', [200 + func_idx*100, 200, 1000, 500]);
    
    subplot_idx = 1;
    for dim_idx = 1:length(config.dimensions)
        dim = config.dimensions(dim_idx);
        dim_str = sprintf('D%d', dim);
        
        subplot(1, 2, subplot_idx);
        
        % Collect data for box plot
        box_data = [];
        box_labels = {};
        
        for alg_idx = 1:length(algorithms)
            alg = algorithms{alg_idx};
            fitness_values = results.(func_name).(dim_str).(alg).best_fitness;
            
            box_data = [box_data; fitness_values];
            box_labels = [box_labels; repmat({alg}, config.num_runs, 1)];
        end
        
        boxplot(box_data, box_labels);
        title(sprintf('D=%d', dim));
        ylabel('Best Fitness Value');
        grid on;
        
        subplot_idx = subplot_idx + 1;
    end
    
    sgtitle(sprintf('%s - %s Statistical Distribution', func_name, config.function_names{func_idx}), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    filename = sprintf('boxplot_%s.png', func_name);
    saveas(gcf, filename);
end

%% Save Results
save('optimization_comparison_results.mat', 'results', 'config', 'summary_table');
fprintf('\nAll results saved to optimization_comparison_results.mat\n');

fprintf('\n=== Comparison Analysis Completed ===\n');
fprintf('Generated files:\n');
fprintf('- optimization_results_summary.csv\n');
fprintf('- convergence_*.png (convergence plots)\n');
fprintf('- boxplot_*.png (statistical distribution plots)\n');
fprintf('- optimization_comparison_results.mat\n');

%% CEC2005 Function Definitions

function f = cec2005_f1(x, o)
    % F1: Shifted Sphere Function
    % f(x) = sum((x_i - o_i)^2) - 450
    if size(x, 1) > 1
        f = zeros(size(x, 1), 1);
        for i = 1:size(x, 1)
            z = x(i, :) - o;
            f(i) = sum(z.^2) - 450;
        end
    else
        z = x - o;
        f = sum(z.^2) - 450;
    end
end

function f = cec2005_f6(x, o)
    % F6: Shifted Rosenbrock Function
    % f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2) + 390
    if size(x, 1) > 1
        f = zeros(size(x, 1), 1);
        for i = 1:size(x, 1)
            z = x(i, :) - o + 1; % Shift and offset
            rosenbrock_sum = 0;
            for j = 1:length(z)-1
                rosenbrock_sum = rosenbrock_sum + 100*(z(j+1) - z(j)^2)^2 + (z(j) - 1)^2;
            end
            f(i) = rosenbrock_sum + 390;
        end
    else
        z = x - o + 1; % Shift and offset
        rosenbrock_sum = 0;
        for j = 1:length(z)-1
            rosenbrock_sum = rosenbrock_sum + 100*(z(j+1) - z(j)^2)^2 + (z(j) - 1)^2;
        end
        f = rosenbrock_sum + 390;
    end
end

function f_val = track_convergence(f_val)
    % Function to track convergence during optimization
    global convergence_data;
    global current_run;
    global current_algorithm;
    global current_function;
    global current_dimension;
    global results;
    
    % Store the current best value
    if isempty(convergence_data)
        convergence_data = f_val;
    else
        convergence_data(end+1) = min(f_val, min(convergence_data));
    end
end
