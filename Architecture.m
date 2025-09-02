%% Figure 1: FLC-Based Assistive Care Environment
figure('Position', [100, 100, 800, 500], 'Color', 'white');
axis off;

% Draw blocks
rectangle('Position', [0.1, 0.65, 0.2, 0.2], 'Curvature', [0.2, 0.2], 'FaceColor', 'y', 'EdgeColor', 'k');
text(0.2, 0.75, 'Sensors\n(Temp, Light,\nTime, Activity)', 'HorizontalAlignment', 'center', 'FontSize', 12);

rectangle('Position', [0.4, 0.65, 0.2, 0.2], 'Curvature', [0.2, 0.2], 'FaceColor', 'c', 'EdgeColor', 'k');
text(0.5, 0.75, 'Fuzzy Logic\nController\n(Mamdani FLC)', 'HorizontalAlignment', 'center', 'FontSize', 12);

rectangle('Position', [0.7, 0.65, 0.2, 0.2], 'Curvature', [0.2, 0.2], 'FaceColor', 'g', 'EdgeColor', 'k');
text(0.8, 0.75, 'Actuators\n(HVAC, Lights,\nBlinds)', 'HorizontalAlignment', 'center', 'FontSize', 12);

% Arrows
annotation('arrow', [0.3, 0.4], [0.75, 0.75]);
annotation('arrow', [0.6, 0.7], [0.75, 0.75]);

% User input
annotation('textbox', [0.45, 0.45, 0.1, 0.1], 'String', 'User\nPreference', 'FontSize', 10, 'HorizontalAlignment', 'center');
annotation('arrow', [0.5, 0.5], [0.55, 0.65]);

title('Figure 1: Smart Home FLC Architecture for Ambient Assisted Living', 'FontSize', 14, 'Interpreter', 'none');