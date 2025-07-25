%% --- 1. SETUP ---
clear; clc; close all;

% --- Data Path ---
dataPath = '..\..\data\01_raw_data\self_assessment\SAM'; 
% Define file matching pattern
filePattern = fullfile(dataPath, 'SAM_P*.csv');
% Define scale labels for results output
dimLabels = ["Valence", "Arousal", "Dominance"];
numScales = length(dimLabels);

%% Define metadata file path and correlation labels
metadata_file = '..\..\data\01_raw_data\self_assessment\Participants_metadata.csv';
corrLabels = ["Valence", "Arousal", "Dominance", "Presentation Order"];

%% --- 2. LOAD AND AGGREGATE DATA ---
fprintf('--- Loading Data ---\n');

%% Load metadata file
if ~isfile(metadata_file)
    error('Metadata file not found at: %s. Cannot perform order correlation analysis.', metadata_file);
else
    % Load metadata, specifying that data starts from the second row
    opts = detectImportOptions(metadata_file);
    opts.DataLines = [2, Inf]; 
    % Name the variables for easier access
    opts.VariableNames = {'ParticipantID', 'Age', 'Gender', 'Handedness', 'Vision', 'ClipOrder', 'MusicExp', 'Hearing', 'Comments'};
    metadata = readtable(metadata_file, opts);
    fprintf('Successfully loaded participant metadata.\n');
end

% Find all files matching the pattern
fileList = dir(filePattern);

% Check if any files were found
if isempty(fileList)
    error('No SAM files found at the specified path. Please check `dataPath` and `filePattern`.');
end

numParticipants = length(fileList);
fprintf('Found rating files for %d participants.\n', numParticipants);

% Read data from the first file to determine the number of clips
try
    firstFileData = readmatrix(fullfile(dataPath, fileList(1).name), 'HeaderLines', 1);
    numVideos = size(firstFileData, 1);
    fprintf('Detected %d clips watched by each participant.\n', numVideos);
catch ME
    error('Error reading the first file: %s\nPlease ensure the file format is correct.', ME.message);
end

% Pre-allocate a 3D matrix to store all data
all_data = nan(numVideos, numScales, numParticipants);
% Pre-allocate a matrix to store presentation order
% all_orders(clip_id, participant_index) will store the presentation position (1st, 2nd, etc.)
all_orders = nan(numVideos, numParticipants);

% Loop through and read each participant's file
for i = 1:numParticipants
    fileName = fullfile(dataPath, fileList(i).name);
    try
        % --- Load Rating Data ---
        participantData = readmatrix(fileName, 'HeaderLines', 1);
        if size(participantData, 1) ~= numVideos
            warning('File %s has a different number of clips (%d) than expected (%d). Skipping this participant.', ...
                    fileList(i).name, size(participantData, 1), numVideos);
            continue;
        end
        all_data(:, :, i) = participantData(:, 2:4);
        
        % Process and store presentation order for this participant
        % Extract participant ID (e.g., 'P05') from the filename
        [~, fname, ~] = fileparts(fileName);
        p_id = strrep(fname, 'SAM_', ''); % Assumes format is SAM_PXX
        
        % Find this participant in the metadata
        meta_row_idx = find(strcmp(metadata.ParticipantID, p_id));
        
        if isempty(meta_row_idx)
            warning('Participant %s found in SAM data but not in metadata. Skipping order analysis for this participant.', p_id);
            continue; % all_orders for this participant will remain NaN
        end
        
        % Get the order string (e.g., '17-16-...')
        order_str = metadata.ClipOrder{meta_row_idx};
        % Split the string and convert to numbers
        clip_order_this_p = str2double(strsplit(order_str, '-'));
        
        % Create an inverse map: for each clip ID, find its position in the sequence
        order_map_this_p = nan(numVideos, 1);
        for pos = 1:length(clip_order_this_p)
            clip_id = clip_order_this_p(pos);
            if clip_id > 0 && clip_id <= numVideos % Safety check
                order_map_this_p(clip_id) = pos;
            end
        end
        all_orders(:, i) = order_map_this_p;
        
    catch ME
        warning('Error reading file %s: %s. Skipping this participant.', fileName, ME.message);
    end
end

% Remove participant data that is all NaN (due to read failures, etc.)
nan_participants_mask = all(all(isnan(all_data),1),2);
all_data(:,:,nan_participants_mask) = [];
all_orders(:, nan_participants_mask) = []; % Also remove from orders matrix

numParticipants = size(all_data, 3); % Update the number of valid participants
fprintf('Successfully loaded data for %d valid participants.\n', numParticipants);


%% --- 3. CALCULATE STATISTICS ---
% This section performs all necessary calculations. Results will be reported collectively in the next section.

% --- 3.1. Statistics per Clip ---
video_means = nan(numVideos, numScales);
video_sds   = nan(numVideos, numScales);
for v = 1:numVideos
    ratingsForThisVideo = squeeze(all_data(v, :, :));
    video_means(v, :) = mean(ratingsForThisVideo, 2, 'omitnan')';
    video_sds(v, :)   = std(ratingsForThisVideo, 0, 2, 'omitnan')';
end

% Calculate the Coefficient of Variation (CV) for each clip
video_cvs = video_sds ./ video_means;
video_cvs(video_means == 0) = NaN; % If the mean is 0, CV is meaningless

% --- 3.2. Overall Statistics Across All Clips ---
mean_cvs = mean(video_cvs, 1, 'omitnan');
std_cvs  = std(video_cvs, 0, 1, 'omitnan'); 

% --- 3.3. Inter-scale Correlation Calculation ---
% Reshape and combine ratings with presentation order
total_num_ratings = numVideos * numParticipants;
% Reshape the 3D rating data into a 2D matrix
all_ratings_2d = reshape(all_data, total_num_ratings, numScales);
% Reshape the 2D order data into a 1D vector (column)
all_orders_1d = reshape(all_orders, total_num_ratings, 1);
% Combine into a single matrix for correlation
all_data_for_corr = [all_ratings_2d, all_orders_1d];
% Remove any rows with NaN values (e.g., from participants skipped in metadata)
all_data_for_corr(any(isnan(all_data_for_corr), 2), :) = [];
% Calculate the 4x4 correlation matrix
[R_interscale, P_interscale] = corr(all_data_for_corr);

%% --- 4. REPORT RESULTS ---
% This section outputs all calculated results to the command window in a clear format.
fprintf('\n\n=================================================================\n');
fprintf('                  Data Consistency and Statistical Analysis Report\n');
fprintf('=================================================================\n');

% --- 4.1. Overall Data Consistency (Based on Coefficient of Variation - CV) ---
fprintf('\n--- Overall Data Consistency (Based on Coefficient of Variation - CV) ---\n');
fprintf('Mean CV (Mean ± SD):\n');
fprintf('%-10s: %.2f ± %.2f\n', dimLabels(1), mean_cvs(1), std_cvs(1)); % Valence
fprintf('%-10s: %.2f ± %.2f\n', dimLabels(2), mean_cvs(2), std_cvs(2)); % Arousal
fprintf('%-10s: %.2f ± %.2f\n', dimLabels(3), mean_cvs(3), std_cvs(3)); % Dominance

% --- 4.2. Consistency Coefficient (CV) per Clip ---
fprintf('\n--- Consistency Coefficient (CV) per Clip ---\n');
consistency_table = table((1:numVideos)', ...
                      video_cvs(:,1), video_cvs(:,2), video_cvs(:,3), ...
                      'VariableNames', {'Clip ID', 'CV_Valence', 'CV_Arousal', 'CV_Dominance'});
disp(consistency_table);

% --- 4.3. Mean Rating per Clip ---
fprintf('\n--- Mean Rating per Clip (Mean ± SD) ---\n');
% Create formatted strings for mean ± std
valence_str   = compose('%.2f ± %.2f', video_means(:,1), video_sds(:,1));
arousal_str   = compose('%.2f ± %.2f', video_means(:,2), video_sds(:,2));
dominance_str = compose('%.2f ± %.2f', video_means(:,3), video_sds(:,3));
% Create and display the table with the formatted strings
mean_ratings_table = table((1:numVideos)', valence_str, arousal_str, dominance_str, ...
                      'VariableNames', {'Clip ID', 'Valence', 'Arousal', 'Dominance'});
disp(mean_ratings_table);

% --- 4.4. Correlation with Reference Study ---
fprintf('\n--- Correlation with Reference Study (Gabert-Quillen et al.) ---\n');
ref_file_path = '..\..\misc\Ratings of Gabert-Quillen et al. study.csv';
if ~isfile(ref_file_path)
    fprintf('Reference file not found. Skipping correlation analysis.\n');
else
    try
        ref_data = readmatrix(ref_file_path, 'HeaderLines', 1);
        ref_video_ids = ref_data(:, 1);
        ref_valence   = ref_data(:, 2);
        ref_arousal   = ref_data(:, 3);
        
        [common_videos_mask, my_data_indices] = ismember(ref_video_ids, (1:numVideos)');
        
        ref_valence_common = ref_valence(common_videos_mask);
        ref_arousal_common = ref_arousal(common_videos_mask);
        my_valence_common  = video_means(my_data_indices(common_videos_mask), 1);
        my_arousal_common  = video_means(my_data_indices(common_videos_mask), 2);
        
        [rho_v, p_v] = corr(my_valence_common, ref_valence_common, 'Type', 'Spearman');
        [rho_a, p_a] = corr(my_arousal_common, ref_arousal_common, 'Type', 'Spearman');
        p_threshold = 0.001;
        if p_v < p_threshold, p_v_str = sprintf('< %.3f', p_threshold); else, p_v_str = sprintf('  %.3f', p_v); end
        if p_a < p_threshold, p_a_str = sprintf('< %.3f', p_threshold); else, p_a_str = sprintf('  %.3f', p_a); end
        fprintf('Spearman''s Correlation Results:\n');
        fprintf('----------------------------------------\n');
        fprintf('Scale      Rho         p-value\n');
        fprintf('----------------------------------------\n');
        fprintf('Valence      %8.3f      %s\n', rho_v, p_v_str);
        fprintf('Arousal      %8.3f      %s\n', rho_a, p_a_str);
        fprintf('----------------------------------------\n');
    catch ME
        fprintf('Error during correlation analysis: %s\n', ME.message);
    end
end

% --- Correlation Between Emotion Scales ---
% Update the correlation reporting section
fprintf('\n--- Correlation Between Emotion Scales and Presentation Order ---\n');
fprintf('Pearson correlations (r) across all individual ratings:\n');
fprintf('------------------------------------------------------------\n');
fprintf('%-30s   r-value   Significance\n', 'Comparison');
fprintf('------------------------------------------------------------\n');
% Define all pairs for correlation, now including the 4th variable (Order)
pairs = {
    'Valence vs. Arousal',        [1, 2]; 
    'Valence vs. Dominance',      [1, 3]; 
    'Arousal vs. Dominance',      [2, 3];
    'Valence vs. Pres. Order',    [1, 4];
    'Arousal vs. Pres. Order',    [2, 4];
    'Dominance vs. Pres. Order',  [3, 4]
};
for i = 1:size(pairs, 1)
    comparison_name = pairs{i, 1};
    idx = pairs{i, 2};
    r_value = R_interscale(idx(1), idx(2));
    p_value = P_interscale(idx(1), idx(2));
    
    stars = '';
    if p_value < 0.001, stars = '***'; elseif p_value < 0.01, stars = '**'; elseif p_value < 0.05, stars = '*'; end
    fprintf('%-30s   %7.3f    %-5s\n', comparison_name, r_value, stars);
end
fprintf('------------------------------------------------------------\n');
fprintf('Significance levels: * p < .05, ** p < .01, *** p < .001\n');

%% --- 5. VISUALIZE RESULTS ---
fprintf('\n\n--- Generating 2D Affective Space Plot ---\n');

% --- Define Plotting Parameters ---
midpoint = 5; 
axis_limits = [1, 9];
marker_color      = [0.2, 0.6, 0.8];
line_color        = [0.5, 0.5, 0.5];
quad_label_color  = [0.5, 0.5, 0.5];
marker_size       = 90;
font_size_axis    = 16;
font_size_title   = 18;
font_size_quad    = 22;
padding           = 0.3;

% --- Plotting ---
figure('Name', 'Valence-Arousal Space Distribution', 'Color', 'white', ...
       'Position', [10, 10, 700, 680]);
ax = gca; 
hold on;

% 1. Draw quadrant dividing lines
xline(ax, midpoint, '--', 'Color', line_color, 'LineWidth', 1.5);
yline(ax, midpoint, '--', 'Color', line_color, 'LineWidth', 1.5);

% 2. Draw scatter plot (using mean ratings for each clip)
scatter(ax, video_means(:, 1), video_means(:, 2), marker_size, ...
    'filled', 'MarkerFaceColor', marker_color, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1.5);

% 3. Add quadrant labels
text(ax, axis_limits(1) + padding, axis_limits(2) - padding, 'HALV', 'FontSize', font_size_quad, 'FontWeight', 'bold', 'Color', quad_label_color, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
text(ax, axis_limits(2) - padding, axis_limits(2) - padding, 'HAHV', 'FontSize', font_size_quad, 'FontWeight', 'bold', 'Color', quad_label_color, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
text(ax, axis_limits(1) + padding, axis_limits(1) + padding, 'LALV', 'FontSize', font_size_quad, 'FontWeight', 'bold', 'Color', quad_label_color, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
text(ax, axis_limits(2) - padding, axis_limits(1) + padding, 'LAHV', 'FontSize', font_size_quad, 'FontWeight', 'bold', 'Color', quad_label_color, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');

% 4. Parameters
title('Distribution of Clips in Valence-Arousal Space', 'FontSize', font_size_title, 'FontWeight', 'normal');
xlabel('Valence', 'FontSize', font_size_axis);
ylabel('Arousal', 'FontSize', font_size_axis);
xlim(axis_limits);
ylim(axis_limits);
axis square;
grid on;
ax.GridAlpha = 0.1;
box on;
set(ax, 'FontSize', font_size_axis, 'LineWidth', 1, 'Layer', 'top');
hold off;

fprintf('Plotting complete.\n');

