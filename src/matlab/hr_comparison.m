clc;
clear;
close all;

% --- 1. Define Data Paths and Parameters ---
ppgBasePath = '..\..\data\02_processed_data\ppg';
mmWaveBasePath = '..\..\data\02_processed_data\mmwave';

% --- 2. Find All Participants and Their Clips Files ---
fprintf('Scanning data files to identify participants and clips...\n');

% Filename format: ppg_PXX_YY.csv (PXX is participant ID, YY is clip ID)
ppgFiles = dir(fullfile(ppgBasePath, 'ppg_P*_*.csv'));

if isempty(ppgFiles)
    error('No files with the format "ppg_P*_*.csv" were found in the specified PPG path "%s". Please check the path.', ppgBasePath);
end

% Use containers.Map to store the list of clip IDs for each participant
participantClipMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
for k = 1:length(ppgFiles)
    % Regular expression to match (P followed by digits) after 'ppg_' and (digits) after '_'
    tokens = regexp(ppgFiles(k).name, 'ppg_(P\d+)_(\d+)\.csv', 'tokens');
    if ~isempty(tokens)
        participantID = tokens{1}{1}; % e.g., 'P05'
        clipID   = tokens{1}{2};   % e.g., '09'
        
        % Add the clip ID to the list for the corresponding participant
        if isKey(participantClipMap, participantID)
            participantClipMap(participantID) = [participantClipMap(participantID), {clipID}];
        else
            participantClipMap(participantID) = {clipID};
        end
    end
end

% Get all unique participant IDs
participantList = keys(participantClipMap);
if isempty(participantList)
    error('Could not parse any participant IDs from filenames. Please check if the filename format is "ppg_PXX_YY.csv".');
end
fprintf('Found %d participants.\n', length(participantList));


% --- 3. Loop to Process Data for Each Participant and Clip ---
% Initialize a structure to store all data
all_participants_data = struct();

fprintf('Starting data processing...\n');

% Loop through each participant
for i = 1:length(participantList)
    participantName = participantList{i};
    uniqueClipIDs = unique(participantClipMap(participantName)); % Get and sort unique clip IDs for this participant

    fprintf('Processing data of participant: %s...\n', participantName);

    current_participant_data = struct(); % To store data for the current participant

    % Loop through each clip for this participant
    for j = 1:length(uniqueClipIDs)
        clipID = uniqueClipIDs{j};
        clipLabel = ['clip_' clipID]; % Used for struct field names, e.g., clip_09

        ppg_data_raw = [];
        mmWave_heartBeat_raw = [];

        % --- Read PPG Data ---
        ppgFileName = ['ppg_' participantName '_' clipID '.csv'];
        ppgFilePath = fullfile(ppgBasePath, ppgFileName);
        fs_ppg = 200; % PPG sampling rate

        if exist(ppgFilePath, 'file')
            try
                ppg_data_raw = readmatrix(ppgFilePath);
            catch ME
                fprintf('    Error reading PPG file %s: %s\n', ppgFilePath, ME.message);
            end
        end

        % --- Read mmWave Data ---
        mmWaveFileName = ['mmwave_' participantName '_' clipID '.csv'];
        mmWaveFilePath = fullfile(mmWaveBasePath, mmWaveFileName);
        fs_mmwave = 100; % mmWave sampling rate

        if exist(mmWaveFilePath, 'file')
            try
                temp_mmWave_data = readmatrix(mmWaveFilePath, 'HeaderLines', 1);
                if size(temp_mmWave_data, 2) >= 3
                    mmWave_heartBeat_raw = temp_mmWave_data(:, 3);
                end
            catch ME
                fprintf('    Error reading mmWave file %s: %s\n', mmWaveFilePath, ME.message);
            end
        end

        % Initialize HR and error results
        hr_ppg_bpm = NaN;
        hr_mmwave_bpm = NaN;
        hr_error_bpm = NaN;
        hr_abs_error_bpm = NaN;
        hr_rel_error_percent = NaN;
        
        % Define Heart Rate frequency range (Hz)
        HR_FREQ_MIN_HZ = 1.0;
        HR_FREQ_MAX_HZ = 1.8;

        % --- Calculate PPG Heart Rate ---
        if ~isempty(ppg_data_raw) && length(ppg_data_raw) > fs_ppg * 2 % At least 2 seconds of data
            hr_ppg_bpm = calculate_hr_from_signal(ppg_data_raw, fs_ppg, HR_FREQ_MIN_HZ, HR_FREQ_MAX_HZ);
        end

        % --- Calculate mmWave Heart Rate ---
        if ~isempty(mmWave_heartBeat_raw) && length(mmWave_heartBeat_raw) > fs_mmwave * 2 % At least 2 seconds of data
            hr_mmwave_bpm = calculate_hr_from_signal(mmWave_heartBeat_raw, fs_mmwave, HR_FREQ_MIN_HZ, HR_FREQ_MAX_HZ);
        end

        % --- Calculate Error ---
        if ~isnan(hr_ppg_bpm) && ~isnan(hr_mmwave_bpm)
            hr_error_bpm = hr_mmwave_bpm - hr_ppg_bpm;
            hr_abs_error_bpm = abs(hr_error_bpm);
            if hr_ppg_bpm ~= 0 % Avoid division by zero
                hr_rel_error_percent = (hr_abs_error_bpm / hr_ppg_bpm) * 100;
            end
        end

        % Store data and results in the structure for the current participant
        current_participant_data.(clipLabel).ppg_raw = ppg_data_raw;
        current_participant_data.(clipLabel).mmHeartBeat_raw = mmWave_heartBeat_raw;
        current_participant_data.(clipLabel).ppg_hr_bpm = hr_ppg_bpm;
        current_participant_data.(clipLabel).mmwave_hr_bpm = hr_mmwave_bpm;
        current_participant_data.(clipLabel).hr_error_bpm = hr_error_bpm;
        current_participant_data.(clipLabel).hr_abs_error_bpm = hr_abs_error_bpm;
        current_participant_data.(clipLabel).hr_rel_error_percent = hr_rel_error_percent;
    end

    % Store the current participant's data into the main structure
    if ~isempty(fieldnames(current_participant_data))
         all_participants_data.(participantName) = current_participant_data;
    end
end

fprintf('------------------------------------\n');
fprintf('All data processing is complete.\n');

% --- Helper Function: Calculate Heart Rate from Signal ---
function hr_bpm = calculate_hr_from_signal(signal_data, fs, f_min_hz, f_max_hz)
    % Ensure the signal is a column vector
    signal_data = signal_data(:);
    % Detrend
    signal_detrended = detrend(signal_data);
    % FFT
    L = length(signal_detrended);
    if L < fs * 1
        hr_bpm = NaN;
        return;
    end
    NFFT = 2^nextpow2(L);
    Y = fft(signal_detrended, NFFT) / L;
    f_axis = fs/2 * linspace(0, 1, NFFT/2+1);
    P = 2*abs(Y(1:NFFT/2+1));
    % Find the peak within the specified frequency range
    freq_indices_in_range = find(f_axis >= f_min_hz & f_axis <= f_max_hz);
    if isempty(freq_indices_in_range)
        hr_bpm = NaN;
        return;
    end
    P_in_range = P(freq_indices_in_range);
    f_axis_in_range = f_axis(freq_indices_in_range);
    [max_power, idx_peak] = max(P_in_range);
    if isempty(idx_peak) || max_power == 0
        hr_bpm = NaN;
        return;
    end
    dominant_freq_hz = f_axis_in_range(idx_peak);
    % Convert frequency to BPM
    hr_bpm = dominant_freq_hz * 60;
end


% =========================================================================
% ================= Heart Rate Error Evaluation and Reporting =============
% =========================================================================

fprintf('\n------------------------------------\n');
fprintf('Starting heart rate error evaluation...\n');
fprintf('------------------------------------\n');

participant_names_list = fieldnames(all_participants_data);
num_participants = length(participant_names_list);

% --- 1. Evaluate Heart Rate Error for Each Participant ---
fprintf('\n--- Heart Rate Error Summary for Each Participant ---\n');

% Initialize lists to collect all valid error data across ALL clips
all_clips_hr_abs_error_bpm = [];
all_clips_hr_rel_error_percent = [];

% Initialize lists to collect participant-level statistics
all_participants_mean_abs_errors = [];
all_participants_std_abs_errors = [];
all_participants_median_abs_errors = [];
all_participants_mean_rel_errors = [];

for i = 1:num_participants
    participant_id = participant_names_list{i};
    current_participant_clips = all_participants_data.(participant_id);
    clip_labels_list = fieldnames(current_participant_clips);

    % Initialize error collection lists for the current participant
    participant_hr_abs_errors = [];
    participant_hr_rel_errors = [];
    valid_clip_pairs_count = 0;

    for j = 1:length(clip_labels_list)
        clip_label = clip_labels_list{j};
        clip_data = current_participant_clips.(clip_label);

        % Check if both PPG and mmWave heart rates were calculated successfully
        if isfield(clip_data, 'hr_abs_error_bpm') && ~isnan(clip_data.hr_abs_error_bpm)
            valid_clip_pairs_count = valid_clip_pairs_count + 1;
            
            % Collect error data for the current participant
            participant_hr_abs_errors(end+1) = clip_data.hr_abs_error_bpm;
            if isfield(clip_data, 'hr_rel_error_percent') && ~isnan(clip_data.hr_rel_error_percent)
                participant_hr_rel_errors(end+1) = clip_data.hr_rel_error_percent;
            end
            
            % Add to the overall lists (for all clips) at the same time
            all_clips_hr_abs_error_bpm(end+1) = clip_data.hr_abs_error_bpm;
            if isfield(clip_data, 'hr_rel_error_percent') && ~isnan(clip_data.hr_rel_error_percent)
                all_clips_hr_rel_error_percent(end+1) = clip_data.hr_rel_error_percent;
            end
        end
    end

    % Calculate and report error statistics for this participant
    fprintf('Participant: %s\n', participant_id);
    if valid_clip_pairs_count > 0
        mean_abs_err = mean(participant_hr_abs_errors);
        std_abs_err = std(participant_hr_abs_errors); % Calculate standard deviation
        median_abs_err = median(participant_hr_abs_errors);
        
        fprintf('  Number of valid clip pairs: %d\n', valid_clip_pairs_count);
        fprintf('  Mean Absolute Error: %.2f ± %.2f BPM\n', mean_abs_err, std_abs_err);
        fprintf('  Median Absolute Error: %.2f BPM\n', median_abs_err);
        
        % Store participant-level stats for overall summary
        all_participants_mean_abs_errors(end+1) = mean_abs_err;
        all_participants_std_abs_errors(end+1) = std_abs_err;
        all_participants_median_abs_errors(end+1) = median_abs_err;

        if ~isempty(participant_hr_rel_errors)
            mean_rel_err = mean(participant_hr_rel_errors);
            std_rel_err = std(participant_hr_rel_errors); % Calculate standard deviation
            fprintf('  Mean Relative Error: %.2f ± %.2f %%\n', mean_rel_err, std_rel_err);
            % Store participant-level stats for overall summary
            all_participants_mean_rel_errors(end+1) = mean_rel_err;
        end
    else
        fprintf('  No valid clip pairs found to calculate error statistics.\n');
    end
    fprintf('\n');
end


% --- 2. Evaluate Overall Heart Rate Error ---
fprintf('\n--- Overall Heart Rate Error Summary (Based on all valid clip pairs from all participants) ---\n');

if ~isempty(all_clips_hr_abs_error_bpm)
    total_valid_pairs = length(all_clips_hr_abs_error_bpm);
    overall_mean_abs_error = mean(all_clips_hr_abs_error_bpm);
    overall_median_abs_error = median(all_clips_hr_abs_error_bpm);
    
    fprintf('Total number of valid cip pairs: %d\n', total_valid_pairs);
    fprintf('Mean Absolute Error (all pairs): %.2f BPM\n', overall_mean_abs_error);
    fprintf('Median Absolute Error (all pairs): %.2f BPM\n', overall_median_abs_error);

    if ~isempty(all_clips_hr_rel_error_percent)
        overall_mean_rel_error = mean(all_clips_hr_rel_error_percent);
        fprintf('Mean Relative Error (all pairs): %.2f %%\n', overall_mean_rel_error);
    end
    
    % --- Report on the range of statistics across participants ---
    fprintf('\n--- Statistics Across Participants ---\n');
    if ~isempty(all_participants_mean_abs_errors)
        fprintf('Range of individual participant MAEs:             %.2f to %.2f BPM\n', min(all_participants_mean_abs_errors), max(all_participants_mean_abs_errors));
        fprintf('Range of individual participant STDs (of Abs. Error): %.2f to %.2f BPM\n', min(all_participants_std_abs_errors), max(all_participants_std_abs_errors));
        fprintf('Range of individual participant Median Abs. Errors:   %.2f to %.2f BPM\n', min(all_participants_median_abs_errors), max(all_participants_median_abs_errors));
    end
    if ~isempty(all_participants_mean_rel_errors)
        fprintf('Range of individual participant Mean Rel. Errors:   %.2f to %.2f %%\n', min(all_participants_mean_rel_errors), max(all_participants_mean_rel_errors));
    end
else
    fprintf('No valid clip pair data found for overall error evaluation.\n');
end

fprintf('\nAll processing and evaluation complete.\n');

