% --- Batch process raw PPG data ---

% Raw PPG data is stored in "..\..\data\01_raw_data\ppg_and_gsr\P[x]\", 
% where [x] is the participant ID (two digits).

% Processed PPG data is saved to "..\..\data\02_processed_data\ppg" with 
% the filename "ppg_[participantID]_[clipNumber].csv", e.g., "ppg_P05_09.csv".

clc;
clear;
close all;

try
    current_script_path = fileparts(mfilename('fullpath'));
    base_data_dir = fullfile(current_script_path, '..', '..', 'data');
catch
    % If running in the Live Editor or command line, mfilename might be empty; use the current working directory
    warning('Could not get path via mfilename, using current working directory as base. Please ensure the script is run from the correct project structure.');
    base_data_dir = fullfile(pwd, '..', '..', 'data');
end
input_dir = fullfile(base_data_dir, '01_raw_data', 'ppg_and_gsr');
output_dir = fullfile(base_data_dir, '02_processed_data', 'ppg');
% Ensure the output directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    disp(['Output directory created: ', output_dir]);
end
% Get all subject folders
subject_dirs = dir(fullfile(input_dir, 'P*'));
subject_dirs = subject_dirs([subject_dirs.isdir]); % Keep only directories
if isempty(subject_dirs)
    error('No subject folders (starting with "P") found in the input directory "%s". Please check the path.', input_dir);
end
% --- Processing Parameters ---
fs = 200; % Sampling frequency (Hz)
% --- Loop through each subject ---
for s = 1:length(subject_dirs)
    subject_name = subject_dirs(s).name;
    disp(['>>> Processing subject: ', subject_name]);
    
    subject_input_dir = fullfile(input_dir, subject_name);
    
    % Get all CSV files for this subject
    csv_files = dir(fullfile(subject_input_dir, '*.csv'));
    
    % --- Loop through each CSV file ---
    for f = 1:length(csv_files)
        csv_filename = csv_files(f).name;
        [~, file_num, ~] = fileparts(csv_filename); % Extract file number (e.g., "09")
        
        disp(['  - Processing file: ', csv_filename]);
        
        data_filepath = fullfile(subject_input_dir, csv_filename);
        
        try
            % --- Data Reading and Preprocessing ---
            % Read data starting from the 3rd row, as the first two rows are headers
            data = readmatrix(data_filepath, 'NumHeaderLines', 2);
            
            % Check data validity
            if isempty(data) || size(data, 1) < 1
                warning('File %s is empty or has insufficient columns, skipping.', csv_filename);
                continue;
            end
            
            % Extract raw PPG data (1st column), process the entire signal
            % The first 10 seconds are for the fixation cross.
            ppgRaw = data(fs * 10 + 1 : end, 1);
            ppgRaw = ppgRaw(:); % Ensure it is a column vector
            
            % --- PPG Signal Processing Pipeline ---
            
            % 1. Band-pass filtering (to keep the typical PPG frequency range of 0.6-5Hz)
            filter_order = 4;
            ppgFiltered = ppgRaw; % Default value, in case filtering fails
            
            % Only apply the filter if the signal is long enough
            if length(ppgRaw) >= 3 * (2 * filter_order) + 1
                try
                    [b, a] = butter(filter_order, [0.6 5]/(fs/2), 'bandpass');
                    ppgFiltered = filtfilt(b, a, ppgRaw);
                catch ME_filter
                     warning('Filtering failed for file %s. Using the unfiltered signal for subsequent processing. Error: %s', ...
                            csv_filename, ME_filter.message);
                end
            else
                warning('Signal length (%d) in file %s is too short, skipping the filtering step.', ...
                        csv_filename, length(ppgRaw));
            end
            
            % 2. Detrending (to remove baseline drift)
            ppgDetrended = detrend(ppgFiltered);
            
            % 3. Smoothing (moving average)
            ppgSmooth = smoothdata(ppgDetrended, 'movmean', 20);
            
            % --- Save Results ---
            % Construct output filename: ppg_PXX_XX.csv
            output_filename = sprintf('ppg_%s_%s.csv', subject_name, file_num);
            output_path = fullfile(output_dir, output_filename);
            
            % Save the processed single-column signal to a CSV file
            writematrix(ppgSmooth, output_path);
            
            disp(['    √ Processed PPG signal saved to: ', output_filename]);
            
        catch ME
            warning('A critical error occurred while processing file %s: %s', csv_filename, ME.message);
            % Print stack trace for debugging
            disp(ME.getReport());
            continue;
        end
    end
end
disp(' ');
disp('=========================');
disp('  PPG signal processing complete');
disp('=========================');