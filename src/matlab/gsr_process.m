% --- Batch process raw GSR data ---

% Raw GSR data is stored in "..\..\data\01_raw_data\ppg_and_gsr\P[x]\", 
% where [x] is the participant ID (two digits).

% Processed GSR data is saved to "..\..\data\02_processed_data\gsr" with 
% the filename "gsr_[participantID]_[clipNumber].csv", e.g., "gsr_P05_09.csv".

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
output_dir = fullfile(base_data_dir, '02_processed_data', 'gsr');

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
            if isempty(data) || size(data, 2) < 2
                warning('File %s is empty or has insufficient columns, skipping.', csv_filename);
                continue;
            end
            
            % Extract skin resistance data (2nd column), process the entire signal
            % The first 10 seconds are for the fixation cross.
            resistance_kohm = data(fs * 10 + 1 : end, 2); 
            resistance_kohm = resistance_kohm(:); % Ensure it is a column vector
            
            % --- Convert to Skin Conductance ---
            % SC (uS) = 1 / R (MOhm) = 1000 / R (kOhm)
            conductance_uS_raw = NaN(size(resistance_kohm));
            valid_resistance_indices = resistance_kohm > 0;
            
            % Report and handle invalid resistance values (zero or negative)
            if ~all(valid_resistance_indices)
                warning('Found %d non-positive resistance values in file %s. These values will be ignored.', ...
                        sum(~valid_resistance_indices), csv_filename);
            end
            
            conductance_uS_raw(valid_resistance_indices) = 1000 ./ resistance_kohm(valid_resistance_indices);
            
            % Handle potential NaN values after conversion (e.g., from invalid resistance values)
            needs_nan_handling = isnan(conductance_uS_raw);
            if any(needs_nan_handling)
                % Fill NaN values with the median of the valid data
                valid_cond = conductance_uS_raw(~needs_nan_handling);
                if ~isempty(valid_cond)
                    median_val = median(valid_cond, 'omitnan');
                    conductance_uS_raw(needs_nan_handling) = median_val;
                    warning('Filled %d NaN values with the median.', sum(needs_nan_handling));
                else
                    warning('All conductance values are invalid, cannot process. Skipping this file.');
                    continue;
                end
            end
            
            % --- Filtering Process ---
            cutoff_freq = 1;      % Low-pass filter cutoff frequency (Hz)
            filter_order = 3;     % Filter order
            nyquist_freq = fs / 2;
            Wn = cutoff_freq / nyquist_freq;
            
            sc_signal = conductance_uS_raw; % Default to the unfiltered signal in case filtering fails
            
            % Only apply filter if the signal is long enough; filtfilt requires length >= 3*(order)+1
            if length(conductance_uS_raw) >= 3 * filter_order + 1
                try
                    [b, a] = butter(filter_order, Wn, 'low');
                    sc_signal = filtfilt(b, a, conductance_uS_raw);
                catch ME_filter
                    warning('Filtering failed for file %s. Using unfiltered signal instead. Error: %s', ...
                            csv_filename, ME_filter.message);
                end
            else
                warning('Signal length (%d) in file %s is too short for filtering. Using unfiltered signal instead.', ...
                        length(conductance_uS_raw), csv_filename);
            end
            
            % --- Save Results ---
            % Construct output filename: gsr_PXX_XX.csv
            output_filename = sprintf('gsr_%s_%s.csv', subject_name, file_num);
            output_path = fullfile(output_dir, output_filename);
            
            % Save the processed single-column signal to a CSV file
            writematrix(sc_signal, output_path);
            
            disp(['    √ Processed signal saved to: ', output_filename]);
            
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
disp('  GSR signal processing complete');
disp('=========================');