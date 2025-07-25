% --- GSR feature extraction ---

% Perform feature extraction on GSR signals and save the feature matrices.
% Read GSR signals from "....\data\02_processed_data\gsr", 
% perform feature extraction, and save the feature matrices to "....\data\03_features\gsr".

% Relative paths
base_data_dir = fullfile('..', '..', 'data'); 
input_dir = fullfile(base_data_dir, '02_processed_data', 'gsr');
output_dir = fullfile(base_data_dir, '03_features', 'gsr');

% Ensure the output directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    disp(['Output directory created: ', output_dir]);
end

% --- Processing Parameters ---
fs = 200;            % Sampling frequency (Hz)
secForFeature = 60;  % Segment length for feature extraction (seconds)
winSize = round(5 * fs);   % Window size (in points)
winStride = round(5 * fs);  % Window stride (in points)

% --- Start Processing ---
disp('Starting batch GSR feature extraction...');
disp(['Input directory: ', input_dir]);
disp(['Output directory: ', output_dir]);

% Get all CSV files that match the naming convention
csv_files = dir(fullfile(input_dir, 'gsr_*.csv'));

if isempty(csv_files)
    warning('No "gsr_*.csv" files found in the input directory. Please check the path and filenames.');
    return;
end

% --- Loop through each CSV file ---
for i = 1:length(csv_files)
    csv_filename = csv_files(i).name;
    
    % Use regular expressions to extract subject ID and clip number from the filename "gsr_PXX_XX.csv"
    tokens = regexp(csv_filename, 'gsr_(P\d+)_(\d+)\.csv', 'tokens');
    
    % Check if the filename format is correct
    if isempty(tokens)
        warning('Filename "%s" does not match the format "gsr_PXX_XX.csv", skipping.', csv_filename);
        continue;
    end
    
    subject_id = tokens{1}{1};
    clip_num = tokens{1}{2};
    
    disp(['--> Processing file: ', csv_filename]);
    
    % Construct the full input file path
    data_filepath = fullfile(input_dir, csv_filename);
    
    try
        % --- Data Reading ---
        sc_signal_raw = readmatrix(data_filepath);
        
        % Check data validity
        if isempty(sc_signal_raw)
            warning('File %s is empty, skipping.', csv_filename);
            continue;
        end
        
        % --- Data Segmentation (take only the last 60 seconds) ---
        points_to_extract = fs * secForFeature;
        
        if length(sc_signal_raw) < points_to_extract
            warning('Data length is less than %d seconds (%d points). Using all available data (%d points).', ...
                secForFeature, points_to_extract, length(sc_signal_raw));
            sc_signal = sc_signal_raw;
        else
            % Extract the segment of the specified length from the end of the data
            sc_signal = sc_signal_raw(end - points_to_extract + 1 : end);
        end
        
        % Ensure the data is a column vector
        sc_signal = sc_signal(:);
        
        % --- Feature Extraction ---
        featureMatrix = utils.featureGSR(sc_signal, winSize, winStride, fs);
        
        % --- Save Results ---
        % Construct the output filename, e.g., "gsrFea_P05_09.mat"
        output_filename = sprintf('gsrFea_%s_%s.mat', subject_id, clip_num);
        output_path = fullfile(output_dir, output_filename);
        
        save(output_path, 'featureMatrix');
        disp(['    Features saved to: ', output_path]);
        
    catch ME
        warning('An error occurred while processing file %s: %s', csv_filename, ME.message);
        % If an error occurs, skip the current file and continue to the next one
        continue; 
    end
end

disp(' ');
disp('============================');
disp('  GSR feature extraction complete');
disp('============================');