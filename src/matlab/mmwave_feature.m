% --- mmWave feature extraction ---

% Perform feature extraction on mmWave signals and save the feature matrices.
% Read mmWave signals from "....\data\02_processed_data\mmwave", 
% perform feature extraction, and save the feature matrices to "....\data\03_features\mmwave".

tic;

clc;
clear;
close all;

%% 1. Basic Parameter Settings
% Basic Signal Parameters
framePeriodicity = 10e-3; % Frame periodicity (s)
fs = 1 / framePeriodicity;      % Sampling frequency (Hz)

% Feature Extraction Parameters
secForFeature = 60;     % Duration for feature extraction (seconds)

winSize = round(fs * 5);      % Window size
winStride = round(fs * 5); % Window stride


%% 2. Path Definitions
try
    projectRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..');
catch
    % If running in the command line, mfilename might be empty; use the current working directory
    projectRoot = fullfile(pwd, '..', '..');
    fprintf('Warning: Could not determine path using mfilename. Assuming project root is the parent-of-parent directory.\n');
end

% Define input and output directories
inputDataDir = fullfile(projectRoot, 'data', '02_processed_data', 'mmwave');
outputFeatureDir = fullfile(projectRoot, 'data', '03_features', 'mmwave');

% Check if the input directory exists
if ~exist(inputDataDir, 'dir')
    error('Input data directory does not exist: %s', inputDataDir);
end

% Check for and create the output directory
if ~exist(outputFeatureDir, 'dir')
    fprintf('Output directory does not exist, creating: %s\n', outputFeatureDir);
    mkdir(outputFeatureDir);
end

fprintf('Input data directory: %s\n', inputDataDir);
fprintf('Feature output directory: %s\n', outputFeatureDir);


%% 3. Batch Process All CSV Files
% Get all CSV files matching the naming convention
csvFiles = dir(fullfile(inputDataDir, 'mmwave_P*_*.csv'));

if isempty(csvFiles)
    error('No "mmwave_P*_*.csv" files found in the input directory.');
end

fprintf('\nFound %d CSV files to process.\n', length(csvFiles));
disp('==================================================');

% Prepare storage containers for the loop
numFiles = length(csvFiles);
featureDataCell = cell(numFiles, 1); % To store feature data for each file
outputPathsCell = cell(numFiles, 1); % To store the output path for each file
errorFlags = false(numFiles, 1);     % To flag files that failed to process

% Note: If the MATLAB Parallel Computing Toolbox is installed, using parfor can significantly speed up processing. 
% If not, change "parfor" below to "for", and the program will run in a slower, serial mode.
parfor i = 1:numFiles
    csvFileName = csvFiles(i).name;
    fprintf('(%d/%d) Processing file: %s\n', i, numFiles, csvFileName);
    
    % Parse subject ID and clip number from the filename (e.g., 'mmwave_P05_09.csv')
    tokens = regexp(csvFileName, 'mmwave_(P\d+)_(\d+).csv', 'tokens');
    if isempty(tokens)
        fprintf('  -> Warning: Filename "%s" does not match the format "mmwave_PXX_YY.csv". Skipping.\n', csvFileName);
        continue;
    end
    subjectID = tokens{1}{1}; % e.g., 'P05'
    clipNum = tokens{1}{2};   % e.g., '09'
    
    try
        %% 3.1. Load and Prepare Data
        csvFullPath = fullfile(inputDataDir, csvFileName);
        
        % Use readtable to read the CSV, which automatically handles headers
        T = readtable(csvFullPath);
        
        % Check if necessary columns exist
        requiredCols = {'vital', 'respiration', 'heartbeat'};
        if ~all(ismember(requiredCols, T.Properties.VariableNames))
            fprintf('  -> Error: File "%s" is missing required columns (vital, respiration, heartbeat). Skipping.\n', csvFileName);
            continue;
        end
        
        % Extract data by header to ensure correct order
        vital_raw = T.vital;
        respiration_raw = T.respiration;
        heartbeat_raw = T.heartbeat;

        % Truncate the data to the last secForFeature seconds
        totalSamples = length(vital_raw);
        samplesToKeep = round(secForFeature * fs);
        
        if totalSamples < samplesToKeep
            fprintf('  -> Warning: Total signal length (%d samples) is less than the specified %d seconds (%d samples). Using all available data.\n', totalSamples, secForFeature, samplesToKeep);
            startIndex = 1;
        else
            startIndex = totalSamples - samplesToKeep + 1;
        end
        
        % Assign the truncated signals to the variable names expected by the feature extraction functions
        phi = vital_raw(startIndex:end);
        respiration = respiration_raw(startIndex:end);
        heartbeat = heartbeat_raw(startIndex:end);
        
        % Ensure signals are column vectors
        phi = phi(:);
        respiration = respiration(:);
        heartbeat = heartbeat(:);
        
        % Check if the signal length is sufficient for at least one window of feature extraction
        if length(phi) < winSize
            fprintf('  -> Warning: Truncated signal length (%d) is smaller than the window size (%d). Cannot extract features. Skipping.\n', length(phi), winSize);
            continue;
        end
        
        %% 3.2. Feature Extraction
        % --- Waveform Features ---
        % Statistical
        [means, sd, meanAbsDiff, meanAbsDiffNor, meanAbsDiff2, meanAbsDiff2Nor] = utils.statisticalFeature(phi, winSize, winStride);
        % Physical
        [nonStationarityIndex, fractalDimension] = utils.physicalFeature(phi, winSize, winStride);
        % PSD Energy
        [respirationPSDEnergy, heartBeatPSDEnergy] = utils.PSDEnergy_mmwave(respiration, heartbeat, winSize, winStride, fs);
        % HHT
        [meanFrequencies, meanSquaredAmplitudes] = utils.HHTFreAmp(phi, winSize, winStride, fs, 4);

        % --- HRV Features ---
        h = 1 / fs; 
        f_2d = heartbeat;
        n = length(f_2d); 
        
        if n < 7
            fprintf('  -> Warning: Heartbeat signal is too short (%d points) to calculate HRV features. Setting HRV features to NaN.\n', n);
            d2f = []; % Mark as invalid
        else
            d2f_core = zeros(1, n - 6);
            for i_hrv = 4:(n - 3)
                numerator = 4*f_2d(i_hrv) + (f_2d(i_hrv+1)+f_2d(i_hrv-1)) - 2*(f_2d(i_hrv+2)+f_2d(i_hrv-2)) - (f_2d(i_hrv+3)+f_2d(i_hrv-3));
                d2f_core(i_hrv - 3) = numerator / (16 * h^2); 
            end
            % Pad d2f to match the length of heartbeat for subsequent window operations
            d2f = [repmat(d2f_core(1), 1, 3), d2f_core, repmat(d2f_core(end), 1, 3)];
            
            % Smooth d2f
            window_size_smooth = round(0.2 * fs);
            if length(d2f) > window_size_smooth && window_size_smooth > 1
                 d2f = smoothdata(d2f, 'gaussian', window_size_smooth);
            end
        end

        % Calculate HRV features, filling with NaN if d2f is invalid
        if ~isempty(d2f) && length(d2f) >= winSize
            m = 50; b1 = fs / 1.8; b2 = fs / 1.0; max_iter = 30;
            [timeDomain, nonlinear] = utils.HRVfeature_mmwave(d2f, winSize, winStride, b1, b2, m, max_iter, fs);
        else
            % Create a structure containing NaNs to maintain a consistent feature matrix structure
            nan_array = NaN(size(means)); % Use the dimensions of an existing feature as a reference
            fields_time = {'meanNN', 'medianNN', 'SDNN', 'RMSSD', 'PNN50', 'meanRate', 'sdRate', 'HRVTi'};
            fields_nonlin = {'PoincareSD1'};
            timeDomain = struct();
            for f_idx = 1:length(fields_time)
                timeDomain.(fields_time{f_idx}) = nan_array;
            end
            nonlinear = struct();
            for f_idx = 1:length(fields_nonlin)
                nonlinear.(fields_nonlin{f_idx}) = nan_array;
            end
        end

        %% 3.3. Consolidate All Features
        statFea = [means', sd', meanAbsDiff', meanAbsDiffNor', meanAbsDiff2', meanAbsDiff2Nor'];
        phyTimeFea = [nonStationarityIndex', fractalDimension'];
        PSDFea = [respirationPSDEnergy', heartBeatPSDEnergy'];
        HHTFea = [meanFrequencies', meanSquaredAmplitudes'];
        
        timeHRVFea = [timeDomain.meanNN', timeDomain.medianNN', timeDomain.SDNN', timeDomain.RMSSD', ...
                      timeDomain.PNN50', timeDomain.meanRate', timeDomain.sdRate', timeDomain.HRVTi'];

        nonlinearHRVFea = nonlinear.PoincareSD1';
        
        % Combine all features into a single matrix
        featureMatrix = [statFea, phyTimeFea, PSDFea, HHTFea, timeHRVFea, nonlinearHRVFea];

        %% 3.4. Store Feature Data for Saving
        % --- Store the results in the cell array ---
        featureDataCell{i} = featureMatrix;
        outputPathsCell{i} = fullfile(outputFeatureDir, sprintf('mmwaveFea_%s_%s.mat', subjectID, clipNum));

    catch ME
        fprintf('  -> ***** A critical error occurred while processing file %s *****\n', csvFileName);
        fprintf('  -> Error message: %s\n', ME.message);
        fprintf('  -> Error occurred in file: %s, at line: %d\n', ME.stack(1).name, ME.stack(1).line);
        errorFlags(i) = true; % Mark as an error
        continue;
    end
end

%% Save files after the loop finishes
fprintf('\n==================================================\n');
fprintf('Processing complete. Now saving %d files...\n', sum(~errorFlags));

for i = 1:numFiles
    % Only save the files that were processed successfully
    if ~errorFlags(i) && ~isempty(featureDataCell{i})
        % Get the data and path from the cell arrays
        featureMatrix = featureDataCell{i};
        outputFullPath = outputPathsCell{i};
        
        try
            save(outputFullPath, 'featureMatrix');
            fprintf('  -> Features successfully saved to: %s\n', outputFullPath);
        catch ME_save
            fprintf('  -> ***** An error occurred while saving file %s *****\n', outputFullPath);
            fprintf('  -> Error message: %s\n', ME_save.message);
        end
    else
        if errorFlags(i)
             fprintf('  -> Skipping save for file #%d because an error occurred during processing.\n', i);
        end
    end
end

disp(' ');
disp('=====================================');
disp('  mmWave feature extraction complete');
disp('=====================================');

elapsedTime = toc;
disp(['代码运行耗时: ', sprintf('%.4f', elapsedTime), ' 秒']);