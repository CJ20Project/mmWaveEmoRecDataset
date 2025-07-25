% --- Batch process raw mmWave data ---

% Raw mmWave data is stored in "..\..\data\01_raw_data\mmwave\P[x]\", 
% where [x] is the participant ID (two digits).

% Processed mmawave data (vital sign, respiration and heartbeat) is 
% saved to "..\..\data\02_processed_data\mmwave" with the filename 
% "mmwave_[participantID]_[clipNumber].csv", e.g., "mmwave_P05_09.csv".

clc;
clear;
close all;

%% Parameters
numTx = 3;              % Number of Tx antennas
numRx = 4;              % Number of Rx antennas
numADCSamples = 256;    % Number of ADC samples per chirp
adcFs = 5120e3;         % ADC sampling rate (Hz)
c = 3e8;                % Speed of light (m/s)
ts = numADCSamples / adcFs;   % ADC sampling time (s)
slope = 66.590e12;      % Frequency slope (Hz/s)
B_valid = ts * slope;   % Effective bandwidth (Hz)
deltaR = c / (2 * B_valid); % Range resolution (m)
startFreq = 60e9;       % Start frequency (Hz)
framePeriodicity = 10e-3; % Frame periodicity (s)
fs = 1 / framePeriodicity;  % Frame rate (Hz)

%% Path Definitions
% Root directory for raw data: '..\..\data\01_raw_data\mmwave'
dataRoot = fullfile('..', '..', 'data', '01_raw_data', 'mmwave');
% Directory to save processed signals: '..\..\data\02_processed_data\mmwave'
signalsRoot = fullfile('..', '..', 'data', '02_processed_data', 'mmwave');

if ~exist(signalsRoot, 'dir')
    mkdir(signalsRoot);
end

%% Process all subject data
% Search for all subject folders starting with 'P'
subjectDirs = dir(fullfile(dataRoot, 'P*'));

for subIdx = 1:length(subjectDirs)
    subjectName = subjectDirs(subIdx).name; % e.g., 'P05'
    fprintf('Processing subject: %s\n', subjectName);
    
    % Get all .mat files for this subject
    matFiles = dir(fullfile(dataRoot, subjectName, '*.mat'));
    
    for fileIdx = 1:length(matFiles)
        matFileName = matFiles(fileIdx).name;
        [~, clipNum, ~] = fileparts(matFileName); % clipNum, e.g., '09'
        fprintf('  Processing file: %s\n', matFileName);
        
        try
            %% Load Data
            adcDataStruct = load(fullfile(dataRoot, subjectName, matFileName));
            adcData = adcDataStruct.adcData;

            % The first 10 seconds are for the fixation cross.
            adcData = adcData(:, fs * numADCSamples * 10 + 1 : end);
            
            %% Processing Pipeline
            numFrame = size(adcData, 2) / numADCSamples;
            virtual_Rx = numTx * numRx;
            data = zeros(numFrame, numADCSamples, virtual_Rx);
            
            % Reshape data into [numFrame, numADCSamples, virtual_Rx]
            for i = 1 : virtual_Rx
                data(:,:,i) = reshape(adcData(i,:), numADCSamples, numFrame).';
            end
            
            % Data related to Rx2 and Rx3 has a 180-degree phase difference that needs correction
            for j = 2 : 4 : 10
                data(:, :, j) = (-1) .* data(:, :, j);
                data(:, :, j + 1) = (-1) .* data(:, :, j + 1);
            end
            
            %% Phase Extraction
            numRangFFT = 256;
            phaseChannels = zeros(numFrame, virtual_Rx);
            for i = 1 : virtual_Rx
                dataTemp = data(:, :, i);
                dataTemp = dataTemp';
                phaseChannels(:, i) = utils.extractPhase(dataTemp, numRangFFT);
            end
            
            phaseChannels = detrend(phaseChannels);
            phiUnwraped = mean(phaseChannels, 2);
            
            % First-order difference
            phi_diff = diff(phiUnwraped, 1);
            phi_diff(end+1) = phi_diff(end); % Pad array to match original length
            phi = phi_diff;
          
            %% Bandpass Filtering
            % Respiration signal
            f1_resp = 0.1;
            f2_resp = 0.5;
            n_total_resp = 4;
            respiration_bandpass = utils.bandPassFilter(phi, fs, f1_resp, f2_resp, n_total_resp);
            
            % Heartbeat signal
            f1_heart = 1.0;
            f2_heart = 1.8;
            n_total_heart = 6;
            heartBeat_bandpass = utils.bandPassFilter(phi, fs, f1_heart, f2_heart, n_total_heart);
            
            respiration = respiration_bandpass;
            heartBeat = heartBeat_bandpass;
            
            % Construct CSV filename and full path
            csvFileName = sprintf('mmwave_%s_%s.csv', subjectName, clipNum);
            csvFullPath = fullfile(signalsRoot, csvFileName);
            
            % Combine the three signals (column vectors) into a table with headers
            T = table(phi, respiration, heartBeat, 'VariableNames', {'vital', 'respiration', 'heartbeat'});
            
            % Write the table to a CSV file
            writetable(T, csvFullPath);
            
            fprintf('  Signals saved to: %s\n', csvFullPath);
            
        catch ME
            fprintf('Error processing file %s: %s\n', matFileName, ME.message);
            % If an error occurs, ensure all figure windows are closed
            close all; 
            continue;
        end
    end
end

disp(' ');
disp('============================');
disp('  mmWave signal processing complete');
disp('============================');