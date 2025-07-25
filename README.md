# An Emotion Recognition Dataset Using Millimeter Wave Radar and Physiological Reference Signals

This GitHub project hosts the source code for the research  "An emotion recognition dataset using millimeter wave radar and physiological reference signals".



## Directory Structure
- **data/**: Contains the dataset folders `01_raw_data`, `02_processed_data` , and `03_features`.
- **misc/**: Contains miscellaneous files related to this research.
- **results/**: Stores the emotion recognition results.
- **src/**: Contains all source code.
  - **matlab/**: Contains MATLAB source code (.m files and the `+utils` function package).
  - **python/**: Contains Python source code.



## Source Code Description
The following is a brief description of each script in the order they are used in the research.
### MATLAB Scripts (`src/matlab/`)
- **`mmwave_process.m`**: This script reads the raw millimeter-wave (mmWave) radar data from `data/01_raw_data/mmwave`, processes data, and stores the processed data for each participant during each clip in `data/02_processed_data/mmwave`. This script calls functions from the `+utils` package: `bandPassFilter.m` for band-pass filtering the signal, and `extractPhase.m` which performs a range-FFT on the input mmWave data, selects the target's range bin (distance), and extracts the phase. **Note:** You can skip this script if you have already downloaded and unzipped `02_processed_data.zip` into the `data` folder.

- **`ppg_process.m`**: This script reads the raw photoplethysmography (PPG) data from `data/01_raw_data/ppg`, processes it, and stores the processed data for each participant during each clip in `data/02_processed_data/ppg`. **Note:** You can skip this script if you have already downloaded and unzipped `02_processed_data.zip` into the `data` folder.

- **`gsr_process.m`**: This script reads the raw galvanic skin response (GSR) data from `data/01_raw_data/gsr`, processes it, and stores the processed data for each participant during each clip in `data/02_processed_data/gsr`. **Note:** You can skip this script if you have already downloaded and unzipped `02_processed_data.zip` into the `data` folder.

- **`mmwave_feature.m`**: This script reads the processed mmWave data from `data/02_processed_data/mmwave`, extracts features, and saves them as `.mat` files in `data/03_features/mmwave` for each participant and clip. It utilizes the following functions from `+utils`: `statisticalFeature.m`: Calculates statistical features of the input signal. `physicalFeature.m`: Calculates the non-stationary index (NSI) and Higuchi fractal dimension (HFD). `PSDEnergy_mmwave.m`: Computes the power spectral density (PSD) energy in different frequency bands of the mmWave signal. `HHTFreAmp.m`: Calculates the mean instantaneous frequency and amplitude of the intrinsic mode functions (IMFs) from the Hilbert-Huang Transform (HHT). `HRVfeature_mmwave.m`: Extracts heart rate variability (HRV) features from the heartbeat signal derived from the mmWave data. **Note**: You can skip this script if you have already downloaded and unzipped `03_features.zip` into the `data` folder.

- **`ppg_feature.m`**: This script reads the processed PPG data from `data/02_processed_data/ppg`, extracts features, and saves them as `.mat` files in `data/03_features/ppg`. It uses the following `+utils` functions: `statisticalFeature.m`, `physicalFeature.m`, `HHTFreAmp.m` (as described above), `PSDEnergy_ppg.m`, computes PSD energy for PPG signals, and `HRVfeature_ppg.m`, calculates HRV features from the PPG signal. **Note:** You can skip this script if you have already downloaded and unzipped `03_features.zip` into the `data` folder.

- **`gsr_feature.m`**: This script reads the processed GSR data from `data/02_processed_data/gsr`, extracts features, and saves them as `.mat` files in `data/03_features/gsr`. It calls the `featureGSR.m` function from `+utils` to compute GSR signal features. **Note:** You can skip this script if you have already downloaded and unzipped `03_features.zip` into the `data` folder.

- **`self_assessment_analysis.m`**: This script validates the participants' self-assessment emotional ratings. It reads the self-assessment scores from `data/01_raw_data/self_assessment/SAM`, calculates the coefficient of variation to measure rating consistency, and visualizes the distribution of clip ratings in the Valence-Arousal space. It also examines the correlation between our study's ratings and a reference study by reading `misc/"Ratings of Gabert-Quillen et al. study.csv"`. Finally, it computes correlations between the emotional scales (valence, arousal, dominance) and the clip presentation order. Results are displayed in the MATLAB command window.

- **`snr_analysis_ppg_and_gsr.m`**: This script validates the signal quality of the PPG and GSR data by reading each signal type and calculating its Signal-to-Noise Ratio (SNR). Results are displayed in the MATLAB command window.

- **`hr_comparison.m`**: This script compares the heart rates derived from mmWave radar and PPG to validate the quality of the mmWave signal. It calculates metrics such as Mean Absolute Error (MAE), Median Absolute Error, and Mean Relative Error. Results are displayed in the MATLAB command window.

### Python Scripts (`src/python/`)
- **`classification.py`**: This script performs emotion classification using the participants' self-assessment scores for each clip as the ground truth. It uses the first 75% of the samples from all trials as the training set and the remaining 25% as the test set. Users can configure the signal type (mmWave, PPG, and GSR) in the global settings before running the script. Detailed results will be saved in the `results/` directory.
- **`resultsSummary.py`**: This script aggregates the classification results and compares the accuracy and F1-scores for valence, arousal, and dominance across different signal types. The results are saved in a table named `resultsTable.txt` within the `results/` folder.



## Environment
- **MATLAB**: tested with R2024b
- **Python**: 3.9+



## Usage Instructions
#### 1. Clone the Repository
   Clone this repository to your local machine.
#### 2. Get the Dataset
   Download the required dataset file(s) and unzip it (them) into the `data/` folder.
#### 3. Configure Python Environment (Virtual Environment Recommended)
   The required packages and their versions are listed in `src/python/requirements.txt`. Install them by navigating to the project directory in your virtual environment and running the following command:

   ```
   pip install -r src/python/requirements.txt
   ```

#### 4. Run the Scripts
   Execute the relevant scripts as described in the "Source Code Description" section to process data, extract features, and run classification experiments.