import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline


# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
# MODEL_TYPE = "random_forest"
MODEL_TYPE = "svm"

# FEATURE_TYPE = "mmwave"
# FEATURE_TYPE = "ppg"
FEATURE_TYPE = "gsr"

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------

# __file__ may be undefined in some interactive environments (like Jupyter), so a fallback is provided
try:
    current_file_path = os.path.dirname(__file__)
except NameError:
    current_file_path = os.getcwd() # Or manually specify the script's directory

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(current_file_path, '..', '..'))

# Data and results path structure
FEATURES_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "03_features")
SAM_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "01_raw_data", "self_assessment", "SAM")
BASE_RESULTS_PATH = os.path.join(PROJECT_ROOT_PATH, "results")

# Define results save path specific to the feature type
FEATURE_SPECIFIC_RESULTS_ROOT_PATH = os.path.join(BASE_RESULTS_PATH, FEATURE_TYPE)
os.makedirs(FEATURE_SPECIFIC_RESULTS_ROOT_PATH, exist_ok=True)
print(f"Project Root Path: {PROJECT_ROOT_PATH}")
print(f"Features Data Source: {FEATURES_DATA_PATH}")
print(f"SAM Data Source: {SAM_DATA_PATH}")
print(f"Results will be saved to: {FEATURE_SPECIFIC_RESULTS_ROOT_PATH}")

# -----------------------------------------------------------------------------
# List of Emotion Scales
# -----------------------------------------------------------------------------
ALL_EMOTION_SCALES = ["valence", "arousal", "dominance"]

# -----------------------------------------------------------------------------

def load_data_for_task(participant_id, emotion_scale, feature_type, features_data_path, sam_data_path):
    """
    Loads and preprocesses data for a specified participant and emotion scale.
    Returns: final_features_for_model (list of np.array), final_labels_for_model (list)
    """

    final_features_for_model = []
    final_labels_for_model = []

    # SAM rating file location
    sam_file_path = os.path.join(sam_data_path, f"SAM_{participant_id}.csv")
    if not os.path.exists(sam_file_path):
        print(f"Warning: SAM rating file not found for participant {participant_id}: {sam_file_path}")
        return [], []
    
    try:
        sam_data = pd.read_csv(sam_file_path, header=0)
        # SAM file: 1st column is video ID, columns 2-4 are V, A, D ratings
        if emotion_scale == "valence":
            scores = sam_data.iloc[:, 1].values
        elif emotion_scale == "arousal":
            scores = sam_data.iloc[:, 2].values
        elif emotion_scale == "dominance":
            scores = sam_data.iloc[:, 3].values
        else:
            print(f"Error: Unknown emotion scale '{emotion_scale}'.")
            return [], []

        video_ids = sam_data.iloc[:, 0].values
        labels_arr = np.where(scores > 5, 1, np.where(scores < 5, 0, -1))

        # Feature file location
        fea_folder = os.path.join(features_data_path, feature_type)
        if not os.path.exists(fea_folder):
            print(f"Warning: Feature folder for {feature_type} not found: {fea_folder}")
            return [], []

        loaded_emotion_features = []
        
        for i, video_id in enumerate(video_ids):
            if labels_arr[i] == -1: # Skip neutral ratings (score of 5)
                continue

            video_id_str = f"{int(video_id):02d}"
            # .mat filename format: "featureTypeFea_participantID_videoID.mat"
            fea_files = glob.glob(os.path.join(fea_folder, f"{feature_type}Fea_{participant_id}_{video_id_str}*.mat"))
            if not fea_files:
                print(f"Info: Feature file for participant {participant_id}, video {video_id_str}, type {feature_type} not found.")
                continue

            try:
                mat_data = sio.loadmat(fea_files[0])
                # Dynamically find the feature matrix variable in the .mat file
                feature_matrix = next((mat_data[key] for key in mat_data if not key.startswith('__') and isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2), None)
                if feature_matrix is None: continue

                # Data cleaning
                feature_matrix = feature_matrix.astype(np.float32)
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, 
                                               posinf=np.finfo(np.float32).max, 
                                               neginf=np.finfo(np.float32).min)
                feature_matrix = np.clip(feature_matrix, np.finfo(np.float32).min, np.finfo(np.float32).max)

                loaded_emotion_features.append(feature_matrix)
                final_labels_for_model.extend([labels_arr[i]] * feature_matrix.shape[0])

            except Exception as e:
                print(f"Error processing file {fea_files[0]}: {e}")

        # Load baseline data (video ID '00')
        baseline_files = glob.glob(os.path.join(fea_folder, f"{feature_type}Fea_{participant_id}_00*.mat"))
        if baseline_files:
            try:
                baseline_mat_data = sio.loadmat(baseline_files[0])
                baseline_matrix = next((baseline_mat_data[key] for key in baseline_mat_data if not key.startswith('__') and isinstance(baseline_mat_data[key], np.ndarray) and len(baseline_mat_data[key].shape) == 2), None)
                if baseline_matrix is not None:
                    # Data cleaning
                    baseline_matrix = baseline_matrix.astype(np.float32)
                    baseline_matrix = np.nan_to_num(baseline_matrix, nan=0.0,
                                                    posinf=np.finfo(np.float32).max,
                                                    neginf=np.finfo(np.float32).min)
                    baseline_matrix = np.clip(baseline_matrix, np.finfo(np.float32).min, np.finfo(np.float32).max)
            except Exception as e:
                print(f"Error processing baseline file {baseline_files[0]}: {e}")

        if loaded_emotion_features:
            final_features_for_model = loaded_emotion_features
            
    except Exception as e:
        print(f"Error processing SAM file or features for participant {participant_id} ({emotion_scale}): {e}")
        return [], []

    if not final_features_for_model:
        print(f"Warning: Failed to load any valid data for participant {participant_id} on the {emotion_scale} scale.")
    else:
        for idx, fm_check in enumerate(final_features_for_model):
            if np.any(np.isinf(fm_check)) or np.any(np.isnan(fm_check)):
                print(f"Warning! Final feature matrix {idx} still contains Inf or NaN! Processing logic needs to be checked.")
            if fm_check.dtype != np.float32:
                 print(f"Warning! Data type of final feature matrix {idx} is not float32, but {fm_check.dtype}!")
    
    return final_features_for_model, final_labels_for_model

def train_and_evaluate_model_intra_video_split(features_list, labels_list, participant_id, emotion_scale, model_type,
                                               feature_type_for_naming,
                                               current_task_specific_folder_path,
                                               train_split_ratio=0.8,
                                               k_folds_cv_grid_search=3):
    """
    Splits data into training and testing sets, then trains and evaluates the model.
    """
    if not features_list or not labels_list:
        print(f"Error: Data for participant {participant_id}, scale {emotion_scale} ({feature_type_for_naming}) is empty. Cannot perform classification.")
        return None
        
    print(f"\nStarting classification for participant {participant_id}, scale {emotion_scale} ({feature_type_for_naming}) using {model_type} model...")

    # -------------------------------------------------------------------------
    # Step 1: Split the data
    # -------------------------------------------------------------------------
    
    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []
    
    current_label_idx = 0
    for video_idx, feature_matrix in enumerate(features_list):
        num_samples = feature_matrix.shape[0]
        if num_samples == 0:
            continue
            
        video_labels = labels_list[current_label_idx : current_label_idx + num_samples]
        split_point = int(num_samples * train_split_ratio)

        if split_point > 0 and split_point < num_samples:
            X_train_parts.append(feature_matrix[:split_point, :])
            y_train_parts.extend(video_labels[:split_point])
            X_test_parts.append(feature_matrix[split_point:, :])
            y_test_parts.extend(video_labels[split_point:])
        elif split_point == num_samples:
            X_train_parts.append(feature_matrix)
            y_train_parts.extend(video_labels)
        else: # split_point == 0
            X_test_parts.append(feature_matrix)
            y_test_parts.extend(video_labels)
            
        current_label_idx += num_samples

    if not X_train_parts or not X_test_parts:
        print(f"Error: After splitting, the training or testing set is empty. Cannot continue.")
        return None

    # Aggregate segments from all videos to get the final training and testing sets
    X_train = np.vstack(X_train_parts)
    y_train = np.array(y_train_parts)
    X_test = np.vstack(X_test_parts)
    y_test = np.array(y_test_parts)
    original_feature_count = X_train.shape[1]

    # Get sample statistics for reporting
    unique_train_labels, counts_train_labels = np.unique(y_train, return_counts=True)
    train_counts_dict = dict(zip(unique_train_labels, counts_train_labels))
    train_samples_info = f"Training set samples: {len(y_train)} (Class 0: {train_counts_dict.get(0, 0)}, Class 1: {train_counts_dict.get(1, 0)})"

    unique_test_labels, counts_test_labels = np.unique(y_test, return_counts=True)
    test_counts_dict = dict(zip(unique_test_labels, counts_test_labels))
    test_samples_info = f"Test set samples: {len(y_test)} (Class 0: {test_counts_dict.get(0, 0)}, Class 1: {test_counts_dict.get(1, 0)})"

    if len(unique_train_labels) < 2:
        print("Error: The training set has fewer than 2 classes after splitting. Cannot train effectively. Skipping this task.")
        return None
    if len(np.unique(y_test)) < 2:
        print(f"Warning: The test set contains only one class after splitting. Evaluation metrics may be meaningless.")

    # -------------------------------------------------------------------------
    # Step 2: Training and Evaluation
    # -------------------------------------------------------------------------
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # --- Model Definition, Pipeline, GridSearchCV ---
    classifier = None
    pipeline_steps = [('scaler', StandardScaler())]
    param_grid = {}
        
    if model_type == 'random_forest':
        pipeline_steps.append(('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')))
        param_grid = {'classifier__n_estimators': range(50, 501, 50), 'classifier__max_depth': [10, 20, 30, None]}

    elif model_type == 'svm':
        pipeline_steps.append(('svc', SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)))
        param_grid = {'svc__C': np.logspace(-5, 5, 11), 'svc__gamma': np.logspace(-4, 1, 6)}

    final_pipeline = SklearnPipeline(pipeline_steps)
    
    cv_strategy = StratifiedKFold(n_splits=k_folds_cv_grid_search, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=final_pipeline, param_grid=param_grid,
                               cv=cv_strategy, n_jobs=-1, verbose=0, scoring='f1_macro')
    try:
        grid_search.fit(X_train, y_train)
        classifier = grid_search.best_estimator_
    except ValueError as e:
        print(f"GridSearchCV for {model_type.upper()} failed: {e}.")
        print(f"y_train counts during error: {np.bincount(y_train.astype(int)) if y_train.size > 0 else []}")
        return None

    if classifier is None:
        print(f"Error: Model training failed.")
        return None

    y_test_pred = classifier.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    zd_flag = 0 if len(np.unique(y_test)) < 2 or len(np.unique(y_test_pred)) < 2 else 1
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=zd_flag)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=zd_flag)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=zd_flag)
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    
    # Print evaluation results
    print(f"\n=== Results ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Save Results
    # -------------------------------------------------------------------------
    
    # Prepare a summary dictionary for easy file writing
    results_summary = {
        'accuracy': float(test_accuracy),
        'f1_score': float(test_f1),
    }
    
    base_filename_without_extension = f"{participant_id}_{feature_type_for_naming}_{emotion_scale}"
    
    # --- Save detailed TXT report ---
    txt_filename = os.path.join(current_task_specific_folder_path, f"{base_filename_without_extension}.txt")
    with open(txt_filename, 'w', encoding='utf-8') as f_txt:
        f_txt.write(f"=== Metrics Summary ===\n")
        f_txt.write(f"Participant ID: {participant_id}\n")
        f_txt.write(f"Feature Type: {feature_type_for_naming}\n")
        f_txt.write(f"Number of Features: {original_feature_count}\n")
        f_txt.write(f"Emotion Scale: {emotion_scale}\n")
        f_txt.write(f"Model Type: {model_type}\n")
        f_txt.write(f"{train_samples_info}\n")
        f_txt.write(f"{test_samples_info}\n\n")
        f_txt.write("--- Results ---\n")
        f_txt.write(f"Accuracy: {test_accuracy:.4f}\n")
        f_txt.write(f"Precision: {test_precision:.4f}\n")
        f_txt.write(f"Recall: {test_recall:.4f}\n")
        f_txt.write(f"F1-Score: {test_f1:.4f}\n")
        f_txt.write(f"Confusion Matrix:\n{np.array(test_cm)}\n")
    print(f"Metrics saved to txt file: {txt_filename}")

    # --- Save a concise file with just accuracy and F1-score ---
    acc_n_f1_filename = os.path.join(current_task_specific_folder_path, "ACCnF1.txt")
    try:
        with open(acc_n_f1_filename, 'w', encoding='utf-8') as f_acc_f1:
            f_acc_f1.write(f"{results_summary['accuracy']:.4f}\n")
            f_acc_f1.write(f"{results_summary['f1_score']:.4f}\n")
        print(f"Concise accuracy and F1-score saved to: {acc_n_f1_filename}")
    except Exception as e:
        print(f"Error: Could not save ACCnF1.txt file: {e}")

    return results_summary


# --- Main Execution Logic ---
if __name__ == "__main__":
    
    current_feature_folder = os.path.join(FEATURES_DATA_PATH, FEATURE_TYPE)
    if not os.path.exists(current_feature_folder):
        raise FileNotFoundError(f"Feature folder does not exist, cannot continue: {current_feature_folder}")
    
    participant_pattern = re.compile(f"{FEATURE_TYPE}Fea_([PS][0-9]+)_")
    all_mat_files = glob.glob(os.path.join(current_feature_folder, f"{FEATURE_TYPE}Fea_*.mat"))
    
    discovered_participants = set()
    for f in all_mat_files:
        match = participant_pattern.search(os.path.basename(f))
        if match:
            discovered_participants.add(match.group(1))

    if not discovered_participants:
        print(f"Warning: No feature files matching the naming convention '{FEATURE_TYPE}Fea_Pxx_...' were found in {current_feature_folder}.")
        ALL_PARTICIPANT_IDS = []
    else:
        ALL_PARTICIPANT_IDS = sorted(list(discovered_participants))
        print(f"Discovered the following participant IDs: {ALL_PARTICIPANT_IDS}")

    for participant_id in ALL_PARTICIPANT_IDS:
        for emotion_scale in ALL_EMOTION_SCALES:
            print(f"\n{'='*20} Starting process: Participant {participant_id} - Feature {FEATURE_TYPE} - Scale {emotion_scale} {'='*20}")

            # Results save path
            scale_specific_path = os.path.join(FEATURE_SPECIFIC_RESULTS_ROOT_PATH, emotion_scale)
            os.makedirs(scale_specific_path, exist_ok=True)
            
            task_specific_folder_name_for_participant_scale = f"{participant_id}_{FEATURE_TYPE}_{emotion_scale}"
            current_task_final_folder_path = os.path.join(scale_specific_path, task_specific_folder_name_for_participant_scale)
            os.makedirs(current_task_final_folder_path, exist_ok=True)

            # Call the data loading function
            features_for_model, labels_for_model = load_data_for_task(
                participant_id=participant_id,
                emotion_scale=emotion_scale,
                feature_type=FEATURE_TYPE,
                features_data_path=FEATURES_DATA_PATH,
                sam_data_path=SAM_DATA_PATH
            )

            if features_for_model and labels_for_model:
                if len(np.unique(labels_for_model)) < 2:
                    print(f"Error: Data for participant {participant_id}, scale {emotion_scale} ({FEATURE_TYPE}) contains only one class ({np.unique(labels_for_model)}). Cannot perform classification.")
                else:
                    # Call the model training and evaluation function
                    train_and_evaluate_model_intra_video_split(
                        features_list=features_for_model,
                        labels_list=labels_for_model,
                        participant_id=participant_id,
                        emotion_scale=emotion_scale,
                        model_type=MODEL_TYPE,
                        feature_type_for_naming=FEATURE_TYPE,
                        current_task_specific_folder_path=current_task_final_folder_path,
                        train_split_ratio=0.75,
                        k_folds_cv_grid_search=3
                    )
            else:
                print(f"\nError: Model training cannot be performed for participant {participant_id}, scale {emotion_scale} ({FEATURE_TYPE}) because no valid data was loaded in the data preparation phase.")
            
            print(f"{'='*20} Finished processing: Participant {participant_id} - Feature {FEATURE_TYPE} - Scale {emotion_scale} {'='*20}\n")
    
    print("\nAll participants and emotion scales have been processed!\n")