import os
import glob
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Script Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE_RESULTS_PATH = os.path.join(PROJECT_ROOT_PATH, "results")

# Define the rows and columns for the tables, ensuring the order is correct
FEATURE_TYPES_ORDER = ["ppg", "mmwave", "gsr"]
EMOTION_DIMENSIONS_ORDER = ["valence", "arousal", "dominance"]

def collect_results(base_path, feature_types, emotion_dimensions):
    """
    Traverses the results directory to collect data from all ACCnF1.txt files.
    
    Returns:
        A nested dictionary with the structure: 
        {feature: {dimension: {'accuracies': [...], 'f1_scores': [...], 'bacc_scores': [...]}}}
    """
    # Initialize data storage structure
    all_results = {
        ft: {
            dim: {'accuracies': [], 'f1_scores': [], 'bacc_scores': []}
            for dim in emotion_dimensions
        }
        for ft in feature_types
    }

    print(f"Starting to collect results from the root directory: {base_path}")

    # Iterate over each feature type
    for feature in feature_types:
        feature_path = os.path.join(base_path, feature)
        if not os.path.exists(feature_path):
            print(f"Warning: Feature directory not found: {feature_path}, skipping...")
            continue
        
        # Iterate over each emotion dimension
        for dimension in emotion_dimensions:
            dimension_path = os.path.join(feature_path, dimension)
            if not os.path.exists(dimension_path):
                print(f"Warning: Dimension directory not found: {dimension_path}, skipping...")
                continue
            
            # Use glob to find results folders for all subjects
            # Pattern: .../results/ppg/valence/P01_ppg_valence/ACCnF1.txt
            search_pattern = os.path.join(dimension_path, '*', 'ACCnF1.txt')
            acc_f1_files = glob.glob(search_pattern)
            
            if not acc_f1_files:
                print(f"Info: No ACCnF1.txt files found in {dimension_path}.")

            # Read each file found
            for f_path in acc_f1_files:
                try:
                    with open(f_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            accuracy = float(lines[0].strip())
                            f1_score = float(lines[1].strip())
                            
                            all_results[feature][dimension]['accuracies'].append(accuracy)
                            all_results[feature][dimension]['f1_scores'].append(f1_score)
                            
                            if len(lines) >= 3:
                                bacc_score = float(lines[2].strip())
                                all_results[feature][dimension]['bacc_scores'].append(bacc_score)
                        else:
                            print(f"Warning: Incorrect file format, skipping: {f_path}")
                except (IOError, ValueError) as e:
                    print(f"Error: Failed to read or parse file: {f_path}, Error: {e}")

    return all_results


def generate_summary_tables(results_data, feature_order, dimension_order, output_path):
    """
    Calculates the mean from the collected data, generates Pandas DataFrame tables,
    prints them to the console, and saves them to a single file.
    """
    # Initialize dictionaries to hold table data
    accuracy_data = {dim.capitalize(): [] for dim in dimension_order}
    f1_score_data = {dim.capitalize(): [] for dim in dimension_order}
    bacc_score_data = {dim.capitalize(): [] for dim in dimension_order}

    # Populate data
    for feature in feature_order:
        for dimension in dimension_order:
            # Safely get lists of accuracies and F1 scores.
            # .get() with a default value handles cases where a feature or dimension was not found.
            accuracies = results_data.get(feature, {}).get(dimension, {}).get('accuracies', [])
            f1_scores = results_data.get(feature, {}).get(dimension, {}).get('f1_scores', [])
            bacc_scores = results_data.get(feature, {}).get(dimension, {}).get('bacc_scores', [])
            
            # Calculate the mean of accuracies
            if accuracies:
                acc_mean = np.mean(accuracies)
                accuracy_data[dimension.capitalize()].append(f"{acc_mean:.3f}")
            else:
                accuracy_data[dimension.capitalize()].append("N/A")

            # Calculate the mean of F1 scores
            if f1_scores:
                f1_mean = np.mean(f1_scores)
                f1_score_data[dimension.capitalize()].append(f"{f1_mean:.3f}")
            else:
                f1_score_data[dimension.capitalize()].append("N/A")
                
            # Calculate the mean of Balanced Accuracies
            if bacc_scores:
                bacc_mean = np.mean(bacc_scores)
                bacc_score_data[dimension.capitalize()].append(f"{bacc_mean:.3f}")
            else:
                bacc_score_data[dimension.capitalize()].append("N/A")

    # Create Pandas DataFrames
    # Row index names and order
    index_labels = [f.upper() for f in feature_order]
    
    df_accuracy = pd.DataFrame(accuracy_data, index=index_labels)
    df_f1_score = pd.DataFrame(f1_score_data, index=index_labels)
    df_bacc_score = pd.DataFrame(bacc_score_data, index=index_labels)
    
    # --- Print to Console ---
    print("\n" + "="*80)
    print("Table 1: Mean Accuracy of Each Feature for Different Emotion Scales")
    print("="*80)
    print(df_accuracy)
    
    print("\n" + "="*80)
    print("Table 2: Mean Balanced Accuracy of Each Feature for Different Emotion Scales")
    print("="*80)
    print(df_bacc_score)

    print("\n" + "="*80)
    print("Table 3: Mean F1-Score of Each Feature for Different Emotion Scales")
    print("="*80)
    print(df_f1_score)

    # --- Save to File ---
    try:
        # Use a single with-statement to write all content to the file
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Table 1: Mean Accuracy of Each Feature for Different Emotion Scales\n")
            f.write("="*80 + "\n")
            f.write(df_accuracy.to_string())
            
            f.write("\n\n\n")  # Add spacing between tables
            f.write("="*80 + "\n")
            f.write("Table 2: Mean Balanced Accuracy of Each Feature for Different Emotion Scales\n")
            f.write("="*80 + "\n")
            f.write(df_bacc_score.to_string())
            
            f.write("\n\n\n")  # Add spacing between tables
            f.write("="*80 + "\n")
            f.write("Table 3: Mean F1-Score of Each Feature for Different Emotion Scales\n")
            f.write("="*80 + "\n")
            f.write(df_f1_score.to_string())
        
        print(f"\nSummary tables have been successfully saved to: {output_path}")

    except IOError as e:
        print(f"\nError: Could not write summary tables to file: {output_path}. Error: {e}")

if __name__ == "__main__":
    # Ensure the base results directory exists to avoid an error when saving the file.
    if not os.path.exists(BASE_RESULTS_PATH):
        print(f"Error: The base results directory does not exist: {BASE_RESULTS_PATH}")
        print("Please create it or check the PROJECT_ROOT_PATH configuration.")
    else:
        # 1. Collect all results data
        collected_data = collect_results(BASE_RESULTS_PATH, FEATURE_TYPES_ORDER, EMOTION_DIMENSIONS_ORDER)
        
        # 2. Define output file path
        output_file_path = os.path.join(BASE_RESULTS_PATH, "resultsTable.txt")
        
        # 3. Generate, print, and save the summary tables
        generate_summary_tables(collected_data, FEATURE_TYPES_ORDER, EMOTION_DIMENSIONS_ORDER, output_file_path)
        
        print("\nSummary report generation process finished.")
