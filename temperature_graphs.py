import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np 

RESULTS_DIR = "./results_temp"
OUTPUT_GRAPH_DIR = "./graphs_temp"


def load_all_results(results_directory: str) -> pd.DataFrame:
    """
    Loads all JSON experiment results from a directory into a pandas DataFrame.
    """
    all_data = []
    json_files = glob.glob(os.path.join(results_directory, "*.json"))

    if not json_files:
        print(f"No JSON files found in directory: {results_directory}")
        return pd.DataFrame()

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            args = data.get("args", {})
            metrics = data.get("final_evaluation_metrics", {})

            temperature = args.get("temperature")

            acc_vs_gt = metrics.get("avg_accuracy_s_vs_gt_script1")
            acc_vs_teacher = metrics.get("avg_accuracy_s_vs_t_script1")
            sim_vs_gt = metrics.get("avg_similarity_avg_s_vs_gt_script1")
            sim_vs_teacher = metrics.get("avg_similarity_avg_s_vs_t_script1")
            precision_vs_gt = metrics.get("avg_precision_s_vs_gt_script1")
            precision_vs_teacher = metrics.get("avg_precision_s_vs_t_script1")
            recall_vs_gt = metrics.get("avg_recall_s_vs_gt_script1")
            recall_vs_teacher = metrics.get("avg_recall_s_vs_t_script1")
            f1_vs_gt = metrics.get("avg_f1_s_vs_gt_script1")
            f1_vs_teacher = metrics.get("avg_f1_s_vs_t_script1")

            avg_loss = metrics.get("avg_loss")
            avg_inference_time = metrics.get("avg_inference_time_per_sample_ms")


            if temperature is not None:
                all_data.append({
                    "temperature": float(temperature),
                    "avg_loss": float(avg_loss) if avg_loss is not None else None,
                    "accuracy_s_vs_gt_script1": float(acc_vs_gt) if acc_vs_gt is not None else None,
                    "accuracy_s_vs_t_script1": float(acc_vs_teacher) if acc_vs_teacher is not None else None,
                    "similarity_avg_s_vs_gt_script1": float(sim_vs_gt) if sim_vs_gt is not None else None,
                    "similarity_avg_s_vs_t_script1": float(sim_vs_teacher) if sim_vs_teacher is not None else None,
                    "precision_s_vs_gt_script1": float(precision_vs_gt) if precision_vs_gt is not None else None,
                    "precision_s_vs_t_script1": float(precision_vs_teacher) if precision_vs_teacher is not None else None,
                    "recall_s_vs_gt_script1": float(recall_vs_gt) if recall_vs_gt is not None else None,
                    "recall_s_vs_t_script1": float(recall_vs_teacher) if recall_vs_teacher is not None else None,
                    "f1_s_vs_gt_script1": float(f1_vs_gt) if f1_vs_gt is not None else None,
                    "f1_s_vs_t_script1": float(f1_vs_teacher) if f1_vs_teacher is not None else None,
                    "avg_inference_time_per_sample_ms": float(avg_inference_time) if avg_inference_time is not None else None,
                    "source_file": os.path.basename(file_path)
                })
            else:
                print(f"Warning: 'temperature' not found in args for {file_path}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            print(f"An error occurred processing file {file_path}: {e}")

    if not all_data:
        print("No valid data could be extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.sort_values(by="temperature").reset_index(drop=True)
    return df

def plot_metrics(df: pd.DataFrame, output_dir: str):
    """
    Generates and saves plots based on the loaded data.
    """
    if df.empty:
        print("DataFrame is empty. No plots will be generated.")
        return

    os.makedirs(output_dir, exist_ok=True)
    unique_temps = sorted(df["temperature"].unique())

    def _create_plot(y_gt_col, y_teacher_col, title, ylabel, filename_suffix, single_line_col=None, single_line_label=None, single_line_color='green'):
        plt.figure(figsize=(10, 6))
        plotted_something = False
        if y_gt_col and y_gt_col in df.columns and not df[y_gt_col].isnull().all():
            plt.plot(df["temperature"], df[y_gt_col], marker='o', linestyle='-', label="Student vs. Ground Truth (GT)")
            plotted_something = True
        if y_teacher_col and y_teacher_col in df.columns and not df[y_teacher_col].isnull().all():
            plt.plot(df["temperature"], df[y_teacher_col], marker='x', linestyle='--', label="Student vs. Teacher")
            plotted_something = True
        if single_line_col and single_line_col in df.columns and not df[single_line_col].isnull().all():
            plt.plot(df["temperature"], df[single_line_col], marker='s', linestyle='-.', color=single_line_color, label=single_line_label or single_line_col)
            plotted_something = True

        plt.title(title)
        plt.xlabel("Temperature")
        plt.ylabel(ylabel)
        plt.xticks(unique_temps)
        if plotted_something:
            plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"temperature_vs_{filename_suffix}.png"))
        plt.close()

    plt.figure(figsize=(12, 8)) 
    
    metrics_to_plot = [
        {"metric": "accuracy", "label": "Accuracy", "color": "blue"},
        {"metric": "f1", "label": "F1-Score", "color": "orange"},
        {"metric": "precision", "label": "Precision", "color": "green"},
        {"metric": "recall", "label": "Recall", "color": "red"},
    ]

    all_values_for_ylim = [] 
    for item in metrics_to_plot:
        metric_name_root = item["metric"]
        metric_label_base = item["label"]
        metric_color = item["color"]

        col_gt = f"{metric_name_root}_s_vs_gt_script1"
        col_teacher = f"{metric_name_root}_s_vs_t_script1"

        if col_gt in df.columns and not df[col_gt].isnull().all():
            gt_values = df[col_gt]
            all_values_for_ylim.extend(gt_values.dropna().tolist())
            plt.plot(df["temperature"], gt_values, marker='o', linestyle='-', color=metric_color, label=f"{metric_label_base} (GT)")
        
        if col_teacher in df.columns and not df[col_teacher].isnull().all():
            teacher_values = df[col_teacher]
            all_values_for_ylim.extend(teacher_values.dropna().tolist())
            plt.plot(df["temperature"], teacher_values, marker='x', linestyle='--', color=metric_color, label=f"{metric_label_base} (Teacher)")

    plt.title("Effect of Temperature on Key Assertion Metrics")
    plt.xlabel("Temperature")
    plt.ylabel("Metric Value")
    plt.xticks(unique_temps)
    

    if all_values_for_ylim:
        min_val = np.min(all_values_for_ylim)
        max_val = np.max(all_values_for_ylim)
        padding = (max_val - min_val) * 0.10 
        if padding == 0:
            padding = 0.05 
        plt.ylim(max(0, min_val - padding), min(1, max_val + padding))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "temperature_vs_combined_metrics.png"))
    plt.close()

    _create_plot(y_gt_col="similarity_avg_s_vs_gt_script1",
                 y_teacher_col="similarity_avg_s_vs_t_script1",
                 title="Effect of Temperature on Average Assertion Similarity",
                 ylabel="Average Similarity",
                 filename_suffix="similarity")

    _create_plot(single_line_col="avg_loss",
                 y_gt_col=None, y_teacher_col=None,
                 title="Effect of Temperature on Average Student Loss",
                 ylabel="Average Loss",
                 filename_suffix="loss",
                 single_line_label="Overall Student Loss",
                 single_line_color='magenta')
    
    _create_plot(single_line_col="avg_inference_time_per_sample_ms",
                 y_gt_col=None, y_teacher_col=None,
                 title="Effect of Temperature on Average Inference Time",
                 ylabel="Avg. Inference Time (ms/sample)",
                 filename_suffix="inference_time",
                 single_line_label="Inference Time",
                 single_line_color='purple')

    print(f"Graphs saved to {output_dir}")

if __name__ == "__main__":
    results_df = load_all_results(RESULTS_DIR)
    if not results_df.empty:
        print("\nLoaded Data:")
        print(results_df.to_string())
        plot_metrics(results_df, OUTPUT_GRAPH_DIR)
    else:
        print(f"\nNo data loaded. Please check your RESULTS_DIR ('{RESULTS_DIR}') and JSON files.")