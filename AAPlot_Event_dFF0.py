
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def analyze_fluorescence(file_path):
    df = pd.read_csv(file_path)

    # Assuming column order: Time, dF/F0, Event
    time = df.iloc[:, 0]
    dff0 = df.iloc[:, 1]
    event = df.iloc[:, 2]

    # Use event indices directly to avoid float time equality issues
    event_indices = event[event == 1].index.to_numpy()
    print(f"Found {len(event_indices)} event markers.")

    # Define trace collection window
    pre_event_time = 20  # seconds before event
    post_event_time = 120  # seconds after event
    # Compute time step (seconds per sample); keep variable name for compatibility
    time_np = time.to_numpy()
    sampling_rate = float(np.diff(time_np).mean()) # seconds per sample (dt)
    
    pre_event_samples = int(pre_event_time / sampling_rate)
    post_event_samples = int(post_event_time / sampling_rate)

    all_traces = []
    for marker_index in event_indices:
        start_index = marker_index - pre_event_samples
        end_index = marker_index + post_event_samples

        # Ensure indices are within bounds (end is exclusive for iloc)
        if start_index >= 0 and end_index <= len(dff0):
            trace = dff0.iloc[start_index:end_index].values
            all_traces.append(trace)
        else:
            marker_time = time.iloc[marker_index]
            print(f"Skipping event at {marker_time}s due to out of bounds trace.")

    # Calculate real baseline (averaging signal between 100-500s)
    real_baseline_start_time = 100 # seconds
    real_baseline_end_time = 500 # seconds

    # Use searchsorted for robust baseline window indexing
    real_baseline_start_index = int(np.searchsorted(time_np, real_baseline_start_time, side='left')) if len(time_np) else 0
    real_baseline_end_index = int(np.searchsorted(time_np, real_baseline_end_time, side='left')) if len(time_np) else len(time_np)

    real_baseline = np.mean(dff0.iloc[real_baseline_start_index:real_baseline_end_index])
    print(f"Calculated real baseline: {real_baseline}")

    baseline_subtracted_traces = []
    for trace in all_traces:
        # Calculate event baseline (first 20 seconds of each trace)
        event_baseline = np.mean(trace[:pre_event_samples])
        delta_baseline = event_baseline - real_baseline
        baseline_subtracted_traces.append(trace - delta_baseline)

    # Calculate z-score for each trace
    z_score_traces = []
    for trace in baseline_subtracted_traces:
        mean_trace = np.mean(trace)
        std_trace = np.std(trace)
        if std_trace != 0:
            z_score_traces.append((trace - mean_trace) / std_trace)
        else:
            z_score_traces.append(np.zeros_like(trace)) # Avoid division by zero

    return baseline_subtracted_traces, z_score_traces, sampling_rate, pre_event_samples

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_fluorescence.py <path_to_csv_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = './output_traces'
    os.makedirs(output_dir, exist_ok=True)

    baseline_subtracted_traces, z_score_traces, sampling_rate, pre_event_samples = analyze_fluorescence(file_path)
    print(f"Collected {len(baseline_subtracted_traces)} baseline-subtracted traces.")
    print(f"Collected {len(z_score_traces)} z-score traces.")

    # Reconstruct time axis for traces (only if we have traces)
    if baseline_subtracted_traces:
        trace_length = len(baseline_subtracted_traces[0])
        time_axis = (np.arange(trace_length) - pre_event_samples) * sampling_rate

    # Merge all dF/F0 traces into one CSV file
    if baseline_subtracted_traces:
        merged_dff0_df = pd.DataFrame({'Time': time_axis})
        for i, trace in enumerate(baseline_subtracted_traces):
            merged_dff0_df[f'signal {i+1}'] = trace
        merged_dff0_df.to_csv(os.path.join(output_dir, 'merged_dff0_traces.csv'), index=False)
        print("Saved merged_dff0_traces.csv")

    # # Output each dF/F0 trace to a CSV file (individual files)
    # for i, trace in enumerate(baseline_subtracted_traces):
    #     output_df = pd.DataFrame(trace, columns=['dF/F0_subtracted'])
    #     output_df.to_csv(os.path.join(output_dir, f'dff0_trace_{i+1}.csv'), index=False)
    #     # print(f"Saved dff0_trace_{i+1}.csv") # Commented out to avoid redundant messages

    # # Output each z-score trace to a CSV file
    # for i, trace in enumerate(z_score_traces):
    #     output_df = pd.DataFrame(trace, columns=['z_score'])
    #     output_df.to_csv(os.path.join(output_dir, f'zscore_trace_{i+1}.csv'), index=False)
    #     # print(f"Saved zscore_trace_{i+1}.csv") # Commented out to avoid redundant messages

    # Plotting
    if baseline_subtracted_traces:
        plt.figure(figsize=(10, 6))
        for trace in baseline_subtracted_traces:
            plt.plot(time_axis, trace, color='lightgrey', alpha=0.7)

        # Calculate and plot averaged trace
        averaged_trace = np.mean(baseline_subtracted_traces, axis=0)
        plt.plot(time_axis, averaged_trace, color='red', linewidth=2, label='Averaged Trace')

        plt.xlabel('Time (s)')
        plt.ylabel('dF/F0')
        plt.title('Fluorescent Signal Traces')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'fluorescent_traces_plot.png'))
        print("Saved fluorescent_traces_plot.png")


