# Batch processing for Spike2 `.txt` exports
#
# Reads every text file in an input directory, aligns each trace to the
# marker (column 3), and creates:
#   • combined aligned DataFrame (optional)
#   • summary metrics CSV (peak mean, auc, max/min, etc.)
#   • a time‑series plot of raw traces + mean
#
# Run with:
#    python fiber_trace_spike2_analysis_batch.py <input_folder> [--output <folder>]
#
# Input files are comma‑separated spreadsheets from Spike2, three columns:
# time, value, marker.  The script has been exercised in a conda env named
# "PLOT" (python3.12/3.13 + pandas numpy matplotlib seaborn).

# ------------------------------------- By Ruyi Cai @ Yulong Li Lab, PKU, China ----------------------------------

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


# -------------------------- user configuration --------------------------
# You may edit the values below directly or supply replacements via
# command‑line arguments (see argument parser in `main`).
#
# 1. `folder_path`: directory containing Spike2 ``.txt`` exports.
folder_path = r'D:\Temp\test_fiber\1 Nicotine 2mpk'  # change to your default input directory

# -------------------------- plot configuration --------------------------
# These values govern the appearance of the generated SVG plot.
# Change them here or add CLI options if desired.

# axis limits (min, max)
x_axis_limits = (-0.5, 2) 
y_axis_limits = (-4, 20)

# average trace appearance
avg_trace_color = 'darkgoldenrod'   #'steelblue' for AEA, 'darkgoldenrod' for 2-AG
avg_trace_width = 3     #default 3

# single (individual) trace appearance
single_trace_color = 'lightgrey'
single_trace_width = 1      #default 1

# tick interval for each axis 
x_tick_interval = 1
y_tick_interval = 10

# -------------------------- analysis window settings --------------------------
# Time ranges (in seconds after event) used for peak/AUC calculations.
# These defaults can be overridden with ``--peak-window`` and ``--auc-window``.
peak_window = (900, 1200)
auc_window = (300, 2100)


# -------------------------------------------------------------------------------
# Core processing functions follow; editing is generally unnecessary.
# -------------------------------------------------------------------------------

# -------------------------- processing code --------------------------

def smooth_data(df: pd.DataFrame, window_length=2001, polyorder=2) -> pd.DataFrame:
    """Return a copy of ``df`` with signal columns smoothed.

    Only the columns between the time column and the last three statistic
    columns are smoothed (mirrors notebook behaviour).
    """
    smoothed = df.copy()
    for col in df.columns[1:-3]:
        smoothed[col] = savgol_filter(df[col], window_length, polyorder, mode='nearest')
    return smoothed


def calculate_metrics(
    df: pd.DataFrame,
    peak_window: tuple[float, float] = (900, 1200), # default 900-1200s (15-20min)
    auc_window: tuple[float, float] = (300, 2100),  # default 300-2100s (5-35min, i.e. 20min window centered on peak_window)
) -> pd.DataFrame:
    """Compute a collection of statistics for each signal column.

    The following quantities are returned:

    * ``max`` and ``min`` over the interval from time 3 min to the right
      edge of the x-axis limits (ignores ``auc_window``)
    * ``area`` (signed trapezoidal integral over ``auc_window``)
    * ``peak_mean`` the average signal during ``peak_window``
    * ``auc`` the signed area during ``auc_window`` (defaulting to
      ``peak_window``)

    In addition, four extra columns record the window boundaries so that the
    output file documents the time ranges used for ``peak`` and ``auc``.

    Parameters
    ----------
    df
        DataFrame produced by ``align_and_compute`` (time in hours).
    peak_window
        Start/end seconds over which the ``peak_mean`` is computed.
    auc_window
        Start/end seconds used for the AUC calculation.  If ``None`` the
        ``peak_window`` is reused.
    """
    # if auc_window omitted, fall back to peak_window for all metrics
    if auc_window is None:
        auc_window = peak_window

    pstart, pend = peak_window
    astart, aend = auc_window

    mask_p = (df['Time(h)'] >= pstart / 3600) & (df['Time(h)'] <= pend / 3600)
    mask_a = (df['Time(h)'] >= astart / 3600) & (df['Time(h)'] <= aend / 3600)

    # mask for max/min: from 0.05h (3 min) to right x-axis limit
    mask_mm = (df['Time(h)'] >= 0.05) & (df['Time(h)'] <= x_axis_limits[1])

    metrics: dict[str, dict] = {}
    for col in df.columns[1:-3]:
        data_p = df.loc[mask_p, col]
        data_a = df.loc[mask_a, col]
        data_mm = df.loc[mask_mm, col]

        entry: dict[str, float] = {}
        if not data_mm.empty:
            entry['max'] = data_mm.max()
            entry['min'] = data_mm.min()
        else:
            entry['max'] = np.nan
            entry['min'] = np.nan
        if not data_a.empty:
            entry['area'] = np.trapezoid(data_a, dx=1/3600)
        else:
            entry['area'] = np.nan

        entry['peak_mean'] = data_p.mean() if not data_p.empty else np.nan
        entry['auc'] = np.trapezoid(data_a, dx=1/3600) if not data_a.empty else np.nan

        metrics[col] = entry

    result = pd.DataFrame(metrics).T
    # add window information for reference
    result['peak_start'] = pstart
    result['peak_end'] = pend
    result['auc_start'] = astart
    result['auc_end'] = aend
    return result


def read_all_txt(folder_path: str) -> tuple[list[pd.DataFrame], list[str]]:
    """Read every .txt file in ``folder_path``.

    Returns a pair ``(dataframes, paths)`` where both lists are sorted by
    filename so processing order is stable.
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    txt_files.sort()
    data = []
    for fn in txt_files:
        df = pd.read_csv(fn, sep=',')
        data.append(df)
    return data, txt_files


def align_and_compute(all_data: list[pd.DataFrame]) -> pd.DataFrame:
    """Align a list of dataframes on the marker event and compute statistics.

    Parameters
    ----------
    all_data
        List of DataFrames each containing at least three columns as described
        above.  The function assumes the first column is time, the second is the
        signal, and the third is the marker.

    Returns
    -------
    pd.DataFrame
        A new dataframe whose first column is the aligned time axis (hours)
        and the remaining columns are the individual signals plus derived
        columns ``Mean``, ``Std``, and ``StdErr``.
    """
    # adjust time so that marker event is at zero for each file
    for i, df in enumerate(all_data):
        # subtract the time of the first marker == 1
        event_time = df.iloc[:, 0][df.iloc[:, 2] == 1].values[0]
        all_data[i].iloc[:, 0] = all_data[i].iloc[:, 0] - event_time
    # estimate dt from the first file (assume consistent sampling)
    dt = all_data[0].iloc[1, 0] - all_data[0].iloc[0, 0]

    # determine index of time=0 in each dataframe
    time0_indices = [df.iloc[:, 0][df.iloc[:, 2] == 1].index[0]
                     for df in all_data]

    max_pre = max(time0_indices)
    max_post = max(len(df) - idx for df, idx in zip(all_data, time0_indices))

    aligned = pd.DataFrame()
    for i, df in enumerate(all_data):
        signal = df.iloc[:, 1]
        pre = signal[:time0_indices[i]].tolist()
        post = signal[time0_indices[i]:].tolist()
        padded_pre = [np.nan] * (max_pre - len(pre)) + pre
        padded_post = post + [np.nan] * (max_post - len(post))
        aligned[f'signal_{i}'] = padded_pre + padded_post

    pre_time = np.linspace(-max_pre * dt/3600, 0, max_pre, endpoint=False)
    post_time = np.linspace(0, max_post * dt/3600, max_post, endpoint=False)[1:]
    time_axis = np.concatenate([pre_time, [0], post_time])

    final = pd.DataFrame({'Time(h)': time_axis})
    for col in aligned.columns:
        final[col] = pd.to_numeric(aligned[col], errors='coerce')
    final['Mean'] = final.iloc[:, 1:].mean(axis=1, skipna=True)
    final['Std'] = final.iloc[:, 1:-1].std(axis=1, skipna=True)
    n = final.iloc[:, 1:-2].count(axis=1)
    final['StdErr'] = final['Std'] / np.sqrt(n)
    return final


def plot_results(final_df: pd.DataFrame, folder_path: str) -> None:
    """Generate and save a time series plot from the aligned dataframe."""
    plt.figure(figsize=(8, 3))
    signal_cols = [c for c in final_df.columns if c.startswith('signal_')]

    # draw individual traces first
    for col in signal_cols:
        plt.plot(final_df['Time(h)'], final_df[col],
                 color=single_trace_color, linewidth=single_trace_width)
    # draw the average trace on top
    plt.plot(final_df['Time(h)'], final_df['Mean'],
             color=avg_trace_color, linewidth=avg_trace_width, label='Mean')

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.xlabel('Time (h)', fontsize=20)
    plt.ylabel('z-score', fontsize=20)

    # apply axis limits from configuration
    plt.xlim(*x_axis_limits)
    plt.ylim(*y_axis_limits)

    # compute ticks based on interval values
    def make_ticks(lim, interval):
        # generate tick positions centered on zero.  We walk outward in both
        # directions from 0 using the given spacing, then trim to the axis
        # limits.  If zero is outside the limits we fall back to the simple
        # behaviour used previously (start at lower bound).
        if interval is None or interval <= 0:
            return [lim[0], lim[1]]
        low, high = lim
        if low <= 0 <= high:
            ticks = [0]
            cur = 0
            # positive side
            while True:
                cur += interval
                if cur <= high:
                    ticks.append(cur)
                else:
                    break
            # negative side
            cur = 0
            while True:
                cur -= interval
                if cur >= low:
                    ticks.append(cur)
                else:
                    break
            return sorted(set(ticks))
        else:
            # zero not in range: fall back to linear spacing from low
            ticks = [low]
            current = low
            while True:
                current += interval
                if current >= high:
                    ticks.append(high)
                    break
                ticks.append(current)
            return ticks

    xt = make_ticks(x_axis_limits, x_tick_interval)
    yt = make_ticks(y_axis_limits, y_tick_interval)

    plt.xticks(xt, fontsize=18)
    plt.yticks(yt, fontsize=18)

    plt.axvline(0, color='black', linestyle='--', linewidth=3)
    plt.gcf().patch.set_alpha(0.0)
    plt.gca().patch.set_alpha(0.0)
    sns.despine()

    out_fig = os.path.join(folder_path, 'plot.svg')
    plt.savefig(out_fig, format='svg', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Batch analyse Spike2 txt files.')
    # the default comes from the configuration variable above; users may edit
    # that or supply a different path on the command line.
    parser.add_argument('input_dir', nargs='?', default=folder_path,
                        help=f'Directory containing .txt files (default: {folder_path})')
    parser.add_argument('--output', '-o', help='Directory to write outputs (default: input_dir)',
                        default=None)
    parser.add_argument('--noplot', action='store_true',
                        help='Do not generate the summary plot')
    parser.add_argument('--peak-window', nargs=2, type=float,
                        metavar=('START','END'),
                        default=list(globals()['peak_window']),
                        help='seconds range for computing peak mean')
    parser.add_argument('--auc-window', nargs=2, type=float,
                        metavar=('START','END'),
                        default=list(globals()['auc_window']) if globals()['auc_window'] is not None else None,
                        help='seconds range for AUC (also used for max/min/area)')
    args = parser.parse_args()

    # determine which folder to process
    folder = args.input_dir or folder_path
    outdir = args.output or folder
    os.makedirs(outdir, exist_ok=True)

    dfs, paths = read_all_txt(folder)
    if not dfs:
        print('No .txt files found in', folder)
        return

    final_df = align_and_compute(dfs)

    # save aligned data (disabled for now; uncomment if you need the file)
    # aligned_file = os.path.join(outdir, 'aligned_data.csv')
    # final_df.to_csv(aligned_file, index=False)

    # compute smooth version and metrics
    smoothed = smooth_data(final_df)

    peak_window = tuple(args.peak_window)
    auc_window = tuple(args.auc_window) if args.auc_window is not None else None

    metrics_df = calculate_metrics(smoothed,
                                   peak_window=peak_window,
                                   auc_window=auc_window)
    # drop helper columns documenting window bounds before writing
    # insert numbering based on file names (before the underscore, first 4 chars)
    ids = [os.path.basename(p).split('_')[0][:4] for p in paths]
    metrics_df.insert(0, 'No.', ids)

    metrics_df_to_save = metrics_df.drop(columns=['peak_start', 'peak_end', 'auc_start', 'auc_end'], errors='ignore')
    # remove the signal_* index column now that 'No.' identifies rows
    metrics_df_to_save = metrics_df_to_save.reset_index(drop=True)

    metrics_file = os.path.join(outdir, 'metrics.csv')
    metrics_df_to_save.to_csv(metrics_file, index=False)
    print('Metrics:')
    print(metrics_df_to_save)

    if not args.noplot:
        plot_results(final_df, outdir)

    print(f'Processed {len(dfs)} files. Results written to {outdir}')


if __name__ == '__main__':
    main()
