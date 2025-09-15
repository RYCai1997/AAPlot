"""
AAPlot Multi-File Analysis Tool - Optimized Version
用于对Spike2输出的txt文件进行多文件时间序列分析和可视化

主要改进：
1. 模块化设计，将功能分解为独立的函数
2. 配置管理，支持参数化设置
3. 错误处理增强
4. 性能优化
5. 代码可读性提升
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class PlotConfig:
    """Configuration class for plot settings and parameters"""
    # Data processing parameters
    pre_event_time: float = 0.5  # hours before event
    post_event_time: float = 2.0  # hours after event
    baseline_window: Tuple[float, float] = (100, 500)  # seconds
    
    # Smoothing parameters
    smoothing_window: int = 2001
    smoothing_polyorder: int = 2
    
    # Analysis time window (seconds)
    analysis_start: float = 300  # 5 minutes
    analysis_end: float = 2100   # 35 minutes
    
    # Plot settings
    figsize: Tuple[int, int] = (8, 3)
    colors: Dict[str, str] = None
    linewidth: Dict[str, float] = None
    font_settings: Dict[str, str] = None
    
    # Axis settings
    x_limits: Tuple[float, float] = (-0.25, 1.5)
    y_limits: Tuple[float, float] = (-5, 15)
    x_ticks: Tuple[float, float, float] = (-0.5, 2.1, 0.5)
    y_ticks: Tuple[float, float, float] = (-5, 15, 5)
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.colors is None:
            self.colors = {
                'individual': 'lightgrey',
                'mean': 'darkgoldenrod',
                'error_band': 'darkgoldenrod'
            }
        
        if self.linewidth is None:
            self.linewidth = {
                'individual': 1,
                'mean': 3,
                'vertical_line': 3
            }
        
        if self.font_settings is None:
            self.font_settings = {
                'family': 'Arial',
                'xlabel_size': 20,
                'ylabel_size': 20,
                'tick_size': 18
            }


class DataProcessor:
    """Main class for processing Spike2 data files"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.all_data: List[pd.DataFrame] = []
        self.final_df: Optional[pd.DataFrame] = None
        
    def load_data_files(self, folder_path: str) -> bool:
        """
        Load all text files from specified folder
        
        Args:
            folder_path: Path to folder containing data files
            
        Returns:
            bool: True if files loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Get all txt files and sort them
            txt_files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
            
            if not txt_files:
                raise FileNotFoundError(f"No .txt files found in {folder_path}")
            
            print(f"Found {len(txt_files)} data files")
            
            # Load all files
            self.all_data = []
            for file_path in txt_files:
                try:
                    df = pd.read_csv(file_path, sep=',')
                    if df.shape[1] < 3:
                        print(f"Warning: File {file_path} has insufficient columns")
                        continue
                    self.all_data.append(df)
                    print(f"Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            
            if not self.all_data:
                raise ValueError("No valid data files loaded")
            
            return True
            
        except Exception as e:
            print(f"Error in load_data_files: {e}")
            return False
    
    def align_data_to_events(self) -> bool:
        """
        Align all data traces to their respective event markers
        
        Returns:
            bool: True if alignment successful, False otherwise
        """
        try:
            if not self.all_data:
                raise ValueError("No data loaded. Call load_data_files first.")
            
            # Reset time to event marker for each dataset
            for i, df in enumerate(self.all_data):
                event_mask = df.iloc[:, 2] == 1
                if not event_mask.any():
                    print(f"Warning: No event markers found in dataset {i}")
                    continue
                
                event_time = df.iloc[:, 0][event_mask].iloc[0]
                df.iloc[:, 0] = df.iloc[:, 0] - event_time
            
            # Calculate time step (assuming consistent sampling rate)
            dt = self.all_data[0].iloc[1, 0] - self.all_data[0].iloc[0, 0]
            print(f"Time step: {dt} seconds")
            
            # Find event marker indices for each dataset
            time_0_indices = []
            valid_datasets = []
            
            for i, df in enumerate(self.all_data):
                event_mask = df.iloc[:, 2] == 1
                if event_mask.any():
                    time_0_index = df.iloc[:, 0][event_mask].index[0]
                    time_0_indices.append(time_0_index)
                    valid_datasets.append(i)
            
            if not valid_datasets:
                raise ValueError("No valid event markers found in any dataset")
            
            # Calculate alignment parameters
            max_pre_len = max(time_0_indices)
            max_post_len = max([len(self.all_data[i]) - idx 
                              for i, idx in zip(valid_datasets, time_0_indices)])
            
            # Create aligned data structure
            aligned_data = {}
            
            for i, (df_idx, time_0_idx) in enumerate(zip(valid_datasets, time_0_indices)):
                df = self.all_data[df_idx]
                signal = df.iloc[:, 1].values
                
                # Align pre-event data
                pre_signal = signal[:time_0_idx]
                padded_pre = np.full(max_pre_len, np.nan)
                padded_pre[-len(pre_signal):] = pre_signal
                
                # Align post-event data
                post_signal = signal[time_0_idx:]
                padded_post = np.full(max_post_len, np.nan)
                padded_post[:len(post_signal)] = post_signal
                
                # Combine aligned signal
                aligned_signal = np.concatenate([padded_pre, padded_post])
                aligned_data[f'signal_{i}'] = aligned_signal
            
            # Create time axis
            total_length = max_pre_len + max_post_len
            time_axis = np.arange(total_length) * dt / 3600  # Convert to hours
            time_axis = time_axis - time_axis[max_pre_len]  # Center at event time
            
            # Create final DataFrame
            self.final_df = pd.DataFrame({'Time(h)': time_axis})
            
            for signal_name, signal_data in aligned_data.items():
                self.final_df[signal_name] = pd.to_numeric(signal_data, errors='coerce')
            
            # Calculate statistics
            self._calculate_statistics()
            
            print(f"Successfully aligned {len(aligned_data)} signals")
            return True
            
        except Exception as e:
            print(f"Error in align_data_to_events: {e}")
            return False
    
    def _calculate_statistics(self):
        """Calculate mean, standard deviation, and standard error"""
        signal_columns = [col for col in self.final_df.columns if 'signal_' in col]
        
        if signal_columns:
            # Calculate statistics, excluding NaN values
            self.final_df['Mean'] = self.final_df[signal_columns].mean(axis=1, skipna=True)
            self.final_df['Std'] = self.final_df[signal_columns].std(axis=1, skipna=True)
            
            # Calculate standard error
            n_valid = self.final_df[signal_columns].count(axis=1)
            self.final_df['StdErr'] = self.final_df['Std'] / np.sqrt(n_valid)
            
            print("Statistics calculated successfully")
    
    def smooth_data(self) -> pd.DataFrame:
        """
        Apply Savitzky-Golay smoothing to the data
        
        Returns:
            pd.DataFrame: Smoothed data
        """
        try:
            from scipy.signal import savgol_filter
            
            if self.final_df is None:
                raise ValueError("No data to smooth. Run alignment first.")
            
            smoothed_df = self.final_df.copy()
            signal_columns = [col for col in smoothed_df.columns if 'signal_' in col]
            
            for col in signal_columns:
                # Remove NaN values for smoothing
                valid_mask = ~pd.isna(smoothed_df[col])
                if valid_mask.sum() > self.config.smoothing_window:
                    smoothed_values = savgol_filter(
                        smoothed_df.loc[valid_mask, col].values,
                        self.config.smoothing_window,
                        self.config.smoothing_polyorder,
                        mode='nearest'
                    )
                    smoothed_df.loc[valid_mask, col] = smoothed_values
            
            print("Data smoothing completed")
            return smoothed_df
            
        except ImportError:
            print("Warning: scipy not available. Returning original data.")
            return self.final_df
        except Exception as e:
            print(f"Error in smooth_data: {e}")
            return self.final_df


class PlotGenerator:
    """Class for generating publication-quality plots"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
    
    def create_main_plot(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Create the main time series plot
        
        Args:
            df: DataFrame with aligned data
            output_path: Path to save the plot
            
        Returns:
            bool: True if plot created successfully, False otherwise
        """
        try:
            # Set up the plot
            plt.figure(figsize=self.config.figsize)
            
            # Get signal columns
            signal_columns = [col for col in df.columns if 'signal_' in col]
            
            # Plot individual traces
            for col in signal_columns:
                plt.plot(df['Time(h)'], df[col], 
                        color=self.config.colors['individual'],
                        linewidth=self.config.linewidth['individual'],
                        alpha=0.7)
            
            # Plot mean trace
            if 'Mean' in df.columns:
                plt.plot(df['Time(h)'], df['Mean'],
                        color=self.config.colors['mean'],
                        linewidth=self.config.linewidth['mean'],
                        label='Mean')
            
            # Add error band (optional)
            if 'Mean' in df.columns and 'StdErr' in df.columns:
                plt.fill_between(df['Time(h)'],
                               df['Mean'] - df['StdErr'],
                               df['Mean'] + df['StdErr'],
                               color=self.config.colors['error_band'],
                               alpha=0.2)
            
            # Formatting
            self._format_plot()
            
            # Save plot
            plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Plot saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            return False
    
    def _format_plot(self):
        """Apply consistent formatting to the plot"""
        # Set font
        plt.rcParams['font.sans-serif'] = [self.config.font_settings['family']]
        
        # Labels and ticks
        plt.xlabel('Time (h)', fontsize=self.config.font_settings['xlabel_size'])
        plt.ylabel('z-score', fontsize=self.config.font_settings['ylabel_size'])
        
        # Tick settings
        x_ticks = np.arange(*self.config.x_ticks)
        y_ticks = np.arange(*self.config.y_ticks)
        plt.xticks(x_ticks, fontsize=self.config.font_settings['tick_size'])
        plt.yticks(y_ticks, fontsize=self.config.font_settings['tick_size'])
        
        # Axis limits
        plt.xlim(self.config.x_limits)
        plt.ylim(self.config.y_limits)
        
        # Add event line
        plt.axvline(0, color='black', linestyle='--', 
                   linewidth=self.config.linewidth['vertical_line'])
        
        # Remove spines and set transparent background
        plt.gcf().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        sns.despine()


class MetricsCalculator:
    """Class for calculating analysis metrics"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate peak values and area under curve
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            pd.DataFrame: Metrics for each signal
        """
        try:
            signal_columns = [col for col in df.columns if 'signal_' in col]
            
            # Create time mask for analysis window
            time_mask = ((df['Time(h)'] >= self.config.analysis_start / 3600) & 
                        (df['Time(h)'] <= self.config.analysis_end / 3600))
            
            metrics = {}
            for col in signal_columns:
                filtered_data = df.loc[time_mask, col].dropna()
                
                if len(filtered_data) > 0:
                    # Calculate metrics
                    max_val = filtered_data.max()
                    min_val = filtered_data.min()
                    
                    # Calculate AUC using trapezoidal integration
                    time_points = df.loc[time_mask, 'Time(h)'].values
                    valid_time_mask = ~pd.isna(df.loc[time_mask, col])
                    
                    if valid_time_mask.sum() > 1:
                        valid_time = time_points[valid_time_mask]
                        valid_data = filtered_data.values
                        auc = np.trapz(valid_data, valid_time)
                    else:
                        auc = np.nan
                    
                    metrics[col] = {
                        'max': max_val,
                        'min': min_val,
                        'auc': auc
                    }
                else:
                    metrics[col] = {'max': np.nan, 'min': np.nan, 'auc': np.nan}
            
            metrics_df = pd.DataFrame(metrics).T
            return metrics_df
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return pd.DataFrame()


def main():
    """Main execution function"""
    # Configuration
    config = PlotConfig()
    
    # Set your data folder path here
    folder_path = r'D:\Science\Experiment data\eCB sensors\Selective sensor\FiberPhotometry\Drug intake\NAcSh_All_data_in_1_folder\01_Summary\2-AG\cocaine'
    
    # Initialize processor
    processor = DataProcessor(config)
    
    # Load and process data
    if not processor.load_data_files(folder_path):
        print("Failed to load data files")
        return
    
    if not processor.align_data_to_events():
        print("Failed to align data")
        return
    
    # Smooth data
    smoothed_df = processor.smooth_data()
    
    # Generate plot
    plot_generator = PlotGenerator(config)
    plot_path = os.path.join(folder_path, 'optimized_plot.svg')
    plot_generator.create_main_plot(processor.final_df, plot_path)
    
    # Calculate metrics
    metrics_calculator = MetricsCalculator(config)
    metrics_df = metrics_calculator.calculate_metrics(smoothed_df)
    
    # Save metrics
    metrics_path = os.path.join(folder_path, 'optimized_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    # Display results
    print("\nMetrics Summary:")
    print(metrics_df)
    
    # Save aligned data (optional)
    data_path = os.path.join(folder_path, 'optimized_aligned_data.csv')
    processor.final_df.to_csv(data_path, index=False)
    print(f"Aligned data saved to: {data_path}")


if __name__ == "__main__":
    main()
