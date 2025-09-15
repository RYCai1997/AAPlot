"""
Optimized Fluorescence Signal Analyzer
Enhanced version of the original fluorescence analysis script with improved error handling,
modularity, and performance.

主要改进：
1. 面向对象设计，提高代码可维护性
2. 增强的错误处理和输入验证
3. 配置管理，支持参数化设置
4. 性能优化和内存管理
5. 详细的日志记录和进度显示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class AnalysisConfig:
    """Configuration class for fluorescence analysis parameters"""
    # Event detection parameters
    pre_event_time: float = 20.0  # seconds before event
    post_event_time: float = 120.0  # seconds after event
    
    # Baseline calculation parameters
    baseline_start_time: float = 100.0  # seconds
    baseline_end_time: float = 500.0  # seconds
    
    # Smoothing parameters (if needed)
    smoothing_window: int = 11  # Savitzky-Golay window size
    smoothing_polyorder: int = 3  # Savitzky-Golay polynomial order
    
    # Output settings
    output_dir: str = './output_traces'
    save_individual_traces: bool = False
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    # Plot settings
    plot_figsize: Tuple[int, int] = (10, 6)
    plot_colors: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.plot_colors is None:
            self.plot_colors = {
                'individual_trace': 'lightgrey',
                'averaged_trace': 'red',
                'baseline': 'blue'
            }


class FluorescenceAnalyzer:
    """Main class for fluorescence signal analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_logging()
        self.raw_data: Optional[pd.DataFrame] = None
        self.baseline_subtracted_traces: List[np.ndarray] = []
        self.z_score_traces: List[np.ndarray] = []
        self.sampling_rate: Optional[float] = None
        self.pre_event_samples: Optional[int] = None
        self.time_axis: Optional[np.ndarray] = None
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> bool:
        """
        Load fluorescence data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"Loading data from: {file_path}")
            
            # Load data with error handling
            self.raw_data = pd.read_csv(file_path)
            
            # Validate data structure
            if self.raw_data.shape[1] < 3:
                raise ValueError(f"Expected at least 3 columns, got {self.raw_data.shape[1]}")
            
            if len(self.raw_data) == 0:
                raise ValueError("Data file is empty")
            
            # Extract columns (assuming: Time, dF/F0, Event)
            self.time_data = self.raw_data.iloc[:, 0]
            self.signal_data = self.raw_data.iloc[:, 1]
            self.event_data = self.raw_data.iloc[:, 2]
            
            self.logger.info(f"Loaded {len(self.raw_data)} data points")
            self.logger.info(f"Time range: {self.time_data.min():.2f} - {self.time_data.max():.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def detect_events(self) -> Tuple[np.ndarray, int]:
        """
        Detect event markers in the data
        
        Returns:
            Tuple of (event_indices, number_of_events)
        """
        try:
            event_indices = self.event_data[self.event_data == 1].index.to_numpy()
            num_events = len(event_indices)
            
            self.logger.info(f"Found {num_events} event markers")
            
            if num_events == 0:
                self.logger.warning("No event markers found in the data")
            
            return event_indices, num_events
            
        except Exception as e:
            self.logger.error(f"Error detecting events: {e}")
            return np.array([]), 0
    
    def calculate_sampling_rate(self) -> bool:
        """
        Calculate sampling rate from time data
        
        Returns:
            bool: True if sampling rate calculated successfully
        """
        try:
            time_np = self.time_data.to_numpy()
            
            if len(time_np) < 2:
                raise ValueError("Insufficient data points to calculate sampling rate")
            
            # Calculate time differences
            time_diffs = np.diff(time_np)
            
            # Check for consistent sampling
            if np.std(time_diffs) / np.mean(time_diffs) > 0.1:
                self.logger.warning("Inconsistent sampling rate detected")
            
            self.sampling_rate = float(np.mean(time_diffs))
            self.logger.info(f"Sampling rate: {self.sampling_rate:.4f} seconds per sample")
            
            # Calculate pre-event samples
            self.pre_event_samples = int(self.config.pre_event_time / self.sampling_rate)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error calculating sampling rate: {e}")
            return False
    
    def extract_traces(self, event_indices: np.ndarray) -> List[np.ndarray]:
        """
        Extract traces around event markers
        
        Args:
            event_indices: Array of event marker indices
            
        Returns:
            List of extracted traces
        """
        try:
            all_traces = []
            post_event_samples = int(self.config.post_event_time / self.sampling_rate)
            
            for i, marker_index in enumerate(event_indices):
                start_index = marker_index - self.pre_event_samples
                end_index = marker_index + post_event_samples
                
                # Check bounds
                if start_index >= 0 and end_index <= len(self.signal_data):
                    trace = self.signal_data.iloc[start_index:end_index].values
                    all_traces.append(trace)
                    self.logger.debug(f"Extracted trace {i+1}: {len(trace)} points")
                else:
                    marker_time = self.time_data.iloc[marker_index]
                    self.logger.warning(f"Skipping event at {marker_time:.2f}s - out of bounds")
            
            self.logger.info(f"Successfully extracted {len(all_traces)} traces")
            return all_traces
            
        except Exception as e:
            self.logger.error(f"Error extracting traces: {e}")
            return []
    
    def calculate_baseline(self) -> float:
        """
        Calculate real baseline from specified time window
        
        Returns:
            float: Baseline value
        """
        try:
            time_np = self.time_data.to_numpy()
            
            # Find baseline window indices
            start_idx = np.searchsorted(time_np, self.config.baseline_start_time, side='left')
            end_idx = np.searchsorted(time_np, self.config.baseline_end_time, side='left')
            
            if start_idx >= end_idx:
                raise ValueError("Invalid baseline window")
            
            baseline_data = self.signal_data.iloc[start_idx:end_idx]
            baseline = float(np.mean(baseline_data))
            
            self.logger.info(f"Calculated baseline: {baseline:.6f} "
                           f"(from {self.config.baseline_start_time}s to {self.config.baseline_end_time}s)")
            
            return baseline
            
        except Exception as e:
            self.logger.error(f"Error calculating baseline: {e}")
            return 0.0
    
    def baseline_subtract_traces(self, traces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform baseline subtraction on traces
        
        Args:
            traces: List of raw traces
            
        Returns:
            List of baseline-subtracted traces
        """
        try:
            real_baseline = self.calculate_baseline()
            baseline_subtracted = []
            
            for i, trace in enumerate(traces):
                # Calculate event baseline (first part of trace)
                event_baseline = np.mean(trace[:self.pre_event_samples])
                delta_baseline = event_baseline - real_baseline
                
                # Subtract baseline difference
                corrected_trace = trace - delta_baseline
                baseline_subtracted.append(corrected_trace)
                
                self.logger.debug(f"Trace {i+1}: event_baseline={event_baseline:.6f}, "
                                f"delta={delta_baseline:.6f}")
            
            self.logger.info("Baseline subtraction completed")
            return baseline_subtracted
            
        except Exception as e:
            self.logger.error(f"Error in baseline subtraction: {e}")
            return traces
    
    def calculate_z_scores(self, traces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate z-scores for traces
        
        Args:
            traces: List of baseline-subtracted traces
            
        Returns:
            List of z-score normalized traces
        """
        try:
            z_score_traces = []
            
            for i, trace in enumerate(traces):
                mean_trace = np.mean(trace)
                std_trace = np.std(trace)
                
                if std_trace != 0:
                    z_score = (trace - mean_trace) / std_trace
                else:
                    z_score = np.zeros_like(trace)
                    self.logger.warning(f"Zero standard deviation for trace {i+1}")
                
                z_score_traces.append(z_score)
            
            self.logger.info("Z-score calculation completed")
            return z_score_traces
            
        except Exception as e:
            self.logger.error(f"Error calculating z-scores: {e}")
            return traces
    
    def create_time_axis(self) -> np.ndarray:
        """
        Create time axis for traces
        
        Returns:
            np.ndarray: Time axis in seconds
        """
        try:
            if self.pre_event_samples is None or self.sampling_rate is None:
                raise ValueError("Sampling parameters not calculated")
            
            if not self.baseline_subtracted_traces:
                raise ValueError("No traces available")
            
            trace_length = len(self.baseline_subtracted_traces[0])
            self.time_axis = (np.arange(trace_length) - self.pre_event_samples) * self.sampling_rate
            
            self.logger.info(f"Created time axis: {self.time_axis[0]:.2f} to {self.time_axis[-1]:.2f} seconds")
            return self.time_axis
            
        except Exception as e:
            self.logger.error(f"Error creating time axis: {e}")
            return np.array([])
    
    def save_traces_to_csv(self, traces: List[np.ndarray], 
                          trace_type: str, time_axis: np.ndarray) -> bool:
        """
        Save traces to CSV files
        
        Args:
            traces: List of traces to save
            trace_type: Type of traces ('dff0' or 'zscore')
            time_axis: Time axis for the traces
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if not traces:
                self.logger.warning("No traces to save")
                return False
            
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save merged traces
            merged_df = pd.DataFrame({'Time': time_axis})
            for i, trace in enumerate(traces):
                merged_df[f'signal {i+1}'] = trace
            
            output_file = os.path.join(self.config.output_dir, f'merged_{trace_type}_traces.csv')
            merged_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved merged {trace_type} traces to: {output_file}")
            
            # Save individual traces if requested
            if self.config.save_individual_traces:
                for i, trace in enumerate(traces):
                    individual_df = pd.DataFrame({
                        'Time': time_axis,
                        trace_type: trace
                    })
                    individual_file = os.path.join(self.config.output_dir, f'{trace_type}_trace_{i+1}.csv')
                    individual_df.to_csv(individual_file, index=False)
                
                self.logger.info(f"Saved {len(traces)} individual {trace_type} trace files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving traces: {e}")
            return False
    
    def create_plot(self, traces: List[np.ndarray], time_axis: np.ndarray, 
                   trace_type: str) -> bool:
        """
        Create and save plot of traces
        
        Args:
            traces: List of traces to plot
            time_axis: Time axis
            trace_type: Type of traces for labeling
            
        Returns:
            bool: True if plot created successfully
        """
        try:
            if not self.config.save_plots or not traces:
                return True
            
            plt.figure(figsize=self.config.plot_figsize)
            
            # Plot individual traces
            for trace in traces:
                plt.plot(time_axis, trace, 
                        color=self.config.plot_colors['individual_trace'],
                        alpha=0.7, linewidth=1)
            
            # Calculate and plot averaged trace
            if len(traces) > 1:
                averaged_trace = np.mean(traces, axis=0)
                plt.plot(time_axis, averaged_trace,
                        color=self.config.plot_colors['averaged_trace'],
                        linewidth=2, label='Averaged Trace')
                
                plt.legend()
            
            # Formatting
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel(trace_type, fontsize=12)
            plt.title('Fluorescent Signal Traces', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_file = os.path.join(self.config.output_dir, f'fluorescent_{trace_type}_plot.{self.config.plot_format}')
            plt.savefig(output_file, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved plot to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
            return False
    
    def run_analysis(self, file_path: str) -> bool:
        """
        Run complete fluorescence analysis pipeline
        
        Args:
            file_path: Path to input CSV file
            
        Returns:
            bool: True if analysis completed successfully
        """
        try:
            self.logger.info("Starting fluorescence analysis")
            
            # Load data
            if not self.load_data(file_path):
                return False
            
            # Detect events
            event_indices, num_events = self.detect_events()
            if num_events == 0:
                self.logger.error("No events found - cannot proceed")
                return False
            
            # Calculate sampling rate
            if not self.calculate_sampling_rate():
                return False
            
            # Extract traces
            raw_traces = self.extract_traces(event_indices)
            if not raw_traces:
                self.logger.error("No traces extracted")
                return False
            
            # Baseline subtraction
            self.baseline_subtracted_traces = self.baseline_subtract_traces(raw_traces)
            
            # Z-score calculation
            self.z_score_traces = self.calculate_z_scores(self.baseline_subtracted_traces)
            
            # Create time axis
            time_axis = self.create_time_axis()
            
            # Save results
            self.save_traces_to_csv(self.baseline_subtracted_traces, 'dff0', time_axis)
            self.save_traces_to_csv(self.z_score_traces, 'zscore', time_axis)
            
            # Create plots
            self.create_plot(self.baseline_subtracted_traces, time_axis, 'dF/F0')
            self.create_plot(self.z_score_traces, time_axis, 'z-score')
            
            self.logger.info("Analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return False


def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Optimized Fluorescence Signal Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimized_fluorescence_analyzer.py data.csv
  python optimized_fluorescence_analyzer.py data.csv --output-dir ./results
  python optimized_fluorescence_analyzer.py data.csv --pre-event 30 --post-event 150
        """
    )
    
    parser.add_argument('file_path', help='Path to input CSV file')
    parser.add_argument('--output-dir', default='./output_traces',
                       help='Output directory (default: ./output_traces)')
    parser.add_argument('--pre-event', type=float, default=20.0,
                       help='Time before event in seconds (default: 20.0)')
    parser.add_argument('--post-event', type=float, default=120.0,
                       help='Time after event in seconds (default: 120.0)')
    parser.add_argument('--baseline-start', type=float, default=100.0,
                       help='Baseline calculation start time (default: 100.0)')
    parser.add_argument('--baseline-end', type=float, default=500.0,
                       help='Baseline calculation end time (default: 500.0)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual trace files')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig(
        pre_event_time=args.pre_event,
        post_event_time=args.post_event,
        baseline_start_time=args.baseline_start,
        baseline_end_time=args.baseline_end,
        output_dir=args.output_dir,
        save_individual_traces=args.save_individual,
        save_plots=not args.no_plots
    )
    
    # Run analysis
    analyzer = FluorescenceAnalyzer(config)
    success = analyzer.run_analysis(args.file_path)
    
    if success:
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print(f"Found {len(analyzer.baseline_subtracted_traces)} valid traces")
    else:
        print("\nAnalysis failed. Check the log messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
