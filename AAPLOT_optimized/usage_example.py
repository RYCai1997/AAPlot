"""
AAPlot 优化版本使用示例
演示如何使用优化后的代码进行数据分析
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def example_multi_file_analysis():
    """多文件分析示例"""
    print("=== 多文件分析示例 ===")
    
    try:
        from optimized_aplot_notebook import DataProcessor, PlotGenerator, MetricsCalculator, PlotConfig
        
        # 创建配置
        config = PlotConfig()
        config.folder_path = r"D:\Science\Experiment data\eCB sensors\Selective sensor\FiberPhotometry\Drug intake\NAcSh_All_data_in_1_folder\01_Summary\2-AG\cocaine"
        
        print(f"配置文件夹路径: {config.folder_path}")
        
        # 检查文件夹是否存在
        if not os.path.exists(config.folder_path):
            print(f"警告: 文件夹不存在 - {config.folder_path}")
            print("请修改 config.folder_path 为实际的数据文件夹路径")
            return
        
        # 初始化处理器
        processor = DataProcessor(config)
        
        # 加载数据
        print("正在加载数据文件...")
        if processor.load_data_files(config.folder_path):
            print("✓ 数据加载成功")
        else:
            print("✗ 数据加载失败")
            return
        
        # 数据对齐
        print("正在对齐数据...")
        if processor.align_data_to_events():
            print("✓ 数据对齐成功")
        else:
            print("✗ 数据对齐失败")
            return
        
        # 生成图表
        print("正在生成图表...")
        plot_generator = PlotGenerator(config)
        plot_path = os.path.join(config.folder_path, 'optimized_plot.svg')
        
        if plot_generator.create_main_plot(processor.final_df, plot_path):
            print(f"✓ 图表已保存到: {plot_path}")
        else:
            print("✗ 图表生成失败")
        
        # 计算指标
        print("正在计算统计指标...")
        metrics_calculator = MetricsCalculator(config)
        metrics_df = metrics_calculator.calculate_metrics(processor.final_df)
        
        # 保存指标
        metrics_path = os.path.join(config.folder_path, 'optimized_metrics.csv')
        metrics_df.to_csv(metrics_path)
        print(f"✓ 指标已保存到: {metrics_path}")
        
        # 显示结果摘要
        print("\n=== 分析结果摘要 ===")
        print(f"数据点数量: {len(processor.final_df)}")
        print(f"信号数量: {len([col for col in processor.final_df.columns if 'signal_' in col])}")
        print(f"时间范围: {processor.final_df['Time(h)'].min():.2f} - {processor.final_df['Time(h)'].max():.2f} 小时")
        
        print("\n统计指标:")
        print(metrics_df.head())
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所需依赖: pandas, numpy, matplotlib, seaborn, scipy")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")

def example_fluorescence_analysis():
    """荧光信号分析示例"""
    print("\n=== 荧光信号分析示例 ===")
    
    try:
        from optimized_fluorescence_analyzer import FluorescenceAnalyzer, AnalysisConfig
        
        # 创建配置
        config = AnalysisConfig()
        config.output_dir = './example_output'
        config.save_individual_traces = True
        
        print(f"输出目录: {config.output_dir}")
        
        # 示例文件路径（请替换为实际文件）
        example_file = "example_data.csv"
        
        if not os.path.exists(example_file):
            print(f"警告: 示例文件不存在 - {example_file}")
            print("请提供实际的CSV数据文件路径")
            return
        
        # 初始化分析器
        analyzer = FluorescenceAnalyzer(config)
        
        # 运行分析
        print("正在分析荧光信号...")
        if analyzer.run_analysis(example_file):
            print("✓ 荧光信号分析完成")
            print(f"✓ 结果保存在: {config.output_dir}")
            print(f"✓ 处理了 {len(analyzer.baseline_subtracted_traces)} 个信号轨迹")
        else:
            print("✗ 荧光信号分析失败")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所需依赖")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")

def show_configuration_options():
    """显示配置选项"""
    print("\n=== 配置选项说明 ===")
    
    print("\n1. 多文件分析配置 (PlotConfig):")
    print("   - folder_path: 数据文件夹路径")
    print("   - pre_event_time: 事件前时间（小时）")
    print("   - post_event_time: 事件后时间（小时）")
    print("   - smoothing_window: 平滑窗口大小")
    print("   - x_limits, y_limits: 图表轴范围")
    
    print("\n2. 荧光分析配置 (AnalysisConfig):")
    print("   - pre_event_time: 事件前时间（秒）")
    print("   - post_event_time: 事件后时间（秒）")
    print("   - baseline_window: 基线计算时间窗口")
    print("   - output_dir: 输出目录")
    print("   - save_plots: 是否保存图表")
    
    print("\n3. 命令行使用:")
    print("   python optimized_fluorescence_analyzer.py data.csv")
    print("   python optimized_fluorescence_analyzer.py data.csv --pre-event 30 --post-event 150")

def main():
    """主函数"""
    print("AAPlot 优化版本使用示例")
    print("=" * 50)
    
    # 显示配置选项
    show_configuration_options()
    
    # 运行示例（需要实际数据文件）
    # example_multi_file_analysis()
    # example_fluorescence_analysis()
    
    print("\n" + "=" * 50)
    print("示例脚本执行完成")
    print("请根据实际需求修改配置参数和数据文件路径")

if __name__ == "__main__":
    main()
