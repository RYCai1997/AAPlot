"""
多文件数据分析运行脚本
替换原始的Jupyter notebook功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from optimized_aplot_notebook import DataProcessor, PlotGenerator, MetricsCalculator, PlotConfig

def main():
    """运行多文件数据分析"""
    print("=== AAPlot 多文件数据分析 ===")
    
    # 步骤1：配置参数
    config = PlotConfig()
    
    # 修改为您的实际数据文件夹路径
    config.folder_path = r"D:\Science\Experiment data\eCB sensors\Selective sensor\FiberPhotometry\Drug intake\NAcSh_All_data_in_1_folder\01_Summary\2-AG\cocaine"
    
    # 可选：调整其他参数
    config.pre_event_time = 0.5  # 事件前0.5小时
    config.post_event_time = 2.0  # 事件后2小时
    config.x_limits = (-0.25, 1.5)  # X轴范围
    config.y_limits = (-5, 15)      # Y轴范围
    
    print(f"数据文件夹: {config.folder_path}")
    
    # 检查文件夹是否存在
    if not os.path.exists(config.folder_path):
        print(f"❌ 错误: 文件夹不存在 - {config.folder_path}")
        print("请修改 config.folder_path 为实际的数据文件夹路径")
        return
    
    try:
        # 步骤2：初始化数据处理器
        processor = DataProcessor(config)
        
        # 步骤3：加载数据文件
        print("📁 正在加载数据文件...")
        if not processor.load_data_files(config.folder_path):
            print("❌ 数据加载失败")
            return
        print("✅ 数据加载成功")
        
        # 步骤4：数据对齐
        print("🔄 正在对齐数据...")
        if not processor.align_data_to_events():
            print("❌ 数据对齐失败")
            return
        print("✅ 数据对齐成功")
        
        # 步骤5：生成图表
        print("📊 正在生成图表...")
        plot_generator = PlotGenerator(config)
        plot_path = os.path.join(config.folder_path, 'optimized_plot.svg')
        
        if plot_generator.create_main_plot(processor.final_df, plot_path):
            print(f"✅ 图表已保存: {plot_path}")
        else:
            print("❌ 图表生成失败")
        
        # 步骤6：计算统计指标
        print("📈 正在计算统计指标...")
        metrics_calculator = MetricsCalculator(config)
        metrics_df = metrics_calculator.calculate_metrics(processor.final_df)
        
        # 保存指标到CSV
        metrics_path = os.path.join(config.folder_path, 'optimized_metrics.csv')
        metrics_df.to_csv(metrics_path)
        print(f"✅ 指标已保存: {metrics_path}")
        
        # 保存对齐后的数据
        data_path = os.path.join(config.folder_path, 'optimized_aligned_data.csv')
        processor.final_df.to_csv(data_path, index=False)
        print(f"✅ 对齐数据已保存: {data_path}")
        
        # 步骤7：显示结果摘要
        print("\n" + "="*50)
        print("📋 分析结果摘要")
        print("="*50)
        
        signal_columns = [col for col in processor.final_df.columns if 'signal_' in col]
        print(f"📊 数据点数量: {len(processor.final_df)}")
        print(f"📈 信号数量: {len(signal_columns)}")
        print(f"⏰ 时间范围: {processor.final_df['Time(h)'].min():.2f} - {processor.final_df['Time(h)'].max():.2f} 小时")
        
        print("\n📊 统计指标预览:")
        print(metrics_df.head())
        
        print(f"\n✅ 分析完成！结果保存在: {config.folder_path}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
