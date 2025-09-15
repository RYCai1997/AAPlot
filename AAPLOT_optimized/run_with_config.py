"""
使用配置文件运行AAPlot分析
支持YAML配置文件进行参数设置
"""

import os
import sys
import yaml
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from optimized_aplot_notebook import DataProcessor, PlotGenerator, MetricsCalculator, PlotConfig

def load_config(config_file="config.yaml"):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_file}")
        return config_dict
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_file}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None

def create_config_from_yaml(config_dict):
    """从YAML配置创建PlotConfig对象"""
    config = PlotConfig()
    
    # 更新配置参数
    if 'data' in config_dict:
        data_config = config_dict['data']
        config.folder_path = data_config.get('folder_path', config.folder_path)
    
    if 'event_detection' in config_dict:
        event_config = config_dict['event_detection']
        config.pre_event_time = event_config.get('pre_event_time', config.pre_event_time)
        config.post_event_time = event_config.get('post_event_time', config.post_event_time)
    
    if 'smoothing' in config_dict:
        smooth_config = config_dict['smoothing']
        config.smoothing_window = smooth_config.get('window_length', config.smoothing_window)
        config.smoothing_polyorder = smooth_config.get('polyorder', config.smoothing_polyorder)
    
    if 'plot' in config_dict:
        plot_config = config_dict['plot']
        config.figsize = tuple(plot_config.get('figsize', config.figsize))
        
        if 'colors' in plot_config:
            colors = plot_config['colors']
            config.colors = {
                'individual': colors.get('individual', config.colors['individual']),
                'mean': colors.get('mean', config.colors['mean']),
                'error_band': colors.get('error_band', config.colors['error_band'])
            }
        
        if 'axis' in plot_config:
            axis_config = plot_config['axis']
            config.x_limits = tuple(axis_config.get('x_limits', config.x_limits))
            config.y_limits = tuple(axis_config.get('y_limits', config.y_limits))
            config.x_ticks = tuple(axis_config.get('x_ticks', config.x_ticks))
            config.y_ticks = tuple(axis_config.get('y_ticks', config.y_ticks))
    
    return config

def main():
    """使用配置文件运行分析"""
    print("=== AAPlot 配置文件使用示例 ===")
    
    # 步骤1：加载配置文件
    config_dict = load_config("config.yaml")
    if config_dict is None:
        print("使用默认配置...")
        config = PlotConfig()
    else:
        config = create_config_from_yaml(config_dict)
    
    print(f"📁 数据文件夹: {config.folder_path}")
    print(f"⏰ 事件前时间: {config.pre_event_time} 小时")
    print(f"⏰ 事件后时间: {config.post_event_time} 小时")
    
    # 检查文件夹是否存在
    if not os.path.exists(config.folder_path):
        print(f"❌ 错误: 文件夹不存在 - {config.folder_path}")
        print("请在 config.yaml 中修改 folder_path 参数")
        return
    
    try:
        # 步骤2：运行分析（与之前相同的流程）
        processor = DataProcessor(config)
        
        print("📁 正在加载数据文件...")
        if not processor.load_data_files(config.folder_path):
            print("❌ 数据加载失败")
            return
        print("✅ 数据加载成功")
        
        print("🔄 正在对齐数据...")
        if not processor.align_data_to_events():
            print("❌ 数据对齐失败")
            return
        print("✅ 数据对齐成功")
        
        print("📊 正在生成图表...")
        plot_generator = PlotGenerator(config)
        plot_path = os.path.join(config.folder_path, 'config_based_plot.svg')
        
        if plot_generator.create_main_plot(processor.final_df, plot_path):
            print(f"✅ 图表已保存: {plot_path}")
        else:
            print("❌ 图表生成失败")
        
        print("📈 正在计算统计指标...")
        metrics_calculator = MetricsCalculator(config)
        metrics_df = metrics_calculator.calculate_metrics(processor.final_df)
        
        metrics_path = os.path.join(config.folder_path, 'config_based_metrics.csv')
        metrics_df.to_csv(metrics_path)
        print(f"✅ 指标已保存: {metrics_path}")
        
        print(f"\n✅ 基于配置文件的分析完成！")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def show_config_example():
    """显示配置文件示例"""
    print("\n📝 配置文件 (config.yaml) 示例:")
    print("""
data:
  folder_path: "D:/your/data/folder"
  
event_detection:
  pre_event_time: 0.5
  post_event_time: 2.0
  
plot:
  figsize: [8, 3]
  colors:
    individual: "lightgrey"
    mean: "darkgoldenrod"
  axis:
    x_limits: [-0.25, 1.5]
    y_limits: [-5, 15]
""")

if __name__ == "__main__":
    # 显示配置示例
    show_config_example()
    
    # 运行分析
    main()
