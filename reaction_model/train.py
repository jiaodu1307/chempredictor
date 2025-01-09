import argparse
from .trainer import ReactionModelTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='反应模型训练脚本')
    
    # 设置所有参数的default为None
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='学习率')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=None,
                      help='最大训练轮数')
    parser.add_argument('--seed', type=int, default=None,
                      help='随机种子')
    parser.add_argument('--data_path', type=str, default=None,
                      help='数据文件路径')
    parser.add_argument('--fp_radius', type=int, default=None,
                      help='Morgan指纹半径')
    parser.add_argument('--fp_bits', type=int, default=None,
                      help='指纹位数')
    parser.add_argument('--mlp_dims', type=str, default=None,
                      help='MLP隐藏层维度列表，格式如"512,256"')
    parser.add_argument('--base_log_dir', type=str, default=None,
                      help='日志保存基础目录')
    parser.add_argument('--use_mpnn', action='store_true',
                      help='是否使用MPNN编码器')
    parser.add_argument('--mpnn_hidden_size', type=int, default=None,
                      help='MPNN隐藏层维度')
    parser.add_argument('--mpnn_depth', type=int, default=None,
                      help='MPNN消息传递深度')
    parser.add_argument('--dropout', type=float, default=None,
                      help='Dropout比率')
    parser.add_argument('--no_plot', action='store_true',
                      help='不绘制训练曲线')
    parser.add_argument('--weight_decay', type=float, default=None,
                      help='权重衰减系数')
    parser.add_argument('--lr_patience', type=int, default=None,
                      help='学习率调度器的耐心值')
    parser.add_argument('--lr_factor', type=float, default=None,
                      help='学习率调度器的衰减因子')
    parser.add_argument('--min_lr', type=float, default=None,
                      help='最小学习率')
    
    args = parser.parse_args()
    
    # 处理mlp_dims参数
    if args.mlp_dims is not None:
        args.mlp_dims = [int(x) for x in args.mlp_dims.split(',')]
        
    # 过滤掉所有None值
    return {k: v for k, v in vars(args).items() if v is not None}

def main():
    # 获取命令行参数
    args = parse_args()
    
    # 定义完整的默认参数字典
    params = {
        'mlp_dims': [512, 256],
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'max_epochs': 200,
        'seed': 42,
        'fp_radius': 2,
        'fp_bits': 2048,
        'use_mpnn': False,
        'mpnn_hidden_size': 300,
        'mpnn_depth': 3,
        'dropout': 0.1,
        'base_log_dir': "lightning_logs/reaction_model",
        'data_path': "data/processed/split_data",
        'plot_training_curves': True,
        'lr_patience': 10,
        'lr_factor': 0.2,
        'min_lr': 1e-5,
    }
    
    # 更新命令行指定的参数
    params.update(args)
    
    # 如果指定了no_plot参数，则设置plot_training_curves为False
    if args.get('no_plot', False):
        params['plot_training_curves'] = False
    
    # 创建训练器实例
    trainer = ReactionModelTrainer(params)
    
    # 开始训练
    model, pl_trainer, results = trainer.train(params['data_path'])
    
    print("训练完成！")
    print("测试结果:", results['test'])
    print("验证结果:", results['val'])
    print("训练结果:", results['train'])

if __name__ == "__main__":
    main() 