import logging

class Pipeline:
    """处理流水线类"""
    
    def __init__(self, config: dict, device: str = 'cpu'):
        """
        初始化流水线
        
        Args:
            config (dict): 流水线配置
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = device
        self.steps = {}
        self._setup_steps()
        
    def _setup_steps(self):
        """设置流水线步骤"""
        from .data_loading import DataLoader
        from .feature_encoding import EncoderFactory
        from .models import get_model
        
        steps_config = self.config['steps']
        
        # 设置数据加载器
        self.steps['data_loading'] = DataLoader(**steps_config['data_loading'])
        
        # 设置特征编码器
        encoder_factory = EncoderFactory()
        self.steps['feature_encoding'] = {}
        for column, config in steps_config['feature_encoding'].items():
            self.steps['feature_encoding'][column] = encoder_factory.create_encoder(
                config['encoder'],
                **config.get('params', {})
            )
        
        # 设置模型
        model_config = steps_config['model_training']
        model_type = model_config['type']
        model_params = model_config.get('params', {})
        
        # 将设备信息传递给模型
        model_params['device'] = self.device
        
        self.model = get_model(
            model_type=model_type,
            task_type=model_config['task_type'],
            **model_params
        ) 