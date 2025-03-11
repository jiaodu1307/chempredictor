"""
ChemPredictor异常类模块
"""

class ChemPredictorError(Exception):
    """化学预测器基础异常类"""
    pass

class ConfigError(ChemPredictorError):
    """配置相关错误"""
    pass

class ModelError(ChemPredictorError):
    """模型相关错误"""
    pass

class PipelineError(ChemPredictorError):
    """流水线相关错误"""
    pass

class DataError(ChemPredictorError):
    """数据处理相关错误"""
    pass

class EncodingError(ChemPredictorError):
    """特征编码相关错误"""
    pass 