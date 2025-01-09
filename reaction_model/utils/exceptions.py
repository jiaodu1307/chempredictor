class ReactionModelError(Exception):
    """反应模型基础异常类"""
    pass

class ConfigError(ReactionModelError):
    """配置错误"""
    pass

class DataError(ReactionModelError):
    """数据错误"""
    pass

class EncoderError(ReactionModelError):
    """编码器错误"""
    pass

class ModelError(ReactionModelError):
    """模型错误"""
    pass 