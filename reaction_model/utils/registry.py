from typing import Dict, Type, Any, List

class Registry:
    """组件注册器"""
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}
    
    def register(self, name: str = None):
        """注册装饰器"""
        def wrapper(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise ValueError(f"{key} 已经在 {self.name} 注册表中注册")
            self._registry[key] = cls
            return cls
        return wrapper
    
    def get(self, name: str) -> Type[Any]:
        """获取已注册的组件"""
        if name not in self._registry:
            raise ValueError(f"未找到 {name} 在 {self.name} 注册表中")
        return self._registry[name]
    
    def list_available(self) -> List[str]:
        """列出所有可用组件"""
        return list(self._registry.keys())

# 创建注册表
ENCODER_REGISTRY = Registry("encoder")
MODEL_REGISTRY = Registry("model") 