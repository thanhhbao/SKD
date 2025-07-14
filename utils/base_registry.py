from typing import Dict, Type, Optional, Any

class Registry:
  """
  Registry template.
  """
  def __init__(self, name: str='Module'):
    self.name = name
    self._modules: Dict[str, Type[Any]] = {}

  def register(self, module_class: Optional[Type[Any]] = None):
    """
    Register a module class.
    If no module_class is provided, returns a decorator function.
    
    Args:
      module_class: The module class to register
        
    Returns:
      The module class if used as a decorator, otherwise None
    """
    def _register(module_cls: Type[Any]) -> Type[Any]:
      module_class_name = module_cls.__name__
      if module_class_name in self._modules:
        raise ValueError(f"{self.name} {module_class_name} is already registered")
      self._modules[module_class_name] = module_cls
      return module_cls

    if module_class is None:
      return _register
    return _register(module_class)

  def get(self, name: str, *args, **kwargs) -> Any:
    """
    Get a module instance by name.
    """
    if name not in self._modules:
      raise KeyError(f"{self.name} {name} is not registered. Available modules: {list(self._modules.keys())}")
    
    module_class = self._modules[name]
    print(f'[BUILD] {self.name}: {name}')
    return module_class(*args, **kwargs)

  def list_all(self) -> list:
    """
    List all registered module names.
    """
    return list(self._modules.keys())