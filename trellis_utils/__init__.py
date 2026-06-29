import importlib

__submodules = [
    'datasets',
    'models',
    'modules',
    'pipelines',
    'renderers',
    'representations',
    'trainers',
    'utils',
]

__all__ = __submodules


def __getattr__(name):
    if name in __submodules:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
