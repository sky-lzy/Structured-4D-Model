import importlib

__attributes = {
    'Octree': 'octree',
    'Gaussian': 'gaussian',
}

__all__ = list(__attributes.keys())


def __getattr__(name):
    if name in __attributes:
        module = importlib.import_module(f'.{__attributes[name]}', __name__)
        value = getattr(module, 'DfsOctree' if name == 'Octree' else name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
