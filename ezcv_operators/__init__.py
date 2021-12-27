import importlib
import pkgutil


def import_submodules(package_name):
    """ Adapted from https://stackoverflow.com/questions/3365740/how-to-import-all-submodules """
    package = importlib.import_module(package_name)
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        importlib.import_module(full_name)
        if is_pkg:
            import_submodules(full_name)


import_submodules(__name__)
