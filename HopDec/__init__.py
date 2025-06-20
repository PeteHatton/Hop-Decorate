# from . import Version
__version__ = None #Version.getVersion()

from . import _version
__version__ = _version.get_versions()['version']
