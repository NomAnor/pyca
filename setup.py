from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import versioneer
import os, sys


# This is needed so setup.py can be run without having numpy installed.
# When bulding the extension, numpy is installed by setup_requires and we can import it.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


if "EPICS_BASE" not in os.environ or "EPICS_HOST_ARCH" not in os.environ:
    print(sys.stderr, "EPICS_BASE and EPICS_HOST_ARCH must be set")
    sys.exit(-1)

if sys.platform == 'darwin':
    libsrc = 'Darwin'
    compiler = 'clang'
elif sys.platform.startswith('linux'):
    libsrc = 'Linux'
    compiler = 'gcc'
else:
    libsrc = None

epics_inc = os.environ["EPICS_BASE"] + "/include"
epics_lib = os.environ["EPICS_BASE"] + "/lib/" + os.getenv("EPICS_HOST_ARCH")


pyca = Extension(
    'pyca',
    language='c++',
    sources=['pyca/pyca.cc'],
    include_dirs=['pyca', epics_inc,
        epics_inc + '/os/' + libsrc,
        epics_inc + '/compiler/' + compiler,
    ],
    library_dirs=[epics_lib],
    runtime_library_dirs=[epics_lib],
    libraries=['Com', 'ca']
)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({'build_ext': build_ext})

setup(
    name='pyca',
    author='SLAC',
    version=versioneer.get_version(),
    description='Python Channel Access library',
    license='SLAC Open Licence',
    url='https://github.com/slaclab/pyca',
    packages=['psp'],
    install_requires=['numpy'],
    setup_requires=['pytest-runner', 'numpy'],
    extras_require={
        'doc': [ 'sphinx', 'sphinx_rtd_theme' ],
        'test': [ 'pytest', 'pytest-timeout', 'pcaspy' ]
    },
    ext_modules = [pyca],
    cmdclass=cmdclass
)
