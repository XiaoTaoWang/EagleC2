# Author: Xiaotao Wang

"""
Setup script for EagleC2.

"""
import os, sys, eaglec, glob
import setuptools

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if (sys.version_info.major!=3) or (sys.version_info.minor<9):
    print('PYTHON 3.9+ IS REQUIRED. YOU ARE CURRENTLY USING PYTHON {}'.format(sys.version.split()[0]))
    sys.exit(2)

# Guarantee Unix Format
for src in glob.glob('scripts/*'):
    text = open(src, 'r').read().replace('\r\n', '\n')
    open(src, 'w').write(text)

setuptools.setup(
    name = 'eaglec',
    version = eaglec.__version__,
    author = eaglec.__author__,
    author_email = 'wangxiaotao@fudan.edu.cn',
    url = 'https://github.com/XiaoTaoWang/EagleC2',
    description = 'A deep-learning framework for predicting a full range of structural variations from bulk and single-cell contact maps',
    keywords = 'Hi-C cooler deep-learning SVs',
    packages = setuptools.find_packages(),
    package_data = {
        '': ['data/*']
    },
    scripts = glob.glob('scripts/*'),
    long_description = read('README.rst'),
    long_description_content_type='text/x-rst',
    classifiers = [
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: POSIX',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
    zip_safe=False
    )

