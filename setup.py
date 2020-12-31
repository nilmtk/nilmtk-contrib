from setuptools import setup
import os
import sys
import warnings

# Loosely based on NILMTK's setup.py

TRAVIS_TAG = os.environ.get('TRAVIS_TAG', '')

if TRAVIS_TAG:
    #TODO: validate if the tag is a valid version number
    VERSION = TRAVIS_TAG
    ISRELEASED = not ('dev' in TRAVIS_TAG)
    QUALIFIER = ''
else:
    MAJOR = 0
    MINOR = 1
    MICRO = 2
    DEV = 1 # For multiple dev pre-releases, please increment this value
    ISRELEASED = False
    VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
    QUALIFIER = ''


FULLVERSION = VERSION
if not ISRELEASED and not TRAVIS_TAG:
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "rev-parse", "--short", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        # Use a local version tag to include the git revision
        FULLVERSION += ".dev{}+git.{}".format(DEV, rev)
    except:
        FULLVERSION += ".dev{}".format(DEV)
        warnings.warn('WARNING: Could not get the git revision, version will be "{}"'.format(FULLVERSION))
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'nilmtk_contrib', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()
        
write_version_py()
# End of Version Check

setup(
    name='nilmtk-contrib',
    version=FULLVERSION,
    packages=['nilmtk_contrib', 'nilmtk_contrib.disaggregate'],
    install_requires=[
        'nilmtk>=0.4',
        'keras>=2.2.4',
        'tensorflow>=2.0',
        'cvxpy>=1.0.0'
    ],
    description="State-of-the-art algorithms for the task of energy disaggregation implemented using NILMTK's Rapid Experimentation API",
    author='NILMTK-contrib developers',
    author_email='',
    url='https://github.com/nilmtk/nilmtk-contrib',
    download_url="https://github.com/nilmtk/nilmtk-contrib/tarball/master#egg=nilmtk-contrib-dev",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache 2.0',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='smartmeters power electricity energy analytics redd '
             'disaggregation nilm nialm nilmtk'
)
