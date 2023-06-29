
import io
import os
import setuptools

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

INSTALL_REQUIRES = [
    'numpy',
    'jax',
    'jaxlib'
]

setuptools.setup(
    name='mclmc',
    version='0.2.5',
    license='Apache 2.0',
    author='Jakob Robnik',
    # author_email='',
    install_requires=INSTALL_REQUIRES,
    # url='',
    packages=setuptools.find_packages(),
    # download_url = "https://pypi.org/project/jax-md/",
    # project_urls={
    #     "Source Code": "https://github.com/google/jax-md",
    #     "Documentation": "https://arxiv.org/abs/1912.04232",
    #     "Bug Tracker": "https://github.com/google/jax-md/issues",
    # },
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Faster gradient based sampling',
    python_requires='>=3.8',
    # classifiers=[
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    #     'License :: OSI Approved :: Apache Software License',
    #     'Operating System :: MacOS',
    #     'Operating System :: POSIX :: Linux',
    #     'Topic :: Software Development',
    #     'Topic :: Scientific/Engineering',
    #     'Intended Audience :: Science/Research',
    #     'Intended Audience :: Developers',
    # ]
    )