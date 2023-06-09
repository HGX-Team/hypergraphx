import io
import os
from setuptools import find_packages, setup

def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

setup(
  name = 'hypergraphx',         
  version = '1.4.2',      
  license='BSD-3-Clause license',        
  description = 'HGX is a multi-purpose, open-source Python library for higher-order network analysis',
  long_description=read("README.md"),
  long_description_content_type="text/markdown",
  author = 'HGX-Team',              
  author_email = 'lotitoqf@gmail.com',      
  url = 'https://github.com/HGX-Team/hypergraphx',   
  keywords = ['hypergraphs', 'networks'], 
  packages=find_packages(exclude=["tests", ".github"]),
  install_requires=['numpy',
                   'scipy',
                   'networkx',
                   'rpy2',
                   'pandas',
                   'scikit-learn',
                   'pytest',
                   'matplotlib',
                   'seaborn'],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',   
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
