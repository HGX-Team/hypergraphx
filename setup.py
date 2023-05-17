import io
import os
from setuptools import find_packages, setup

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
  name = 'hypergraphx',         
  version = '1.0',      
  license='BSD-3-Clause license',        
  description = 'HGX is a multi-purpose, open-source Python library for higher-order network analysis',   
  long_description=read("README.md"),
  long_description_content_type="text/markdown",
  author = 'HGX-Team',              
  author_email = 'lotitoqf@gmail.com',      
  url = 'https://github.com/HGX-Team/hypergraphx',   
  download_url = 'https://github.com/HGX-Team/hypergraphx/archive/refs/tags/1.1.1.tar.gz',    
  keywords = ['hypergraphs', 'networks'], 
  packages=find_packages(exclude=["tests", ".github"]),
  install_requires=read_requirements("requirements.txt"),
  #extras_require={"test": read_requirements("requirements-test.txt")},
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