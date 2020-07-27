from setuptools import setup

setup(name='embed',
      version='0.1',
      description='Basic immplementation of knowledge graph embedding. ',
      url='https://github.com/pbloem/embed',
      author='Peter Bloem',
      author_email='embed@peterbloem.nl',
      license='MIT',
      packages=['embed'],
      install_requires=[
            'torch',
            'tensorboard',
            'tqdm'
      ],
      zip_safe=False)