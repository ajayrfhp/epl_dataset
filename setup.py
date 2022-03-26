from setuptools import setup

setup(name='epl_dataset',
      version='0.1',
      description='Dataset to train machine learning models on english premier league data',
      url='https://github.com/ajayrfhp/epl_dataset',
      author='ajayrfhp',
      author_email='ajayrfhp1710@gmail.com',
      license='MIT',
      packages=['epl_dataset'],
      install_requires=[
          'torch',
          'numpy',
          'pandas'
      ],
      zip_safe=False)