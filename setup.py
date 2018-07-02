from distutils.core import setup

setup(
    name='FAR-HO',
    version='',
    packages=['far_ho', 'far_ho.examples'],
    url='',
    license='',
    author='Luca Franceschi',
    author_email='',
    description='Forward And Reverse gradient-based hyperparameter optimization package for TensorFlow',
    requires=['numpy', 'datapackage']
)
