from setuptools import setup

setup(
    name='synt',
    version='0.1.0',
    author='Tawlk',
    author_email='team@tawlk.com',
    packages=['synt', 'synt.utils'],
    scripts=['bin/synt'],
    url='http://github.com/Tawlk/synt',
    license='LICENSE',
    description='Synt (pronounced: "cent") is a python library aiming to be a general solution to identifying a given peice of text, particularly social network statuses, as either negative, neutral, or positive.',
    long_description=open('README.md').read(),
    install_requires=[
        'pyyaml',
        'redis',
        'nltk'
    ]

)
