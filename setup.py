import os
from setuptools import setup, find_packages

setup(
    name='nectargan',
    version='0.1.0',
    author='Zachary Bork',
    description='A graphical GAN development environment and model assembly framework.',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/ZacharyBork/NectarGAN',
    packages=find_packages(exclude=['scripts', 'docs']),
    install_requires=[
        'albumentations==2.0.8',
        'matplotlib==3.10.3',
        'numpy==2.3.0',
        'onnxruntime==1.21.0',
        'opencv-python==4.11.0.86',
        'Pillow==11.2.1',
        'pyqtgraph==0.13.7',
        'PySide6==6.9.1',
        'PySide6-Addons==6.9.0',
        'PySide6-Essentials==6.9.0',
        'torch==2.6.0+cu118',
        'torchvision==0.21.0+cu118',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: User Interfaces',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Utilities',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        'Framework :: PySide',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pix2pix-train=scripts.paired.train:main',
            'pix2pix-test=scripts.paired.test:main',
            'pix2pix-ui=scripts.toolbox:main',
        ],
    },
)