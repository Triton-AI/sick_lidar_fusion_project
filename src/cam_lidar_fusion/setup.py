from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cam_lidar_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'model']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join(os.path.join("share", package_name), "model"), glob("model/*.pt")),
        (os.path.join(os.path.join("share", package_name), "blob_model"), glob("blob_model/*")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='davidq1688@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = cam_lidar_fusion.fusion_node:main'
        ],
    },
)
