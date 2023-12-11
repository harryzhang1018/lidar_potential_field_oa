from setuptools import find_packages, setup

package_name = 'lidar_potential_field_oa'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    py_modules=[
        'lidar_potential_field_oa.auto_driving',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='art',
    maintainer_email='art@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_driving = lidar_potential_field_oa.auto_driving:main',
        ],
    },
)
