from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name='dccapp',
    version='0.1.0',
    description='the application of dcc clustering',
    author='jasper gui',
    author_email = 'Jasper Gui'
    packages=find_packages(),
    license='MIT',
    install_requires=[
        # pandas,
    ],
    include_package_data=True,
    package_data={"": ["*.zip"]},
    zip_safe=False,
    scripts=[
        # 'bin/daemonctl',
    ],
)
