from setuptools import setup, find_packages

found_packages = find_packages()
print(f"Found packages: {found_packages}") # Print found packages
 
setup(
    name='a3x',
    version='0.1',
    packages=found_packages, # Use the found packages
) 