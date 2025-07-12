# Importing necessary tools from setuptools and typing
from setuptools import find_packages, setup  # Tools for packaging Python projects
from typing import List  # For type hinting: List[str]

# Define a constant used to identify editable installs in requirements
HYPEN_E_DOT = '-e .'

# Define a function to read and clean a requirements.txt file
def get_requirements(file_path: str) -> List[str]:
    requirements = []  # Initialize an empty list

    # Open the file at the given path
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()  # Read all lines from the file

        # Remove newline characters from each requirement
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' if it's present (editable install directive)
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements  # Return the cleaned list of requirements

# Call the setup function to define the package configuration
setup(
    name="Xray",  # Project name
    version="0.0.1",  # Version number
    author="sunny savita",  # Author name
    author_email="abhi95696719@gmail.com",  # Author contact email

    # Dependencies that should be installed, loaded from the given file
    install_requires=get_requirements(
        r"C:\\Users\ABHISHEK MAURYA\\Chest-X-Ray-Medical- DL\\requirements_dev.txt"
    ),

    packages=find_packages()  # Automatically find all sub-packages in the project folder
)

