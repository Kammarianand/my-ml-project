from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_req(file_path: str) -> List[str]:
    # this function will get list of requirements
    req = []
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req = [r.replace('\n', '') for r in req]
    
    if HYPEN_E_DOT in req:
        req.remove(HYPEN_E_DOT)

    return req

setup(
    name='my-ml-project',
    version='0.0.1',
    author='Anand',
    author_email='kammarianand2000@gmail.com',
    packages=find_packages(),  # Corrected this line
    install_requires=get_req('requirements.txt')
)