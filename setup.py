# Created to install local packages in virtual environments


from setuptools import find_packages,setup

setup(
    name = "Text Summerization",
    version = "0.0.1",
    author="Govind Lipne",
    author_email="govind.lipne11@gmail.com",
    install_requires = ['openai','langchain','streamlit','python-dotenv','PyPDF2','langchain-openai'],
    packages=find_packages()
)