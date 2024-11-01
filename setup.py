from setuptools import find_packages, setup

setup(
    name="Resume Chatbot",
    version="0.0.1",
    author="Akshay",
    author_email="ambewadkarakshay@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain ','datasets','pypdf','python-dotenv','flask']
)