from setuptools import setup, find_packages

setup(
    name="rep_graph_sample",
    version="0.1.0",
    packages=find_packages(),
    # install_requires=[
    #     # List your package dependencies here
    # ],
    # include_package_data=True,
    # description="A short description of your package",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # author="Your Name",
    # author_email="your.email@example.com",
    # url="https://github.com/yourusername/my_package",  # URL to the project's homepage
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
