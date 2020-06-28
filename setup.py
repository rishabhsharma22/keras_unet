import setuptools

with open("README.md",'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name = 'keras_unet',
    version='0.0.1',
    author= "Rishabh Sharma",
    author_email= "rsharma26@uh.edu",
    description="A package to import UNet model in keras",
    url = "https://github.com/rishabhsharma22/keras_unet",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language:: Python ::3",
    ]
)