import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='ADAPY',  

     version='0.1',

     scripts=['adaptation.py','auxilliary.py','batch_generator.py'] ,

     author="George Pikramenos",

     author_email="deepak.kumar.iet@gmail.com",

     description="Domain Adaptation package",

     long_description=long_description,

   long_description_content_type="text/markdown",

     url="https://gitlab.com/cillab/adapy/",

     packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

 )
