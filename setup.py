import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='adapy',  

     version='0.1',

     scripts=['adaptation.py','auxilliary.py','batch_generator.py'] ,

     author="George Pikramenos, Eleanna Vali",

     author_email="gpik@di.uoa.gr",

     description="Package for Adversarial Domain Adaptation",

     long_description=long_description,

   long_description_content_type="text/markdown",

     url="git@gitlab.com:cillab/adapy.git",

     packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

 )
