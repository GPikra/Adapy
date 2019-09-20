from distutils.core import setup
setup(
  name = 'adapy',
  packages = ['adapy'],
  version = '0.4',
  license='MIT',
  description = 'Library for domain adaptation',
  author = 'George Pikramenos, Eleanna Vali',
  author_email = 'gpik@di.uoa.gr',
  url = 'https://github.com/GPikra/Adapy',
  download_url = 'https://github.com/GPikra/Adapy/archive/v_04.tar.gz',
  keywords = ['domain adaptation','tranfer learning'],
  install_requires=[
          'keras',
          'tqdm',
          'numpy',
          'imageio',
          'tensorflow'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',   
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',    
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)