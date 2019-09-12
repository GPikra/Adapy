from distutils.core import setup
setup(
  name = 'adapy',
  packages = ['adapy'],
  version = '0.1',
  license='MIT',
  description = 'Library for domain adaptation',
  author = 'George Pikramenos, Eleanna Vali',
  author_email = 'gpik@di.uoa.gr',
  url = 'https://gitlab.com/cillab/adapy',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # not ready!!!!!
  keywords = ['adaptation'],   # Keywords that define our package best
  install_requires=[
          'keras',
          'tqdm',
          'numpy',
          'imageio'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package    
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',    
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)