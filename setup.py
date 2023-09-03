#!/usr/bin/env python

from setuptools import setup

setup(name='gru',  # package name is always the same as the directory.
      version='1.0',
      # list folders, not files
      packages=['gru',
                'gru.utils', # export subpackage, or it will not be callable from external
                #'test',
                #'bin',
               ],
      #scripts=['bin/test.py'],
      #package_data={'src': ['data/data.txt']
      )