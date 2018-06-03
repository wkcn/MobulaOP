.. MobulaOP documentation master file, created by
   sphinx-quickstart on Tue May 15 10:37:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MobulaOP's Documentation!
========================================
**MobulaOP** is a simple and flexible cross framework operators toolkit.

You can write the custom operators by Python/C++/C/CUDA without rebuilding deep learning framework from source.

Architecture
============
There are 3 components in MobulaOP, namely **op**, **func** and **glue**.

- op
    Creating Custom Operators
- func
    Writing Efficient Functions
- glue
    Linking MobulaOP and Deep Learning Framework

.. toctree::
   :maxdepth: 2
   :caption: Contents

   create_custom_operators
   write_efficient_functions
   link_mobulaop_and_framework
