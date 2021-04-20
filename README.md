# EduDL-public

This is the repository for the open source deep learning network EduDL,
a network written for educational purposes, rather than production.

## Installation

You need to customize the makefile to your local machine by editing `Make.inc`:

1. Indicate whether you have the [BLIS](https://github.com/flame/blis)
 library 
2. Set your compiler and options
3. Indicate the location of the [cxxops](https://github.com/jarro2783/cxxopts) library which is needed for the example networks

## Example applications

Currently there are two main programs that can be built with the supplied Makefile:

    make testdl
    make posneg
    
These test programs are driven by commandline options; run 

    ./testdl -h
    ./posneg -h
    
to see available options.

The `testdl` parses the MNIST dataset, and you can test it by:

    ./testdl -e 4 -l 4 -d ../../mnist/

You can make your own network by taking either of these as example.
More thorough documentation is forthcoming.

## High performance

All linear algebra routines have been abstracted to a number of matrix and vector operations, for which currently two implementations are given:

1. The textbook implementations in files with `_impl_reference.cpp` names, and
2. Optimized implementations using the BLIS library (see above) in files with `_impl_blis.cpp` names.