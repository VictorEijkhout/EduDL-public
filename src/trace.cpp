/****************************************************************
 ****************************************************************
 ****
 **** This text file is part of the source of 
 **** `Introduction to High-Performance Scientific Computing'
 **** by Victor Eijkhout, copyright 2012-2021
 ****
 **** Deep Learning Network code 
 **** copyright 2021 Ilknur Mustafazade
 ****
 ****************************************************************
 ****************************************************************/

int trace_level;

void set_trace_level(int ell) { trace_level = ell; };
bool trace_progress() { return trace_level>=1; };
bool trace_scalars() { return trace_level>=2; };
bool trace_arrays()  { return trace_level>=3; };
