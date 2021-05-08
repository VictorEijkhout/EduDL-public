################################################################
####
#### Makefile for building EduDL network
####
#### 2021
####
################################################################

info ::
	@echo && echo "%%%% Edu DL %%%%" && echo "Available rules:"

include Make.inc

#
# for now just a single build line
#
LIBSRCS := vector2.cpp matrix.cpp net.cpp dataset.cpp layer.cpp funcs.cpp vector.cpp trace.cpp
ifeq "${USE_BLIS}" "1"
 LIBSRCS += matrix_impl_blis.cpp vector_impl_blis.cpp vectorbatch_impl_blis.cpp
else
 LIBSRCS += matrix_impl_reference.cpp vector_impl_reference.cpp vectorbatch_impl_reference.cpp
endif
LIBOBJS = $(patsubst %.cpp,%.o,${LIBSRCS})

%.o : %.cpp
	@echo "compiling $<"
	@${CXX} -c $< \
	    -I${CXXOPTS}/include \
	    ` if [ "${DEBUG}" = "1" ] ; then echo "-DDEBUG" ; fi `\
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-DBLISNN -I${BLIS_INC_DIR}" ; fi ` \
	    ` if [ "${USE_GSL}" = "1" ] ; then echo "-DUSE_GSL -I${GSL_INC_DIR}" ; fi `

vector2.o vector_impl_blis.o vectorbatch_impl_blis.o : vector2.h
dataset.o layer.o matrix.o net.: matrix.h
dataset.o : dataset.h 
funcs.o : funcs.h matrix.h
layer.o : layer.h funcs.h
net.o : net.h dataset.h layer.h
test.o : matrix.h net.h dataset.h layer.h funcs.h
funcs.o net.o layer.o vectorbatch_impl_reference.o vectorbatch_impl_blis.o trace.o : trace.h

#
# implementation specific files have to be recompiled
# if we tinker with settings
#
BLAS_FILES = $(wildcard *_impl_*.cpp)
BLAS_OBJS = $(patsubst %.cpp,%.o,${BLAS_FILES})
${BLAS_OBJS} : Make.inc

TESTS = mnist posneg linear
TEST = mnist
info ::
	@echo "make test TEST=.... (out of: ${TESTS}, default=${TEST})"
.PHONY: test
test :  ${LIBOBJS}
	@program=test_${TEST} \
	 && make --no-print-directory $$program.o DEBUG=${DEBUG} \
	 && echo && echo "Linking test program <<$$program>>" \
	 && ${CXX} -o $$program $$program.o ${LIBOBJS} \
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi ` \
	 && echo ".. done"

info ::
	@echo "make run TEST=... RUN_OPTIONS=..."
.PHONY: run
run :
	@echo "Running: test_${TEST} ${RUN_OPTIONS}" \
	 && if [ ! -z "${TACC_TAU_DIR}" ] ; then \
	        tau_out=tau_out.${TEST} \
	         && echo ".. tau output in: $${tau_out}" \
	         && export TAU_TRACE=1 \
	         && export TAU_PROFILE=1 \
	         && export TRACEDIR=$${tau_out} \
	         && export PROFILEDIR=$${tau_out} \
	         && rm -rf $${tau_out} && mkdir -p $${tau_out} \
	      ; fi \
	 && ./test_${TEST} ${RUN_OPTIONS}

##
## old rules. must go
##
PROGRAMS = testdl posneg mpidl
info ::
	@echo "make PROGRAM (where PROGRAM in ${PROGRAMS}) : make application"
.SECONDEXPANSION:
testdl : test.o ${LIBOBJS}
	@echo "Linking test program <<$@>>"
	@${CXX} -o $@ $^ \
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
mpidl : test_mpi.o ${LIBOBJS} net_mpi.o
	@echo "Linking test program <<$@>>"
	@${CXX} -o $@ $^ \
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
posneg : $$@.o ${LIBOBJS}
	@echo "Linking test program <<$@>>"
	@${CXX} -o $@ $^ \
	    ` if [ "${USE_BLIS}" = "1" ] ; then echo "-L${BLIS_LIB_DIR} -lblis -lm" ; fi `
test.o posneg.o : vector2.h net.h dataset.h vector.h

.PHONY: clean
info ::
	@echo "make clean : remove junk and executables"
clean ::
	@rm -f *~ *.o *.o.tmp a.out ${PROGRAMS}
	@for t in ${TESTS} ; do rm -f test_$$t ; done
