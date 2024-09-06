# file: $MLADP/util/nedc_evaluate_model/Makefile
#

# define source and object files
#
SRC = nedc_mladp_eval.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	rm -f $(MLADP)/bin/$(BFILE)
	cp -f $(SRC) $(MLADP)/bin/$(BFILE)
	cd $(MLADP)/bin; chmod u+rwx,g+rwx,o+rw $(BFILE)

#
# end of file
