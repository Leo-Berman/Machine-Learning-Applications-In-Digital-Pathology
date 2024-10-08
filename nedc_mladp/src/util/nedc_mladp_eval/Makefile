# file: $MLADP/util/nedc_evaluate_model/Makefile
#

# define source and object files
#
SRC = nedc_mladp_eval.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	rm -f $(MLADP)/nedc_mladp/bin/$(BFILE)
	cp -f $(SRC) $(MLADP)/nedc_mladp/bin/$(BFILE)
	cd $(MLADP)/nedc_mladp/bin; chmod u+rwx,g+rwx,o+rw $(BFILE)

#
# end of file
