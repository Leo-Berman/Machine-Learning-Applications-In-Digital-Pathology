# file: $MLADP/functs/nedc_mladp_feats_tools/Makefile
#

# define source and object files
#
SRC = $(wildcard *.py)

# define an installation target
#
install:
	cd $(MLADP)/nedc_mladp/lib
	cp -f $(SRC) $(MLADP)/nedc_mladp/lib/
	cd $(MLADP)/nedc_mladp/lib; chmod u+rw,g+rw,o+r $(SRC)

#
# end of file
