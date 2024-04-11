# file: $MLADP/functs/nedc_mladp_ann_tools/Makefile
#

# define source and object files
#
SRC = $(wildcard *.py)

# define an installation target
#
install:
	cd $(MLADP)/lib; rm -f $(SRC)
	cp -f $(SRC) $(MLADP)/lib/
	cd $(MLADP)/lib; chmod u+rw,g+rw,o+r $(SRC)

#
# end of file
