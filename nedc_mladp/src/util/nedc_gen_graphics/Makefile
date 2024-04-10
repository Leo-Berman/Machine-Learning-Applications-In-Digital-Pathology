# file: $MLADP/util/nedc_gen_graphics/Makefile
#

# define source and object files
#
SRC = gen_feats.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	rm -f $(MLADP)/bin/$(BFILE)
	cp -f $(SRC) $(MLADP)/bin/$(BFILE)
	cd $(MLADP)/bin; chmod u+rwx,g+rwx,o+rw $(BFILE)

#
# end of file
