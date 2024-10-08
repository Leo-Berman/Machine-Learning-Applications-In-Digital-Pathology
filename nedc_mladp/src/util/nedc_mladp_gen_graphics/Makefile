# file: $MLADP/util/nedc_mladp_gen_graphics/Makefile
#

# define source and object files
#
SRC = nedc_mladp_gen_graphics.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	
	cp -f $(SRC) $(MLADP)/nedc_mladp/bin/$(BFILE)
	cd $(MLADP)/nedc_mladp/bin; chmod u+rwx,g+rwx,o+rw $(BFILE)

#
# end of file
