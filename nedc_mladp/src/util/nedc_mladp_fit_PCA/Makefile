# file: $MLADP/util/nedc_mladp_gen_feats/Makefile
#

# define source and object files
#
SRC = nedc_mladp_gen_PCA.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	cp -f $(SRC) $(MLADP)/nedc_mladp/bin/$(SRC)
	cd $(MLADP)/nedc_mladp/bin; chmod u+rwx,g+rwx,o+rw $(SRC)

#
# end of file
