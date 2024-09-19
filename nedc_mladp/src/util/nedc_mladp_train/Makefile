# file: $MLADP/util/nedc_mladp_train/Makefile
#

# define source and object files
#
SRC = nedc_mladp_train.py
BFILE := $(basename $(SRC))

# define an installation target
#
install:
	rm -f /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/bin/$(BFILE)
	cp -f $(SRC) /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/bin/$(BFILE)
	cd /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/bin; chmod u+rwx,g+rwx,o+rw $(BFILE)

#
# end of file
