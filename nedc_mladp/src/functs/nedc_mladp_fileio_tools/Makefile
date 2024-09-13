# file: $MLADP/functs/nedc_mladp_fileio_tools/Makefile
#

# define source and object files
#
SRC = $(wildcard *.py)

# define an installation target
#
install:
	cd /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib; rm -f $(SRC)
	cp -f $(SRC) /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib/
	cd /data/isip/exp/tuh_dpath/exp_0280/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib; chmod u+rw,g+rw,o+r $(SRC)

#
# end of file
