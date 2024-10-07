# file: $MLADP/functs/nedc_mladp_ann_tools/Makefile
#

# define source and object files
#
SRC = $(wildcard *.py)

# define an installation target
#
install:
	cd $(MLADP)/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib; rm -f $(SRC)
	cp -f $(SRC) $(MLADP)/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib/
	cd $(MLADP)/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib; chmod u+rw,g+rw,o+r $(SRC)

#
# end of file
