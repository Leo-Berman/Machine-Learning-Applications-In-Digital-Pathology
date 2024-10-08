# Machine-Learning-Applications-In-Digital-Pathology
A software system that automatically determines if a digital image of a pathology slide is cancerous  The software system will be implemented in Python and use a variety of local and standard packages  A user's guide for the software and an open source distribution that generic users can run"

How To Run:

Step 1: Add following to ~/.bashrc:

     . /data/isip/tools/GET_ENV.sh; # enable the isip conda environment
     
     MLADP=<path to github repo or whatever you renamed the github repo to > # path to the parent directory of the cloned repo IMPORTANT DO NOT HAVE BACKSLASH AT END
     
     PYTHONPATH="$NEDC_NFC/lib:$MLADP/nedc_mladp/lib:." # update Python path to contain requisite libraries
     
     export MLADP PYTHONPATH # export those two environment variables

Step 2 Activate new environment by executing in commandline:

     source ~/.bashrc

Step 3: Create a virtual environment by executing in comandline:

     python -m venv <name of virtual environment>

Step 4: Activate virtual environment by executing in comandline:

     source <path of virtual environment>/bin/actiate

Step 5: Install required packages by executing in comandline:

     pip install -r requirements.txt

Step 6 execute following commnds:

     cd $MLADP/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/

     ./make.sh

Step 7 create parameter files:

     cd $MLADP/nedc_mladp/data/

     In this directory these is a file called Example_Parameters.txt with an explanation of how to use the parameter file for our program.

Step 8 run the program:

     cd $MLADP/nedc_mladp/bin/

     ./nedc_mlad_run -p <absolute path to parameter file>


Notes:

	in order to extract a list of xml files and svs files, I recommend using the following commands.

	list of xml files:

	     find <absolute path of where your looking> -name *.xml>

	list of svs files:

	     find <absolute path of where your looking> -name *.xml>


	at the time of this being written you can find a train, dev, and eval set here:

	   /data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs