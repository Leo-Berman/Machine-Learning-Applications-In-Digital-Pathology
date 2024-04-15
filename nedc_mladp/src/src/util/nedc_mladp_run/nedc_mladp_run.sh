#!/bin/bash

while getopts 'p:h' opt; do
    case "$opt" in
	p)
	    arg=$(realpath "$OPTARG")
	    echo "Parameter file = ${arg}"
	    ;;
	?|h)
	    echo "Usage: $(basename $0) [-p param_file]"
	    exit 1
	    ;;

    esac
done     

./nedc_mladp_gen_feats -p ${arg}
./nedc_mladp_train -p ${arg}
./nedc_mladp_eval -p ${arg}
