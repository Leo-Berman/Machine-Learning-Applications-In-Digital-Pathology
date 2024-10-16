#!/usr/bin/env python
from nedc_mladp_gen_feats import gen_feats
from nedc_mladp_train_model import train_model
from nedc_mladp_gen_preds import gen_preds

def main():
    feature_files = gen_feats()
    model = train_model(feature_files)
    gen_preds(feature_files,model)

if __name__ == "__main__":
    main()
