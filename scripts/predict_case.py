"""Research inference from four registered MRI volumes, a mask, age and sex."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from glioma_idh.data import sha256
from glioma_idh.features import encode_views, make_encoder
from glioma_idh.imaging import preprocess_case
from glioma_idh.multimodal import predict_cached


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["t1","t1ce","t2","flair","mask"]:
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--age",type=float,required=True)
    parser.add_argument("--sex",choices=["male","female","unknown"],required=True)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--weights-dir",type=Path,default=Path("outputs/weights"))
    args=parser.parse_args()
    if not 18 <= args.age <= 110:
        raise ValueError("Input age outside the adult research-cohort range.")
    torch.set_num_threads(4)
    row={"patient_id":"INFERENCE_CASE",**{f"{name}_path":str(getattr(args,name)) for name in ["t1","t1ce","t2","flair","mask"]}}
    views,info,_=preprocess_case(row)
    encoder=make_encoder(args.weights_dir)
    model_info=torch.load(args.models/"fusion.pt",weights_only=True,map_location="cpu")
    if model_info["feature_provenance"]["weights_sha256"] != sha256(args.weights_dir/"checkpoints/resnet18-f37072fd.pth"):
        raise ValueError("Encoder weights differ from training provenance.")
    image=encode_views(encoder,views)[None,:]
    patients=pd.DataFrame({"age":[args.age],"sex":[args.sex]})
    probability=float(predict_cached(args.models,"fusion",image,patients)[0])
    print(json.dumps({"scope":"research_only","probability_idh_mutant_uncalibrated":probability,
        "preprocessing_version":info["preprocess_version"],"threshold_applied":False,
        "limitation":"No clinical interpretation or validated operating threshold is supplied."},indent=2))


if __name__=="__main__":
    main()
