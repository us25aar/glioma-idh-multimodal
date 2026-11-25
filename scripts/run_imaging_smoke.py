"""Execute the real preprocessing/encoder/fusion code on marked synthetic NIfTIs.

This verifies software integration and runtime only. No patient accuracy is claimed.
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from glioma_idh.data import write_json
from glioma_idh.features import extract_features
from glioma_idh.imaging import prepare_manifest
from glioma_idh.multimodal import fit_head


def main():
    root=Path("outputs/ipr/synthetic_integration")
    root.mkdir(parents=True,exist_ok=True)
    raw=root/"nifti"
    raw.mkdir(exist_ok=True)
    rows=[]
    rng=np.random.default_rng(314)
    grid=np.indices((48,48,32))
    brain=((grid[0]-24)/22)**2+((grid[1]-24)/22)**2+((grid[2]-16)/15)**2<1
    started=time.perf_counter()
    for i in range(12):
        patient=f"SYNTHETIC_{i:03d}"
        row={"patient_id":patient}
        mask=(((grid[0]-(20+i%5))/6)**2+((grid[1]-24)/7)**2+((grid[2]-16)/5)**2<1).astype(np.uint8)
        for j,sequence in enumerate(["t1","t1ce","t2","flair"]):
            data=(brain*(rng.normal(70+j*20,12,brain.shape)+mask*(10+j*7))).astype(np.float32)
            path=(raw/f"{patient}_{sequence}.nii.gz").resolve()
            nib.save(nib.Nifti1Image(data,np.eye(4)),path)
            row[f"{sequence}_path"]=str(path)
        path=(raw/f"{patient}_mask.nii.gz").resolve()
        nib.save(nib.Nifti1Image(mask,np.eye(4)),path)
        row["mask_path"]=str(path)
        rows.append(row)
    manifest=root/"image_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest,index=False)
    qc=prepare_manifest(manifest,root/"processed",data_scope="synthetic_software_test")
    assert qc.status.eq("passed").all()
    feature_info=extract_features(root/"processed/image_qc.csv",root/"features",Path("outputs/weights"))
    with np.load(root/"features/features.npz",allow_pickle=False) as values:
        x=values["features"]
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(x[:8])
    c=rng.normal(size=(12,3)).astype(np.float32)
    y=np.array([0,1]*6)
    for mode in ["clinical","mri","fusion"]:
        model,prediction,history,_=fit_head(scaler.transform(x[:8]),c[:8],y[:8],scaler.transform(x[8:]),c[8:],y[8:],mode,42,max_epochs=3)
        assert prediction.shape==(4,) and np.isfinite(prediction).all()
        torch.save(model.state_dict(),root/f"{mode}_software_test.pt")
    evidence={"scope":"synthetic_software_integration_only","synthetic_cases":12,"real_patient_cases":0,
              "preprocessing_passed":int(qc.status.eq("passed").sum()),"feature_shape":list(x.shape),
              "pretrained_encoder_executed":True,"heads_trained":["clinical","mri","fusion"],
              "epochs_maximum_per_head":3,"test_set_accessed":False,"elapsed_seconds":round(time.perf_counter()-started,3),
              "feature_extraction":feature_info,"claim_limit":"No research performance estimate; fixtures are not real MRI."}
    write_json(root/"integration_evidence.json",evidence)
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
