"""Descriptive analyses of clinical out-of-fold predictions, without test access."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from glioma_idh.data import write_json
from glioma_idh.evaluation import paired_auc_bootstrap


def main():
    root=Path("outputs/ipr/clinical")
    predictions=pd.read_csv(root/"clinical_oof_predictions.csv")
    development=pd.read_csv(root/"development_patients.csv")
    clinical=predictions[predictions.model.eq("clinical_logistic")].sort_values("patient_id")
    age=predictions[predictions.model.eq("age_logistic")].sort_values("patient_id")
    assert clinical.patient_id.tolist()==age.patient_id.tolist()
    paired=paired_auc_bootstrap(clinical.idh_label,clinical.probability_mutant,age.probability_mutant,samples=2000)
    paired["scope"]="descriptive_development_oof_only_not_confirmatory"
    paired["comparison"]="age_and_sex_logistic_minus_age_only_logistic"
    rows=[]
    joined=clinical.merge(development[["patient_id","age","sex"]],on="patient_id",validate="one_to_one")
    for sex,group in joined.groupby("sex"):
        rows.append({"group":sex,"patients":len(group),"idh_mutant":int(group.idh_label.sum()),
            "oof_auc":float(roc_auc_score(group.idh_label,group.probability_mutant)) if group.idh_label.nunique()==2 else None,
            "brier_score":float(brier_score_loss(group.idh_label,group.probability_mutant))})
    pd.DataFrame(rows).to_csv(root/"clinical_sex_subgroups.csv",index=False)
    write_json(root/"exploratory_comparison.json",{"paired_comparison":paired,"subgroups":rows,
        "caution":"Intervals condition on cross-validated predictions and do not include refitting uncertainty; subgroup differences are exploratory, not fairness certification."})
    print(json.dumps(paired,indent=2))


if __name__=="__main__":
    main()
