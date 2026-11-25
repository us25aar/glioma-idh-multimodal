"""Summarise development errors and descriptive demographic subgroup metrics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=Path('outputs/development/validation'))
    args = parser.parse_args()
    rows, errors = [], []
    for seed_dir in sorted(args.output.glob('seed_*')):
        predictions_path = seed_dir / 'validation_oof.csv'
        if not predictions_path.is_file():
            continue
        predictions = pd.read_csv(predictions_path)
        cohort = pd.read_csv(seed_dir / 'common_development_patients.csv')
        joined = predictions.merge(cohort[['patient_id', 'age', 'sex']], on='patient_id', validate='many_to_one')
        joined['age_group'] = pd.cut(joined.age, [17, 39, 59, np.inf], labels=['18-39', '40-59', '60+'])
        for mode, model in joined.groupby('mode'):
            errors.append(model.loc[model.prediction.ne(model.idh_label),
                                     ['patient_id', 'idh_label', 'fold', 'mode', 'raw_probability',
                                      'calibrated_probability', 'threshold', 'prediction']].assign(seed=seed_dir.name))
            for attribute in ['sex', 'age_group']:
                for group_name, group in model.groupby(attribute, observed=True):
                    label = group.idh_label.to_numpy()
                    positive, negative = label == 1, label == 0
                    prediction = group.prediction.to_numpy()
                    rows.append({'seed': seed_dir.name, 'mode': mode, 'attribute': attribute,
                                 'group': str(group_name), 'patients': len(group),
                                 'idh_mutant': int(positive.sum()), 'idh_wildtype': int(negative.sum()),
                                 'raw_auc': float(roc_auc_score(label, group.raw_probability)) if positive.any() and negative.any() else None,
                                 'calibrated_brier': float(brier_score_loss(label, group.calibrated_probability)),
                                 'accuracy': float((prediction == label).mean()),
                                 'sensitivity': float((prediction[positive] == 1).mean()) if positive.any() else None,
                                 'specificity': float((prediction[negative] == 0).mean()) if negative.any() else None})
    if not rows:
        raise ValueError('Completed seed outputs are required before subgroup analysis.')
    pd.DataFrame(rows).to_csv(args.output / 'descriptive_subgroups.csv', index=False)
    pd.concat(errors, ignore_index=True).to_csv(args.output / 'development_errors.csv', index=False)
    print('Saved descriptive subgroup and error records for', len(rows), 'model/group/seed combinations.')


if __name__ == '__main__':
    main()
