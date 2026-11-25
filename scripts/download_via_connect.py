"""Use the official running Connect client for public development-only downloads."""
import json
import argparse
import time
import uuid
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pandas as pd
from tcia_public import authenticate,request

LOCAL='http://127.0.0.1:33003/v5/connect'
SESSION = Path('outputs/ipr/download_session.json')
if SESSION.exists():
    APP_ID = json.loads(SESSION.read_text())['app_id']
else:
    APP_ID = str(uuid.uuid4())
    SESSION.parent.mkdir(parents=True, exist_ok=True)
    SESSION.write_text(json.dumps({'app_id': APP_ID}))


def local(path,data=None, timeout=15):
    headers={'Origin':'https://faspex.cancerimagingarchive.net','Content-Type':'application/json'}
    if data is not None:
        data['aspera_connect_settings'] = {'app_id': APP_ID, 'back_link': 'https://faspex.cancerimagingarchive.net'}
    encoded=json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(LOCAL+path,data=encoded,headers=headers),timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as error:
        body = json.loads(error.read())
        details = body.get('error', body)
        safe = {key: details.get(key) for key in ['code', 'user_message', 'internal_message', 'message']
                if isinstance(details, dict) and key in details}
        raise RuntimeError('Connect API request failed: '+str(error.code)+' '+json.dumps(safe)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select-folder', action='store_true')
    parser.add_argument('--patients-per-class-fold', type=int, default=0)
    args = parser.parse_args()
    if args.patients_per_class_fold < 0:
        parser.error('--patients-per-class-fold must be nonnegative')
    destination=Path(json.loads(SESSION.read_text()).get('destination', 'dataset')).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if args.select_folder:
        chosen = local('/windows/select-open-folder-dialog/', {'title': 'Select Glioma MRI download folder',
                       'allowMultipleSelection': False}, timeout=180)
        print('Selected destination:', chosen, flush=True)
        selected_paths = [item.get('name') for item in chosen.get('dataTransfer', {}).get('files', [])]
        if len(selected_paths) != 1:
            raise ValueError('One destination folder must be selected.')
        destination = Path(selected_paths[0]).resolve()
        if not destination.is_relative_to(Path.cwd().resolve()):
            raise ValueError('The download destination must be inside the project workspace.')
        SESSION.write_text(json.dumps({'app_id': APP_ID, 'destination': str(destination)}))
    api,headers,package=authenticate()
    dev=pd.read_csv('outputs/week1/development_patients.csv')
    if args.patients_per_class_fold:
        selected = dev.sort_values('patient_id').groupby(['fold','idh_label'], group_keys=False).head(args.patients_per_class_fold)
    else:
        selected=pd.concat([dev[dev.idh_label.eq(label)].head(2) for label in [0,1]])
    selected[['patient_id','idh_label','fold']].to_csv('outputs/ipr/download_selection.csv', index=False)
    rows=[]
    for patient in selected.patient_id:
        required = ['brain_t1.nii.gz', 'brain_t1ce.nii.gz', 'brain_t2.nii.gz',
                    'brain_flair.nii.gz', 'tumorseg_FeTS.nii.gz']
        if all((destination / patient / name).is_file() for name in required):
            rows.append({'patient_id': patient, 'status': 'already_present'})
            print(patient, 'required files already present', flush=True)
            continue
        spec=json.loads(request(api+f'/packages/{package}/transfer_spec/download?transfer_type=connect&type=received',
            {'paths':[{'path':f'/UTSW-Glioma/{patient}'}]},headers).read())
        spec['destination_root']=str(destination)
        settings={'app_id': APP_ID, 'request_id': str(uuid.uuid4()),
                  'back_link': 'https://faspex.cancerimagingarchive.net',
                  'allow_dialogs':True,'use_absolute_destination_path':True}
        try:
            result=local('/transfers/start',{'transfer_specs':[{'transfer_spec':spec,'aspera_connect_settings':settings}]}, timeout=60)
        except TimeoutError:
            # The client may have started the transfer while awaiting its UI acknowledgement.
            result = {}
        transfers=result.get('transfer_specs',[])
        ids=[t.get('uuid') or t.get('aspera_connect_settings',{}).get('transfer_id') for t in transfers]
        print('Started official-client transfer',patient,'response fields',list(result),flush=True)
        status='unknown'
        previous_status = None
        for _ in range(300):
            time.sleep(2)
            activities=local('/transfers/activity',{'iteration_token':0})
            candidates=activities.get('transfers',[])
            matching=[t for t in candidates if t.get('uuid') in ids or
                      t.get('aspera_connect_settings',{}).get('request_id') == settings['request_id']]
            if matching:
                item=matching[-1]
                status=item.get('status')
                if status != previous_status:
                    print(patient,status, 'bytes', item.get('bytes_written'), '/', item.get('bytes_expected'),flush=True)
                    previous_status = status
                if status in ['completed','failed','cancelled']:
                    break
        rows.append({'patient_id':patient,'status':status})
        Path('outputs/ipr/connect_transfer_attempt.json').write_text(json.dumps({'scope':'real_development_download_attempt','transfers':rows},indent=2))
        if status!='completed':break
    output=Path('outputs/ipr/connect_transfer_attempt.json')
    output.write_text(json.dumps({'scope':'real_development_download_attempt','transfers':rows},indent=2))
    print(output,flush=True)


if __name__=='__main__':main()
