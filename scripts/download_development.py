"""Resume acquisition of the five required files for every development patient."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from glioma_idh.data import sha256, write_json
from glioma_idh.utsw_imaging import UTSW_FILES
from download_via_connect import APP_ID, SESSION, local
from tcia_public import authenticate, request


def missing_inputs(root: Path, patients: list[str]) -> list[dict]:
    missing = []
    for patient in patients:
        if not patient.startswith('BT') or not patient[2:].isdigit():
            raise ValueError('Unexpected UTSW patient ID.')
        for name in UTSW_FILES.values():
            path = root / patient / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append({'patient_id': patient, 'filename': name})
    return missing


def map_destinations(spec: dict, expected: list[dict]) -> None:
    destinations = {f"{row['patient_id']}/{row['filename']}" for row in expected}
    actual = []
    for path in spec['paths']:
        destination = '/'.join(path['source'].split('/')[-2:])
        if destination not in destinations:
            raise ValueError('Transfer specification contains an unexpected source file.')
        path['destination'] = destination
        actual.append(destination)
    if len(actual) != len(set(actual)) or set(actual) != destinations:
        raise ValueError('Transfer specification does not match the requested files.')


def wait_for_transfer(request_id: str, state: dict, state_path: Path, timeout: int) -> str:
    started = time.monotonic()
    previous = None
    missing_polls = 0
    while time.monotonic() - started < timeout:
        activities = local('/transfers/activity', {'iteration_token': 0})
        matching = [item for item in activities.get('transfers', [])
                    if item.get('aspera_connect_settings', {}).get('request_id') == request_id]
        if matching:
            missing_polls = 0
            item = matching[0]
            state.update({key: item.get(key) for key in ['uuid', 'status', 'bytes_written', 'bytes_expected']})
            write_json(state_path, state)
            progress = (state['status'], int((state.get('bytes_written') or 0) / 50_000_000))
            if progress != previous:
                print('Transfer', state['status'], state.get('bytes_written'), '/', state.get('bytes_expected'), flush=True)
                previous = progress
            if state['status'] in {'completed', 'failed', 'cancelled'}:
                return state['status']
        else:
            missing_polls += 1
            if missing_polls >= 6:
                state['status'] = 'not_found'
                write_json(state_path, state)
                return state['status']
        time.sleep(5)
    state['status'] = 'monitor_timeout'
    write_json(state_path, state)
    return state['status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('outputs/development/acquisition'))
    parser.add_argument('--transfer-timeout', type=int, default=1800)
    parser.add_argument('--select-folder', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.transfer_timeout < 30:
        parser.error('Batch size must be positive and transfer timeout at least 30 seconds.')
    session = json.loads(SESSION.read_text())
    if args.select_folder:
        chosen = local('/windows/select-open-folder-dialog/',
                       {'title': 'Select development MRI download folder', 'allowMultipleSelection': False}, timeout=180)
        selected = chosen.get('dataTransfer', {}).get('files', [])
        if len(selected) != 1:
            raise ValueError('One project download folder must be selected.')
        destination = Path(selected[0]['name']).resolve()
        if not destination.is_relative_to(Path.cwd().resolve()):
            raise ValueError('Select a folder inside the project workspace.')
        session['destination'] = str(destination)
        write_json(SESSION, session)
    root = Path(session.get('destination', 'dataset')).resolve()
    development = pd.read_csv('outputs/week1/development_patients.csv').sort_values('patient_id')
    lock_path = Path('outputs/week1/split_lock.json')
    lock = json.loads(lock_path.read_text())
    if set(development.patient_id) != set(lock['development_ids']):
        raise ValueError('Development table differs from the original patient lock.')
    if set(development.patient_id) & set(lock['test_ids']):
        raise ValueError('Reserved patient entered development acquisition.')
    args.output.mkdir(parents=True, exist_ok=True)
    development[['patient_id', 'idh_label', 'fold']].to_csv(args.output / 'selection.csv', index=False)
    state_path = args.output / 'transfer_state.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('status') in {'submitted', 'initiating', 'running', 'queued', 'monitor_timeout'}:
            status = wait_for_transfer(previous['request_id'], previous, state_path, args.transfer_timeout)
            if status not in {'completed', 'failed', 'cancelled', 'not_found'}:
                raise RuntimeError('The existing transfer still needs monitoring before another starts.')
    api, headers, package = authenticate()
    patients = development.patient_id.tolist()
    for start in range(0, len(patients), args.batch_size):
        batch = patients[start:start + args.batch_size]
        pending = missing_inputs(root, batch)
        if not pending:
            continue
        for patient in batch:
            (root / patient).mkdir(parents=True, exist_ok=True)
        payload = {'paths': [{'path': f"/UTSW-Glioma/{row['patient_id']}/{row['filename']}"} for row in pending]}
        endpoint = f'/packages/{package}/transfer_spec/download?transfer_type=connect&type=received'
        try:
            spec = json.loads(request(api + endpoint, payload, headers).read())
        except HTTPError as error:
            if error.code != 401:
                raise
            api, headers, package = authenticate()
            spec = json.loads(request(api + endpoint, payload, headers).read())
        map_destinations(spec, pending)
        spec.update({'destination_root': str(root), 'create_dir': True,
                     'resume': 'sparse_checksum', 'overwrite_policy': 'diff'})
        request_id = str(uuid.uuid4())
        settings = {'app_id': APP_ID, 'request_id': request_id, 'allow_dialogs': True,
                    'back_link': 'https://faspex.cancerimagingarchive.net', 'use_absolute_destination_path': True}
        state = {'request_id': request_id, 'status': 'submitted', 'patients': batch,
                 'requested_files': len(pending), 'destination': str(root),
                 'split_lock_sha256': sha256(lock_path)}
        write_json(state_path, state)
        print('Starting development batch', start + 1, 'to', start + len(batch), 'files', len(pending), flush=True)
        try:
            local('/transfers/start', {'transfer_specs': [{'transfer_spec': spec, 'aspera_connect_settings': settings}]}, timeout=60)
        except TimeoutError:
            pass
        except RuntimeError:
            state['status'] = 'start_failed'
            write_json(state_path, state)
            raise
        status = wait_for_transfer(request_id, state, state_path, args.transfer_timeout)
        if status != 'completed' or missing_inputs(root, batch):
            raise RuntimeError(f'Batch ended with {status}. The recorded transfer can be resumed.')
        complete = sum(not missing_inputs(root, [patient]) for patient in patients)
        write_json(args.output / 'progress.json', {'selected_patients': len(patients),
                   'complete_file_sets': complete, 'reserved_patients_requested': 0,
                   'required_files_per_patient': 5, 'destination': str(root)})
        print('Complete development file sets', complete, '/', len(patients), flush=True)
    write_json(args.output / 'progress.json', {'selected_patients': len(patients),
               'complete_file_sets': len(patients), 'reserved_patients_requested': 0,
               'required_files_per_patient': 5, 'destination': str(root)})
    print('Development acquisition complete', len(patients), flush=True)


if __name__ == '__main__':
    main()
