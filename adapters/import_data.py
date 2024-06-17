import os
import mimetypes
import zipfile
from domain.env import DATADIR

def list_sws():
    sws = []
    for item in os.listdir(DATADIR):
        if mimetypes.guess_type(item)[0] == 'application/zip':
            sws.append(DATADIR + item)
    return sws

def extract_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(DATADIR)
        path, = zipfile.Path(zip_ref).iterdir()
        print(f"Extracted {path.name} to {DATADIR}")
        return path.name

def select_data(sws=None):
    selected_sws = ""
    if not sws:
        sws = list_sws()
        for i, sys in enumerate(sws):
            print(i, sys)
        choice = int(input('Select a data set: '))
        if not os.path.isdir(DATADIR + sws[choice].split('/')[-1].split('.')[0]):
            selected_sws = extract_zip(sws[choice])
        else:
            selected_sws = sws[choice].split('/')[-1].split('.')[0]
    else:
        print(f"Checking if {sws} isn't already extracted")
        if not os.path.isdir(DATADIR + '/' + sws):
            print(f"available sws: {list_sws()}")
            if DATADIR + sws + '.zip' in list_sws():
                print(f"Found {sws} in {DATADIR}")
                selected_sws = extract_zip(DATADIR + sws + '.zip')
            else:
                print('Data set not found')
                exit(1)
        else:
            selected_sws = sws
    data_header = {
        'sws_name': selected_sws,
        'sws_path': DATADIR + selected_sws,
        'measurements_file': DATADIR + selected_sws + '/measurements.csv',
        'measurements_file_cleared': DATADIR + selected_sws + '/measurements-cleared.csv',
        'featuremodel': DATADIR + selected_sws + '/featuremodel.xml',
    }
    return data_header
