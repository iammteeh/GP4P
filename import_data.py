import os
import mimetypes
import zipfile
from env import DATADIR

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
    sws_dir = ""
    if not sws:
        sws = list_sws()
        for i, sys in enumerate(sws):
            print(i, sys)
        choice = int(input('Select a data set: '))
        if not os.path.isdir(DATADIR + sws[choice].split('/')[-1].split('.')[0]):
            sws_dir = extract_zip(sws[choice])
        else:
            sws_dir = sws[choice].split('/')[-1].split('.')[0]
    else:
        if not os.path.isdir(DATADIR + '/' + sws):
            if sws in list_sws():
                sws_dir = extract_zip(DATADIR + sws + '.zip')
            else:
                print('Data set not found')
                exit()
    data_header = {
        'sws_name': sws_dir,
        'sws_path': DATADIR + sws_dir,
        'measurements_file': DATADIR + sws_dir + '/measurements.csv',
        'measurements_file_cleared': DATADIR + sws_dir + '/measurements-cleared.csv',
        'featuremodel': DATADIR + sws_dir + '/featuremodel.xml',
    }
    return data_header
