import json
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

import socket

hostname = socket.gethostname()

json_f_dict = {
    'yenchi-pc': {
        'abc': 'info-yc-abc.json',
        'shapenet': '.json',
        'pix3d': 'info-euclid-pix3d.json'
    }, 
    'euclid': {
        'abc': 'info-euclid-abc.json',
        'shapenet': '.json',
        'pix3d': 'info-euclid-pix3d.json'
    }, 
    'u110459': {
        'abc': 'info-euclid-abc.json',
        'shapenet': '.json',
        'pix3d': 'info-euclid-pix3d.json'
    }, 
    'ip-172-17-18-173': {
        'abc': 'info-euclid-abc.json',
        'shapenet': '.json',
        'pix3d': 'info-euclid-pix3d.json'
    }, 
}

def get_all_info(dset):
    json_f = os.path.join(CUR_PATH, 'info_files', json_f_dict[hostname][dset])

    with open(json_f) as json_file:
        data = json.load(json_file)
        # lst_dir, cats, all_cats, raw_dirs = data["lst_dir"], data['cats'], data['all_cats'], data["raw_dirs_v1"]
        if dset == 'abc':
            lst_dir, cats, all_cats, raw_dirs = None, None, None, data['raw_dirs_v1']
            
        elif dset == 'pix3d':
            lst_dir, cats, all_cats, raw_dirs = data["lst_dir"], data['cats'], data['all_cats'], data['raw_dirs_v1']
    return lst_dir, cats, all_cats, raw_dirs

if __name__ == "__main__":

    # nohup python -u create_file_lst.py &> create_imgh5.log &

    # lst_dir, cats, all_cats, raw_dirs = get_all_info()
    cats, all_cats, raw_dirs = get_all_info()
