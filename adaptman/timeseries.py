"""Extract region timeseries for analysis"""

import os
import subprocess
import pandas as pd

from adaptman.config import Config
from adaptman.utils import get_files, display

pjoin = os.path.join

def main():

    config = Config()
    subjects = pjoin(config.resources, 'valid_subjects.tsv')
    out_dir = config.dataset_dir

    # extract only qualified subjects
    subjects = pd.read_table(subjects)['sub_id'].tolist()
    input_files = []
    regressor_files = []
    for sub in subjects:
        sub_dir = pjoin(config.fmriprep_dir, sub)
        funcs = get_files(pjoin(sub_dir, '*/func/*bold.dtseries.nii'))
        input_files.extend(funcs)
        
        confs = get_files(pjoin(sub_dir, '*/func/*regressors.tsv'))
        regressor_files.extend(confs)
    
    # extract
    n_jobs = 45 # dat HCP firepower
    config_file = pjoin(config.resources, 'nixtract_config.json')
    nixtract_cmd = (f"nixtract-cifti {out_dir} --roi_file {config.atlas} "
                    f"--input_files {' '.join(input_files)} "
                    f"--regressor_files {' '.join(regressor_files)} "
                    f"-c {config_file} --n_jobs {n_jobs}")
    subprocess.run(nixtract_cmd.split())

    # quality assessment for each session separately
    display('QC assessment')
    schaefer = 'Schaefer2018_1000Parcels_7Networks_corrected.nii.gz'
    dist_file = pjoin(config.resources, 'atlases', schaefer)

    for ses in ['ses-01', 'ses-02']:
        tseries = " ".join(get_files(pjoin(out_dir, f'*{ses}*.tsv')))
        regs = [x for x in regressor_files if ses in x]
        qc_cmd = (f"nixtract-qc -t {tseries} -c {' '.join(regs)} -o "
                f"{out_dir}/qc-{ses} --distance_file {dist_file} "
                f"--n_jobs {n_jobs}")
        subprocess.run(qc_cmd.split())


if __name__ == '__main__':
    main()