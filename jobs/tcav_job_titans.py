import os, time, datetime, copy
os.chdir('/home/s194260/BENDR')

now = datetime.datetime.now()
now_str = now.strftime("%H%M%S_%d%m%y")

data_path = '/scratch/s194260/'
# concepts_folders = ['TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_10.0_221052_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_50.0_221427_010523',
#  'TUH_clean_all_divide_Falsesigma_Falseabs_50.0_221859_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_100.0_211218_010523',
#  'TUH_clean_all_subtract_Truesigma_Falseabs_100.0_211650_010523',
#  'TUH_clean_all_subtract_Falsesigma_Falseabs_100.0_213033_010523',
#  'TUH_clean_alpha_subtract_Falsesigma_Falseabs_100.0_213925_010523',
#  'TUH_clean_alpha_divide_Falsesigma_Falseabs_100.0_214308_010523',
#  'TUH_clean_all_subtract_Falsesigma_Trueabs_100.0_212500_010523',
#  'TUH_clean_all_subtract_Truesigma_Trueabs_100.0_212049_010523',
#  'TUH_clean_alpha_subtract_Truesigma_Trueabs_100.0_215437_010523',
#  'TUH_clean_alpha_subtract_Truesigma_Falseabs_100.0_215052_010523',
#  'TUH_clean_alpha_subtract_Falsesigma_Trueabs_100.0_220101_010523',
#  'TUH_clean_all_divide_Falsesigma_Falseabs_10.0_220629_010523',
#  'TUH_clean_alpha_divide_Truesigma_Falseabs_100.0_214650_010523']

# concepts_folders = ['TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_10.0_221052_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_50.0_221427_010523',
#  'TUH_clean_all_divide_Truesigma_Falseabs_100.0_211218_010523',
#  'TUH_clean_all_divide_Falsesigma_Falseabs_50.0_221859_010523',
#  'TUH_clean_all_divide_Falsesigma_Falseabs_10.0_220629_010523']

#concepts_folders = ["TUH_clean_all_subtract_Truesigma_Trueabs_100.0_212049_010523", "TUH_clean_all_subtract_Falsesigma_Trueabs_100.0_212500_010523",
#                    "TUH_clean_all_divide_Truesigma_Falseabs_50.0_221427_010523", "TUH_clean_all_divide_Truesigma_Falseabs_10.0_221052_010523"]
#concepts_folders = ["TUH_clean_alpha_divide_Falsesigma_Falseabs_100.0_214308_010523", "TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523"]

#concepts_folders = ['TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523']

concepts = ["Alpha_Dorsal Stream Visual Cortex-lh", "Alpha_Dorsal Stream Visual Cortex-rh",
            "Alpha_Early Visual Cortex-lh", "Alpha_Early Visual Cortex-rh",
            "Alpha_Premotor Cortex-lh", "Alpha_Premotor Cortex-rh",
            "Alpha_Primary Visual Cortex (V1)-lh","Alpha_Primary Visual Cortex (V1)-rh",
            "Alpha_Somatosensory and Motor Cortex-lh", "Alpha_Somatosensory and Motor Cortex-rh"]

concepts_folders = ['TUH_clean_alpha_subtract_Truesigma_Trueabs_100.0_215437_010523']

concepts = "'" + "' '".join(concepts) + "'"

for concepts_folder in concepts_folders:
    concepts_tmp = []
    
    # Copy the folder '/scratch/s194260/Left fist, imagined' and its context into every folder beginning with TUH_clean in /scratch/s194260
    for folder in os.listdir('/scratch/s194260/' + concepts_folder):
        if folder in concepts:
            if len(os.listdir('/scratch/s194260/' + concepts_folder + '/' + folder)) >= 25:
                concepts_tmp.append(folder)
            else:
                print(folder)
                
    concepts_tmp = "'" + "' '".join(concepts_tmp) + "'"    
    
    _, _, a, b, c, d, e, _, _ = concepts_folder.split('_')
    name = '_'.join([a, b, c, d, e])
    #name = "MMIDB"
    
    job = f"""#!/bin/sh
    #SBATCH --job-name={name}
    #SBATCH --output=/home/s194260/BENDR/logs/output_{name}_%J.out 
    #SBATCH --cpus-per-task=3
    #SBATCH --time=90:00
    #SBATCH --mem=16gb

    source ~/.bashrc
    conda activate BENDR

    python3.10 tcav/tcav_run.py --concept_folder {concepts_folder} --concepts {concepts_tmp} --data_path {data_path}"""

    #print(job)

    with open('temp_submit.s', 'w') as file:
        file.write(job)

    os.system('sbatch temp_submit.s')
    time.sleep(0.5)
    os.remove('temp_submit.s')