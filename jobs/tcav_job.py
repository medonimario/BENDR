import os, time, datetime
os.chdir('/zhome/33/6/147533/BENDR/')

now = datetime.datetime.now()
now_str = now.strftime("%H%M%S_%d%m%y")

data_path = '/work1/s194260/'
concepts_folder = f"TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523"

concepts = ["Alpha_Dorsal Stream Visual Cortex-lh", "Alpha_Dorsal Stream Visual Cortex-rh",
            "Alpha_Early Visual Cortex-lh", "Alpha_Early Visual Cortex-rh",
            "Alpha_MT+ Complex and Neighboring Visual Areas-lh", "Alpha_MT+ Complex and Neighboring Visual Areas-rh",
            "Alpha_Premotor Cortex-lh", "Alpha_Premotor Cortex-rh",
            "Alpha_Primary Visual Cortex (V1)-lh","Alpha_Primary Visual Cortex (V1)-rh",
            "Alpha_Somatosensory and Motor Cortex-lh", "Alpha_Somatosensory and Motor Cortex-rh",
            "Alpha_Ventral Stream Visual Cortex-lh", "Alpha_Ventral Stream Visual Cortex-rh"]

concepts = "'" + "' '".join(concepts) + "'"

name = f"BENDR"

job = f"""#!/bin/sh
#BSUB -J {name}
#BSUB -q hpc
#BSUB -n 20
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -o logs/output_{name}_%J.out 
#BSUB -e logs/error_{name}_%J.err 
module load scipy/1.9.1-python-3.10.7
module load cuda/11.7 
source /work1/s194260/BENDR-ENV/bin/activate
python3.10 tcav/tcav_run.py --concept_folder {concepts_folder} --concepts {concepts} --data_path {data_path}"""

with open('temp_submit.sh', 'w') as file:
    file.write(job)

os.system('bsub < temp_submit.sh')
time.sleep(0.5)
os.remove('temp_submit.sh')