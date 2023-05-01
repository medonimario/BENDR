import os
from tqdm import tqdm
if not os.getcwd().endswith('BENDR'): os.chdir(os.path.dirname(os.getcwd()))
import random
import shutil

source_dir = "/work1/s194260/TUH_clean_all_divide_Falsesigma_Falseabs_100.0_210702_010523"

concepts = [concept for concept in os.listdir(source_dir) if 'random' not in concept 
            and 'Left' not in concept and 'Right' not in concept
            and len(os.listdir(os.path.join(source_dir, concept))) > 2]

for i in range(50):

    target_dir = source_dir + f'/random500_{i}'
    count = 0

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

        with tqdm(total=30) as pbar:
            while count <= 30:
                concept = random.choice(concepts)
                files = [file for file in os.listdir(os.path.join(source_dir, concept))] # if 'Alpha' in file]

                if len(files) > 1:
                    random_file = random.sample(files, 1)[0]
                    shutil.copy(os.path.join(source_dir, concept, random_file), os.path.join(target_dir, concept))
                    count += 1
                    pbar.update(1)
                else:
                    continue