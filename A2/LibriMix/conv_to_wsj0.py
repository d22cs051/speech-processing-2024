from tqdm.auto import tqdm
import os
import shutil

def convert_libri2mix_to_wsj0_2mix(libri2mix_path, wsj0_2mix_path):
    try:
        # Create wsj0-2mix directory structure
        for split in ["tr", "cv"]:
            for subdir in ["mix_both", "mix_clean", "s1", "s2"]:
                os.makedirs(os.path.join(wsj0_2mix_path, split, subdir), exist_ok=True)

        # Traverse Libri2Mix and copy files to wsj0-2mix structure
        for root, dirs, files in os.walk(libri2mix_path):
            if "test" in root or "train" in root:
                split = "tr" if "train" in root else "cv"
                for subdir in ['mix_both', 'mix_clean', 's1', 's2']:
                    source_dir = os.path.join(root, subdir)
                    if not os.path.exists(source_dir):
                        print(f"Skipping directory: {source_dir}")
                        continue
                    target_dir = os.path.join(wsj0_2mix_path, split, subdir)
                    os.makedirs(target_dir, exist_ok=True)
                    for file in os.listdir(source_dir):
                        source_path = os.path.join(source_dir, file)
                        target_path = os.path.join(target_dir, file)
                        shutil.copyfile(source_path, target_path)
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to Libri2Mix dataset
libri2mix_path = "Libri2Mix"

# Path to save wsj0-2mix formatted dataset
wsj0_2mix_path = "wsj0-2mix"

convert_libri2mix_to_wsj0_2mix(libri2mix_path, wsj0_2mix_path)
print("Conversion complete!")