from tqdm.auto import tqdm
import pandas as pd
from verification import verification_batch
import os

df = pd.read_csv("/DATA1/bikash_dutta/CS/SP/A2/UniSpeech/downstreams/speaker_verification/hindi_test_known.csv")

# print(df.head())


# testing for 100 files
# df = df[:15]

# add column to dataframe with verification score b/w i.e -1 and 1 for each model name in the list
model_names = ['ecapa_tdnn', 'hubert_large', "wavlm_base_plus", "wavlm_large"]
batch_size = 128  # batch size for verification

for model_name in model_names:
    file_not_found_count = 0
    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {model_name}"):
        batch = df.iloc[i:i+batch_size]
        batch_files1 = []
        batch_files2 = []
        for index, row in batch.iterrows():
            file_path_1 = f"{row['person1']}"
            file_path_2 = f"{row['person2']}"
            if not os.path.exists(file_path_1): # if file is not in val split
                file_path_1 = f"{row['person1']}"
                if not os.path.exists(file_path_1):
                    print(f"File not found: {file_path_1}")
                    file_not_found_count += 1
                    continue
            
            if not os.path.exists(file_path_2): # if file is not in val split
                file_path_2 = f"{row['person2']}"
                if not os.path.exists(file_path_2):
                    print(f"File not found: {file_path_2}")
                    file_not_found_count += 1
                    continue

            batch_files1.append(file_path_1)
            batch_files2.append(file_path_2)

        if len(batch_files1) == 0 or len(batch_files2) == 0:
            continue

        verification_scores = verification_batch(model_name=model_name, batch_wav1=batch_files1, batch_wav2=batch_files2, use_gpu=True)

        # print(f"Verification scores: {verification_scores}")
        # adding verification scores to dataframe
        for j, score in enumerate(verification_scores):
            df.at[i+j, model_name] = score

    if file_not_found_count > 0:
        print(f"File not found: {file_not_found_count}")

# save dataframe to csv
df.to_csv(f"verification_scores_hindi_batch_{batch_size}.csv",index=False)