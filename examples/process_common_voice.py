import os
import corpus_utils as cu
from corpus_normalizer import TextNormalizer
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

path = "./common_voice_v18/eu" # Path to the directory where the comomn_voice files are

validated_tsv = f"{path}/validated.tsv" # Contains all the data (train + test + dev + others)
test_tsv = f"{path}/test.tsv" # Contains only the test data
dev_tsv = f"{path}/dev.tsv" # Contains only the dev data

clips_folder = f"{path}/clips"
manifests_path = f"./manifests"
os.makedirs(manifests_path, exist_ok=True)

normalizer = TextNormalizer(lang="eu", remove_acronyms=True, verbose='all')

dataset={}
for tsv in [validated_tsv, test_tsv, dev_tsv]:
    name = os.path.split(tsv)[1][:-4]
    manifest_filepath = f"{manifests_path}/{name}.json"
    data = cu.tsv2data(tsv_filepath=tsv, clips_folder=clips_folder, audio_field="path", calculate_duration=False, text_field="sentence")

    # Clean and write the manifest
    data = normalizer(data)
    cu.write_manifest(manifest_filepath, data)

    dataset[name] = data

validated_data = dataset['validated']
test_data = dataset['test']
dev_data = dataset['dev']

# Reduce dev split size and write
random.shuffle(dev_data)
dev_s_data = dev_data[:620]
cu.write_manifest(f"{manifests_path}/dev-s.json", dev_s_data)

# Clean validated data from test and dev-s data
compare_data = test_data + dev_s_data
train_data = cu.reduce_data(data=validated_data,compare_data=compare_data)
cu.write_manifest(f"{manifests_path}/train.json", train_data)

# Show statistics
cu.manifest_time_stats(f"{manifests_path}/train.json")
cu.manifest_time_stats(f"{manifests_path}/test.json")
cu.manifest_time_stats(f"{manifests_path}/dev-s.json")




