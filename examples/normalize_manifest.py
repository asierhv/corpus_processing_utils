import logging
from corpus_normalizer import TextNormalizer
import corpus_utils as cu

logging.basicConfig(level=logging.INFO, format="%(message)s")
    
blacklist_terms =[
    "\(inint\)", "\(inint\(", "\(Inint\)", "\(init\)",
    "\(gabe\)","\(Many speakers\)",
    "\(Ri\)","\(RI\)","\(RU\)",
    "\(MU\)","\(LL\)","\(BO\)","\-c\}","\-n\}"
]
json = "example_es.json"
data = cu.read_manifest(f"./manifests/{json}")
eu_normalizer = TextNormalizer(lang='es', keep_cp=False, blacklist_terms=blacklist_terms, verbose='all')
clean_data = eu_normalizer(data)
json_clean=json.replace(".json","_clean.json")
cu.write_manifest(f"./manifests/processed/{json_clean}", clean_data)