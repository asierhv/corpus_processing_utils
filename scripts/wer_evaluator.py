import jiwer
from normalizer import TextNormalizer

def calculate_wer(data, normalizer = TextNormalizer, pred_text_tag: str="pred_text", manifest_name: str="sample.json", verbose: bool=True, return_wer: bool=False):
    """ 
    Calculate WER with and without C&P 
    Returns the original data with those 2 calculated WERs per item
    It also returns the total and mean WER for the dataset
    """
    normalizer_params = vars(normalizer).copy()
    normalizer_params["tag"] = pred_text_tag
    _ = normalizer_params.pop('unclean_char_list')
    _ = normalizer_params.pop('clean_char_list')
    secondary_normalizer = TextNormalizer(**normalizer_params)
    wer_list = []
    wer_cp_list = []
    data_clean = [dict(item) for item in data]
    data_clean = normalizer.clean_sentences(data_clean)
    if secondary_normalizer:
        data_clean = secondary_normalizer.clean_sentences(data_clean)
    new_data = []
    total_reference = ""
    total_reference_clean = ""
    total_pred = ""
    total_pred_clean = ""
    for i,item in enumerate(data):
        item_clean = data_clean[i]
        # Calculate wer with C&P for each sentence
        wer_cp = jiwer.wer(item["text"], item["pred_text"].strip())
        wer_cp_list.append(wer_cp)
        item['wer_cp'] = wer_cp
        # Calculate normalized wer for each sentence
        wer = jiwer.wer(item_clean["text"], item_clean["pred_text"].strip())
        wer_list.append(wer)
        item['wer'] = wer
        new_data.append(item)
        total_reference += " " + item["text"].strip()
        total_reference_clean += " " + item_clean["text"].strip()
        total_pred += " " + item["pred_text"].strip()
        total_pred_clean += " " + item_clean["pred_text"].strip()
    total_wer_cp = jiwer.wer(total_reference.strip(), total_pred.strip())
    total_wer = jiwer.wer(total_reference_clean.strip(), total_pred_clean.strip())
    mean_wer_cp = sum(wer_cp_list)/len(wer_cp_list)   
    mean_wer = sum(wer_list)/len(wer_list)
    if verbose:
        print(f"=============[ {manifest_name} ]=============")
        print(f"\t Mean WER C&P: {round(mean_wer_cp*100,2)} %")
        print(f"\t     Mean WER: {round(mean_wer*100,2)} %")
        print(f"\tTotal WER C&P: {round(total_wer_cp*100,2)} %")
        print(f"\t    Total WER: {round(total_wer*100,2)} %")
        print(f"==============={'='*len(manifest_name)}===============")
        
    result = {
        "filename": manifest_name.replace(".json",""),
        "mean_wer_cp": mean_wer_cp,
        "mean_wer": mean_wer,
        "total_wer_cp": total_wer_cp,
        "total_wer": total_wer
        }
    
    if return_wer: 
        return new_data, result