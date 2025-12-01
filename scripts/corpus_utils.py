import json
import os
import statistics
import openpyxl
import logging
import jiwer
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from typing import Union, Optional, List, Tuple
from corpus_normalizer import TextNormalizer
from corpus_types import ItemManifest, ExcelRecord

def read_manifest(manifest_filepath: str, verbose: bool = True) -> List[ItemManifest]:
    """
    Read a manifest file (JSONL format) into a list of dictionaries.

    Parameters
    ----------
    manifest_filepath : str
        Path to the manifest file. Each line of the file should be a valid JSON object.
    
    verbose : bool, optional (default=True)
        If True, logs a message indicating which file is being read.

    Returns
    -------
    list of dict
        A list where each element is a dictionary representing one line from the manifest.

    Raises
    ------
    Exception
        If the file cannot be opened or read.
    
    Notes
    -----
    The manifest file is expected to be in JSON Lines format (one JSON object per line).
    """
    if verbose==True:
        logging.info(f"Reading: {manifest_filepath}")
    try:
        f = open(manifest_filepath, 'r', encoding='utf-8')
        data = [json.loads(line) for line in f]
        f.close()
        return data
    except:
        raise Exception(f"Manifest file could not be opened: {manifest_filepath}")

def write_manifest(
    manifest_filepath: str,
    data: List[ItemManifest],
    ensure_ascii: bool = False,
    return_manifest_filepath: bool = False,
    verbose: bool = True
    ) -> Optional[str]:
    """
    Write a list of dictionaries to a manifest file in JSONL format.

    Parameters
    ----------
    manifest_filepath : str
        Path where the manifest file will be written.
    
    data : list of dict
        List of dictionaries to write to the file. Each dictionary will be written as a JSON object on a separate line.
    
    ensure_ascii : bool, optional (default=False)
        If True, the output will have all non-ASCII characters escaped. Otherwise, non-ASCII characters are written as-is.

    return_manifest_filepath : bool, optional (default=False)
        If True, the function returns the path of the written manifest file. Otherwise, returns None.

    verbose : bool, optional (default=True)
        If True, logs a message when writing is finished.

    Returns
    -------
    str or None
        The manifest file path if `return_manifest_filepath=True`, else None.

    Notes
    -----
    This function overwrites the file if it already exists.
    Each dictionary in `data` is written as a single line in JSON format.
    """
    f = open(manifest_filepath, "w", encoding="utf-8")
    for item in data:
        f.write(json.dumps(item,ensure_ascii=ensure_ascii) + "\n")
    f.close()
    if verbose==True:
        logging.info(f"End Writing manifest: {manifest_filepath}")
    if return_manifest_filepath:
        return manifest_filepath
    else:
        return None

def tsv2data(
    tsv_filepath: str,
    clips_folder: str = "",
    sep: str = "\t",
    audio_field: Union[str, int] = "path",
    text_field: Union[str, int] = "sentence",
    duration_field: Optional[Union[str,int]] = None,
    calculate_duration: bool = False,
    header: Optional[Union[int, list, str]] = 'infer'
    ) -> List[ItemManifest]:
    """
    Reads a TSV file containing audio file paths and corresponding text sentences,
    returning a structured list of dictionaries suitable for downstream processing.

    Each row in the TSV represents a single audio-text pair. Optionally, if a column
    for precomputed durations is provided, it will be used; otherwise, durations can
    be calculated directly from the audio files using the `soundfile` library.

    Parameters
    ----------
    tsv_filepath : str
        Path to the TSV file to read.

    clips_folder : str, optional (default="")
        Base folder to prepend to audio file paths. Useful if the paths in the TSV
        are relative.

    sep : str, optional (default="\t")
        Column separator in the TSV file.

    audio_field : str or int, optional (default="path")
        Column containing audio file paths. Can be a string (column name) if headers
        exist, or an integer index if the TSV has no headers.

    text_field : str or int, optional (default="sentence")
        Column containing the corresponding text for each audio file. Can be a string
        or integer index.

    duration_field : str or int, optional (default=None)
        Column containing precomputed duration values. If not provided, durations
        will be calculated if `calcuate_duration` is True.

    calculate_duration : bool, optional (default=False)
        Whether to compute audio durations from the audio files using `soundfile`.
        If False, duration will be set to None unless `duration_field` is provided.

    header : int, list of int, 'infer', or None, optional (default='infer')
        Row(s) to use as the column names. Follows the same convention as `pandas.read_csv`.
        Set to None if the TSV has no headers.

    Returns
    -------
    list of dict
        Each dictionary contains:
            - 'audio_filepath': full path to the audio file (str)
            - 'text': corresponding text (str)
            - 'duration': duration in seconds (float) or None if not calculated

    Notes
    -----
    - Raises an Exception if an audio file cannot be opened when duration calculation
      is enabled.
    - The `clips_folder` is prepended to each audio path using `os.path.join`.
    """
    data=[]
    df = pd.read_csv(tsv_filepath, sep=sep, header=header)
    for idx in tqdm(range(len(df))):
        audio_filepath = os.path.join(clips_folder,df[audio_field][idx])
        text = df[text_field][idx]
        if calculate_duration:
            if duration_field:
                try:
                    duration = df[duration_field][idx]
                except:
                    raise Exception(f"Audio file could not be opened: {audio_filepath}")
            else:
                f = sf.SoundFile(audio_filepath)
                duration = len(f) / f.samplerate
        else:
            duration = None
        item = {
            'audio_filepath': audio_filepath,
            'text': text,
            'duration': duration
        }
        data.append(item)
    return data
    
def pairedfiles2data(clips_folder: str, sentences_folder: str) -> List[ItemManifest]:
    """
    Build a structured dataset by pairing text files with their corresponding audio files.

    This function scans `sentences_folder` for `.txt` files and expects each one to have
    a `.wav` file with the same base name in `clips_folder`. For every valid pair, it
    loads the text, opens the audio file to retrieve its duration, and returns a dataset
    where each entry contains:

        - audio_filepath : full path to the `.wav` file
        - text           : sentence string loaded from the `.txt` file
        - duration       : audio duration in seconds

    Parameters
    ----------
    clips_folder : str
        Path to the directory containing `.wav` audio files.
    sentences_folder : str
        Path to the directory containing `.txt` sentence files.

    Returns
    -------
    list of dict
        A list of paired items. Each dictionary contains:
            {
                'audio_filepath': str,
                'text': str,
                'duration': float
            }

    Raises
    ------
    Exception
        If a corresponding audio file cannot be opened or does not exist.

    Notes
    -----
    - `.txt` filenames must match the `.wav` filenames (same stem).
    - Duration is obtained using `sf.SoundFile`, which must be available and functional.
    """
    data=[]
    for file in os.listdir(sentences_folder):
        if file.endswith(".txt"):
            audio_filepath = os.path.join(clips_folder,file.replace(".txt",".wav"))
            try:
                duration = sf.SounFile(audio_filepath)
            except:
                raise Exception(f"Audio file could not be opened: {audio_filepath}")
            f = open(os.path.join(sentences_folder,file),"r",encoding="utf-8")
            sentence = f.read()
            item = {
                'audio_filepath': audio_filepath,
                'text': sentence,
                'duration': duration,
            }
            data.append(item)
    return data

def hash_sentences(data: List[ItemManifest]) -> List[int]:
    """
    Compute hash values for the `"text"` field of each item in a dataset.

    Parameters
    ----------
    data : list of dict
        A list of items where each item must contain a `"text"` key.

    Returns
    -------
    list of int
        A list of hash values corresponding to `hash(item["text"])` for each item
        in `data`.

    Notes
    -----
    - The built-in `hash()` function is used, so hash values may differ across
      Python sessions unless hash randomization is disabled.
    - This function is typically used to accelerate duplicate detection or
      cross-dataset comparisons.
    """
    hashed_sentences = [hash(item["text"]) for item in tqdm(data)]
    return hashed_sentences

def reduce_data(
    data: List[ItemManifest],
    compare_data: Optional[List[ItemManifest]] = None,
    hashed_data: Optional[List[int]] =None,
    hashed_compare: Optional[List[int]] =None
    ) -> List[ItemManifest]:
    """
    Reduce a dataset by removing duplicates or by filtering out items
    that appear in another dataset. Hashes of the `"text"` field are used
    for fast comparison.

    Parameters
    ----------
    data : list of dict
        The primary dataset to reduce. Each item must contain a `"text"` key.
    compare_data : list of dict, optional
        If provided, items from `data` whose `"text"` hashes appear in
        `compare_data` will be removed. If None, the function removes
        duplicates within `data` itself.
    hashed_data : list of int, optional
        Precomputed hash values for each item in `data`. Must be aligned by index.
        If not provided, hashes are computed internally as `hash(item["text"])`.
    hashed_compare : list of int, optional
        Precomputed hash values for each item in `compare_data`. Used only when
        `compare_data` is provided. If None, hashes are computed automatically.

    Returns
    -------
    list of dict
        The reduced dataset with duplicates removed or filtered based on
        `compare_data`.

    Notes
    -----
    - Duplicate detection relies solely on the hash of the `"text"` field.
      If two items have identical text, only the first is kept.
    - When `compare_data` is provided, the function removes all `data` items
      whose `"text"` hash is found in `compare_data`.
    - The function logs the number and percentage of removed items.
    """
    logging.info("::::: Reducing dataset :::::")
    if hashed_data is None:
        hashed_data = hash_sentences(data)
    datalen = len(data)
    if compare_data is None:
        # Remove duplicates within the same dataset
        seen_hashes = set()
        reduced_data = []
        for h, item in tqdm(zip(hashed_data, data), total=datalen, desc="Removing duplicates"):
            if h not in seen_hashes:
                reduced_data.append(item)
                seen_hashes.add(h)
    else:
        # Remove items in data that exist in compare_data
        if hashed_compare is None:
            hashed_compare = hash_sentences(compare_data)
        hashed_compare_set = set(hashed_compare)
        reduced_data = [item for h, item in tqdm(zip(hashed_data, data), total=datalen, desc="Filtering compare_data") if h not in hashed_compare_set]
    removed_count = datalen - len(reduced_data)
    logging.info(f"- Removed: {removed_count}/{datalen} ({round(100*removed_count/datalen, 2)}%)")
    return reduced_data

def manifest_time_stats(
    manifest: Union[str, List[ItemManifest]],
    return_stats: bool = False,
    verbose: bool = True
    ) -> ExcelRecord:
    """
    Compute duration statistics from a manifest and optionally print them.

    Parameters
    ----------
    manifest : str or list
        - If a string: treated as a filepath to a manifest JSON/JSONL file
          readable by `read_manifest()`.  
        - If a list: assumed to be an already-loaded list of dicts where each
          item contains a `"duration"` field.

    return_stats : bool, optional (default=False)
        If True, return a dictionary containing all computed statistics.

    verbose : bool, optional (default=True)
        If True, log the statistics to the console using `logging.info()`.

    Returns
    -------
    dict (optional)
        Returned only if `return_stats=True`. Contains:
            - "filename": source of the manifest  
            - "t_min": minimum segment duration (s)  
            - "t_mean": mean segment duration (s)  
            - "t_median": median segment duration (s)  
            - "t_max": maximum segment duration (s)  
            - "t_total": [sum of durations in seconds, hours]  
            - "t_total_median": [median * count in seconds, hours]  
            - "sentences": number of entries in the manifest

    Notes
    -----
    The function expects each entry in the manifest to contain a
    `"duration"` key whose value can be converted to a float.
    """
    if isinstance(manifest, str):
        data = read_manifest(manifest)
        filename = os.path.split(manifest)[1]
    elif isinstance(manifest, list):
        data = manifest
        filename = "in-memory data"
    else:
        raise Exception(f"ERROR: 'manifest' must be 'str' or 'list'")
    duration = [float(item['duration']) for item in data]
    record: ExcelRecord = {
        "filename": filename,
        "t_min": round(min(duration),2),
        "t_mean": round(statistics.mean(duration),2),
        "t_max": round(max(duration),2),
        "t_total": [round(sum(duration),2), round(sum(duration)/3600,2)],
        "t_median": round(statistics.median(duration),2),
        "t_total_median": [round(statistics.median(duration)*len(duration),2), round(statistics.median(duration)*len(duration)/3600,2)],
        "sentences": len(data)
    }
    if verbose:
        logging.info(f"=============[ {record['filename']} ]=============")
        logging.info(f"- Min time: {record['t_min']} s")
        logging.info(f"- Mean time: {record['t_mean']} s")
        logging.info(f"- Max time: {record['t_max']} s")
        logging.info(f"{'-'*(30+len(record['filename']))}")
        logging.info(f"- Total time (sum): {record['t_total'][0]} s | {record['t_total'][1]} h")
        logging.info(f"- Total sentences: {record['sentences']}")
        logging.info(f"{'-'*(30+len(record['filename']))}")
        logging.info(f"- Median time: {record['t_median']} s")
        logging.info(f"- Total time (median): {record['t_total_median'][0]} s | {record['t_total_median'][1]} h")
        logging.info(f"{'='*(30+len(record['filename']))}")
    if return_stats:
        return record
    
def calculate_wer(
    manifest: Union[str, List[ItemManifest]],
    lang: str="es",
    text_tag: str="text",
    cp_text_tag: str="cp_text",
    pred_text_tag: str="pred_text",
    cp_pred_text_tag: str="cp_pred_text",
    return_wer: bool=False,
    verbose: bool=True
    ) -> Union[None, Tuple[List[ItemManifest], ExcelRecord]]:
    """
    Calculate sentence-level and corpus-level Word Error Rate (WER) from a manifest file.

    Parameters
    ----------
    manifest : str or list
        - If a string: treated as a filepath to a manifest JSON/JSONL file
          readable by `read_manifest()`.  
        - If a list: assumed to be an already-loaded list of dicts where each
          item contains a `"text"` and `"pred_text"` fields.

    lang : str, optional (default="es")
        Language code used for text normalization.

    text_tag : str, optional (default="text")
        Key in the manifest representing the reference text.

    cp_text_tag : str, optional (default="cp_text")
        Key representing case-preserved reference text. Used only when `cp_field=True`.

    pred_text_tag : str, optional (default="pred_text")
        Key representing the predicted text.

    cp_pred_text_tag : str, optional (default="cp_pred_text")
        Key representing case-preserved predicted text. Used only when `cp_field=True`.

    return_wer : bool, optional (default=False)
        If True, returns both the cleaned manifest entries and the result dictionary.

    verbose : bool, optional (default=True)
        If True, logs mean and total WER values to the console.

    Returns
    -------
    tuple (list of dict, dict), optional
        Returned only if `return_wer=True`.

        - data_clean : list of dict  
            Manifest entries after text normalization and per-sentence WER annotation.  
            Keys added:
                * "wer"  
                * "wer_cp"

        - result : dict  
            Summary WER statistics containing:
                * "filename"  
                * "mean_wer"  
                * "total_wer"  
                * "mean_wer_cp"
                * "total_wer_cp"

    Notes
    -----
    - Text normalization is performed using `TextNormalizer`, once for reference text and once for predictions.
    - Per-sentence WER is computed using `jiwer.wer()`.
    - Corpus-level WER is computed by concatenating all texts and evaluating on the full strings.
    - Output values are raw WER scores (0.0–1.0), not percentages.
    - Log output shows percentages for readability.
    """
    filename = (os.path.split(manifest)[1]).replace(".json","")

    data = read_manifest(manifest, verbose=False)
    data_clean = [dict(item) for item in data] 

    # Create the normalizers for each field
    normalizer = TextNormalizer(lang=lang, tag=text_tag, verbose=False)
    pred_normalizer = TextNormalizer(lang=lang, tag=pred_text_tag, verbose=False)
    data_clean = normalizer(data_clean)
    data_clean = pred_normalizer(data_clean)
    
    if "cp_text" in data_clean[0] and "cp_pred_text" in data_clean[0]:
        cp_field = True
    else:
        cp_field = False
    
    if cp_field:
        cp_normalizer = TextNormalizer(lang=lang, tag=cp_text_tag, keep_cp=True, verbose=False)
        cp_pred_normalizer = TextNormalizer(lang=lang, tag=cp_pred_text_tag, keep_cp=True, verbose=False)
        data_clean = cp_normalizer(data_clean)
        data_clean = cp_pred_normalizer(data_clean)

    wer_list = []
    wer_cp_list = []
    total_text = ""
    total_cp_text = ""
    total_pred_text = ""
    total_cp_pred_text = ""
    for item in data_clean:
        # Calculate normalized wer for each sentence
        wer = jiwer.wer(item["text"], item["pred_text"])
        wer_list.append(wer)
        item['wer'] = wer
        total_text += " " + item["text"]
        total_pred_text += " " + item["pred_text"]

        if cp_field:
            # Calculate wer with C&P for each sentence
            wer_cp = jiwer.wer(item["cp_text"], item["cp_pred_text"])
            wer_cp_list.append(wer_cp)
            item['wer_cp'] = wer_cp
            total_cp_text += " " + item["cp_text"]
            total_cp_pred_text += " " + item["cp_pred_text"]

    total_wer = jiwer.wer(total_text.strip(), total_pred_text.strip())   
    mean_wer = sum(wer_list)/len(wer_list)
    if cp_field:
        total_wer_cp = jiwer.wer(total_cp_text.strip(), total_cp_pred_text.strip())
        mean_wer_cp = sum(wer_cp_list)/len(wer_cp_list)
    else:
        total_wer_cp = None
        mean_wer_cp = None

    record: ExcelRecord = {
        "filename": filename,
        "mean_wer": mean_wer,
        "total_wer": total_wer,
        "mean_wer_cp": mean_wer_cp,
        "total_wer_cp": total_wer_cp,
        }
    
    if verbose:
        logging.info(f"=============[ {record['filename']} ]=============")
        logging.info(f"- Mean WER: {round(record['mean_wer']*100,2)} %")
        logging.info(f"- Total WER: {round(record['total_wer']*100,2)} %")
        if cp_field:
            logging.info(f"{'-'*(30+len(record['filename']))}") 
            logging.info(f"- Mean WER C&P: {round(record['mean_wer_cp']*100,2)} %")
            logging.info(f"- Total WER C&P: {round(record['total_wer_cp']*100,2)} %")
        logging.info(f"{'='*(30+len(record['filename']))}")    
    if return_wer: 
        return data_clean, record

def write_excel_record(
    excel_record: Union[List[ExcelRecord], ExcelRecord],
    dst_filepath: str,
    title: Optional[str] = None   
    ):
    """
    Export a list of WER (Word Error Rate) results to an Excel file.

    Parameters
    ----------
    excel_record : list of dict or just a dict
        Should be in teh format ExcelRecord
    
    dst_filepath : str
        Path where the Excel (.xlsx) file will be saved.

    Returns
    -------
    None

    Notes
    -----
    - The Excel file will have a single sheet named as `"title"`.
    - Existing files at the destination path will be overwritten.
    """
    if isinstance(excel_record, dict):
        excel_record = [excel_record]
    
    headers = list(excel_record[0].keys())
    if not title:
        if 'mean_wer' in headers:
            title = "WER Results"
        elif 't_mean' in headers:
            title = "Time Stats"
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title
    ws.append(headers)
    for record in excel_record:
        row = []
        for header in headers:
            element = record[header]
            if isinstance(element, list):
                element = element[1]
            row.append(element)
        ws.append(row)
    wb.save(dst_filepath)