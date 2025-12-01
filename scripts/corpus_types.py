from typing import TypedDict, Optional, List

class ItemManifest(TypedDict, total=False):
    audio_filepath: str
    text: str
    cp_text: Optional[str]
    duration: Optional[float]
    speaker: Optional[str]
    language: Optional[str]
    pred_text: Optional[str]
    cp_pred_text: Optional[str]
    wer: Optional[float]
    wer_cp: Optional[float]
    
class ExcelRecord(TypedDict):
    filename: str
    t_min: Optional[float] # seconds
    t_mean: Optional[float] # seconds
    t_max: Optional[float] # seconds
    t_total: Optional[List[float]]  # [seconds, hours]
    t_median: Optional[float] # seconds
    t_total_median: Optional[List[float]]  # [seconds, hours]
    sentences: Optional[int]
    mean_wer: Optional[float]
    total_wer: Optional[float]
    mean_wer_cp: Optional[float]
    total_wer_cp: Optional[float]   