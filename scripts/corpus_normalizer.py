import re
import logging
from tqdm import tqdm
from typing import Union, Optional, List
from corpus_types import ItemManifest

class TextNormalizer:
    def __init__(
        self,
        lang: str,
        tag: str = "text",
        keep_cp: bool = False, 
        remove_acronyms: bool = False,
        remove_emptytext: bool = True,
        blacklist_terms: Optional[List[str]] = None, 
        min_duration: float = 0.025,
        max_duration: float = 240,
        verbose: Union[bool,str] = True
        ):
        """
        Initializes the sentence cleaner with the necessary parameters.
        :param lang: Language ('es' or 'eu') if Bilingual 'es+eu' is wanted just select 'es'.
        :param tag: Key field for the text in the data.
        :param keep_cp: Whether to preserve Capitalization and Punctuation.
        :param remove_acronyms: Whether to remove entries with acronyms
        :param remove_emptytext: Whether to remove emptytext entries
        :param blacklist_terms: List of terms to remove (if provided).
        :param min/max_duration: duration threshold in seconds for the audios, will remove the sentence if it's out of bounds.
        :param verbose: 'False= Not show anything, 'True'= Show basic logging info, 'all'= Show detailed logging info.
        """
        self.lang = lang.lower()
        if self.lang not in ['es', 'eu']:
            raise ValueError(f"ERROR: Language '{lang}' NOT Supported.\n Supported languages:\n\t- Spanish: 'es'.\n\t- Basque: 'eu'")
        self.tag = tag
        self.keep_cp = keep_cp
        self.remove_acronyms = remove_acronyms
        self.remove_emptytext = remove_emptytext
        self.blacklist_terms = blacklist_terms
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.verbose = verbose

        self.unclean_char_list = set()
        self.clean_char_list = set()
        self.diacritic_map = {
            r"[Æ]": "Ae", r"[Œ]": "Oe", r"[Ж]": "Zh", r"[Х]": "H", r"[Щ]": "Shch", r"[Ш]": "Sh", r"[Ф]": "F",
            r"[Ч]": "Ch", r"[Ц]": "Ts", r"[Þ]": "Th", r"[Α]": "A", r"[Β]": "V", r"[Γ]": "G", r"[Δ]": "D",
            r"[Ζ]": "Z", r"[Η]": "I", r"[Θ]": "Th", r"[Κ]": "K", r"[Λ]": "L", r"[Μ]": "M", r"[Ν]": "N",
            r"[Ξ]": "Ks", r"[Π]": "P", r"[Ρ]": "R", r"[Σ]": "S", r"[Τ]": "T", r"[Υ]": "I", r"[Φ]": "F",
            r"[Χ]": "J", r"[Ψ]": "Ps", r"[Ω]": "O", r"[ß]": "ss", r"[ж]": "zh", r"[х]": "h", r"[щ]": "shch",
            r"[ш]": "sh", r"[ф]": "f", r"[ч]": "ch", r"[ц]": "ts", r"[ð]": "d", r"[ђ]": "dj", r"[α]": "a",
            r"[β]": "v", r"[γ]": "g", r"[δ]": "d", r"[ζ]": "z", r"[η]": "i", r"[θ]": "th", r"[κ]": "k",
            r"[λ]": "l", r"[μ]": "m", r"[ν]": "n", r"[ξ]": "ks", r"[π]": "p", r"[ρ]": "r", r"[σς]": "s",
            r"[υ]": "u", r"[φ]": "f", r"[χ]": "j", r"[ψ]": "ps", r"[ÈËÊЕЭ]": "E", r"[АÃÂÀÄÅ]": "A",
            r"[ÙÛŪ]": "U", r"[ÔÖÒÕØΟ]": "O", r"[ÇĆČ]": "C", r"[ÏÌÎĪ]": "I", r"[ÑŃǸ]": "Ñ", r"[ÝŶŸ]": "Y",
            r"[èëēêе]": "e", r"[аãâāàä]": "a", r"[ùūû]": "u", r"[ôōòöõ]": "o", r"[ćç]": "c", r"[ïīìî]": "i",
            r"[ż]": "z", r"[ ]": " "
        }

    def replace_diacritics(self, item: ItemManifest) -> ItemManifest:
        """Replaces diacritic characters with their normalized versions."""
        for pattern, replacement in self.diacritic_map.items():
            item[self.tag] = re.sub(pattern, replacement, item[self.tag])
        if self.lang == "eu":
            eu_specific = {
                r"[É]": "E", r"[Á]": "A", r"[ÚÜ]": "U", r"[Ó]": "O", r"[Í]": "I",
                r"[é]": "e", r"[á]": "a", r"[úü]": "u", r"[ó]": "o", r"[í]": "i"
            }
            for pattern, replacement in eu_specific.items():
                item[self.tag] = re.sub(pattern, replacement, item[self.tag])
        return item

    def remove_special_chars_whitelist(self, item: ItemManifest) -> ItemManifest:
        """Removes not allowed special characters."""
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúüÁÉÍÓÚÜñÑ "
        if self.keep_cp:
            allowed_chars += ".,¿?¡!;:"
        allowed_chars_pattern = f"[^{allowed_chars}]"
        if self.blacklist_terms:
            for term in self.blacklist_terms:
                item[self.tag] = re.sub(term, "", item[self.tag], flags=re.IGNORECASE)
        item[self.tag] = re.sub(allowed_chars_pattern, " ", item[self.tag])
        item[self.tag] = re.sub(r" +", " ", item[self.tag]).strip()
        if not self.keep_cp:
            item[self.tag] = item[self.tag].lower()
        return item
    
    def in_duration_threshold(self, item: ItemManifest) -> ItemManifest:
        """Returns True if duration is within min/max threshold or missing; False if duration exists and is out of bounds."""
        duration = item.get("duration")
        if duration is None:
            return True
        if not (self.min_duration <= duration <= self.max_duration):
            return False
        return True

    def clean_sentences(self, data: List[ItemManifest]) -> List[ItemManifest]:
        clean_data = []
        emptytext_entries = []
        acronyms_entries = []
        outbound_entries = []
        for item in tqdm(data, disable=not self.verbose):
            if self.in_duration_threshold(item):
                acronyms = bool(re.search(r'\b[\w\d]*[A-Z]{2,}[\w\d]*\b', item[self.tag])) if self.remove_acronyms else False
                if not acronyms:
                    self.unclean_char_list.update(set(item[self.tag]))
                    item = self.replace_diacritics(item)
                    item = self.remove_special_chars_whitelist(item)
                    if self.remove_emptytext and not re.search(r"[A-Za-z]",item["text"]):
                        emptytext_entries.append(item)
                    else:
                        clean_data.append(item)
                    self.clean_char_list.update(set(item[self.tag]))
                else:
                    acronyms_entries.append(item)
            else: 
                outbound_entries.append(item)
                
        if self.verbose:
            l = len(outbound_entries)
            m = len(emptytext_entries)
            n = len(acronyms_entries)
            total = len(data)
            
            logging.info(f"::::: Character List :::::")
            logging.info(f"- Before cleaning (size: {len(self.unclean_char_list)})\n  {sorted(self.unclean_char_list)}")
            logging.info(f"- After cleaning (size: {len(self.clean_char_list)})\n  {sorted(self.clean_char_list)}")
            
            logging.info(f"\n::::: Removed sentences :::::")
            logging.info(f"- Total: {l+m+n}/{total} ({round(100*(l+m+n) / total, 2)}%)")
            
            logging.info(f"- Entries out of bounds:")
            logging.info(f"  · {l}/{total} ({round(100*l / total, 2)}%)")
            if self.verbose == 'all':
                for entry in outbound_entries:
                    logging.info(f"    audio: {entry['audio_filepath']}")
                    logging.info(f" duration: {entry['duration']}")
                    
            logging.info(f"- Entries without Text:")
            logging.info(f"  · {m}/{total} ({round(100*m / total, 2)}%)")
            if self.verbose == 'all':
                for entry in emptytext_entries:
                    logging.info(f"    audio: {entry['audio_filepath']}")     
                    
            logging.info(f"- Entries with Acronyms:")
            logging.info(f"  · {n}/{total} ({round(100*n / total, 2)}%)")
            if self.verbose == 'all':
                for entry in acronyms_entries:
                    logging.info(f"    audio: {entry['audio_filepath']}")
                    logging.info(f"     text: {entry[self.tag]}")              
            
        return clean_data
    
    def __call__(self, data: List[ItemManifest]):
        return self.clean_sentences(data)
