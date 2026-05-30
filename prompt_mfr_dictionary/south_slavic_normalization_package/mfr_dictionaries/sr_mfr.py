"""SR MFR wrapper."""
from __future__ import annotations
from pathlib import Path
try:
    from mfr_dictionaries.south_slavic_mfr import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    from south_slavic_mfr import *  # type: ignore # noqa: F401,F403

DEFAULT_DICTIONARY_PATH = Path(__file__).with_name("sr_mfr_dictionary.json")

def load_sr_mfr_dictionary(path: str | Path = DEFAULT_DICTIONARY_PATH) -> dict:
    return load_mfr_dictionary(path)  # type: ignore[name-defined]

def apply_sr_mfr_to_tokens(tokens, dictionary: dict | None = None, **kwargs):
    d = load_sr_mfr_dictionary() if dictionary is None else dictionary
    return apply_mfr_to_tokens(tokens, d, **kwargs)  # type: ignore[name-defined]
