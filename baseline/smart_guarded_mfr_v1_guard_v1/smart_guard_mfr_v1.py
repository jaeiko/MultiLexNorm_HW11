import math

GUARD_V1_CONFIG = {
    "default": {"mode": "base"},
    "by_lang": {
        "vi":   {"mode": "margin_guard", "margin": 0.20},
        "iden": {"mode": "confidence_guard", "confidence": 0.80},
        "en":   {"mode": "margin_guard", "margin": 0.20},
        "id":   {"mode": "change_guard", "change_rate": 0.75},
        "sl":   {"mode": "margin_guard", "margin": 0.20},
        "sr":   {"mode": "change_guard", "change_rate": 0.75},
        "th":   {"mode": "margin_guard", "margin": 0.20},
        "hr":   {"mode": "base"},
        "nl":   {"mode": "change_guard", "change_rate": 0.75},
        "de":   {"mode": "change_guard", "change_rate": 0.75},
        "ja":   {"mode": "margin_guard", "margin": 0.20},
        "ko":   {"mode": "change_guard", "change_rate": 0.75},
    },
}


def normalize_lang_code(lang):
    lang = str(lang).lower()
    if lang in {"id-en", "id_en", "iden"}:
        return "iden"
    if lang in {"tr-de", "tr_de", "trde"}:
        return "trde"
    return lang


def get_lang(sample):
    return normalize_lang_code(sample.get("lang", sample.get("language", "")))


def should_apply_mfr_v1(info, config):
    if info is None:
        return False

    mode = config.get("mode", "base")

    if mode == "base":
        return True

    min_count = config.get("min_count", 1)
    if info.get("total_count", 0) < min_count:
        return False

    if mode == "change_guard":
        return info.get("change_rate", 0.0) >= config.get("change_rate", 0.75)

    if mode == "confidence_guard":
        return info.get("confidence", 0.0) >= config.get("confidence", 0.80)

    if mode == "margin_guard":
        return info.get("margin", 0.0) >= config.get("margin", 0.20)

    if mode == "entropy_guard":
        return info.get("entropy", float("inf")) <= config.get("entropy", 0.50)

    if mode == "combined_guard":
        return (
            info.get("confidence", 0.0) >= config.get("confidence", 0.0)
            and info.get("change_rate", 0.0) >= config.get("change_rate", 0.0)
            and info.get("margin", 0.0) >= config.get("margin", 0.0)
            and info.get("entropy", float("inf")) <= config.get("entropy", float("inf"))
        )

    raise ValueError(f"Unknown guard mode: {mode}")


def predict_smart_guarded_mfr_v1(samples, mfr, stats, guard_config=None):
    """Predict normalized token lists using Smart Guard v1.

    Parameters
    ----------
    samples:
        list of dicts with at least {'raw': list[str], 'lang': str}
    mfr:
        dict keyed by (lang, raw_token), value = top normalized token
    stats:
        dict keyed by (lang, raw_token), value containing confidence/change_rate/margin/entropy
    guard_config:
        optional config with {'default': ..., 'by_lang': ...}

    Returns
    -------
    list[list[str]]
    """
    if guard_config is None:
        guard_config = GUARD_V1_CONFIG

    default_config = guard_config.get("default", {"mode": "base"})
    by_lang = guard_config.get("by_lang", {})

    predictions = []

    for sample in samples:
        lang = get_lang(sample)
        config = by_lang.get(lang, default_config)
        pred_sent = []

        for tok in sample["raw"]:
            key = (lang, tok)
            info = stats.get(key)
            if info is not None and should_apply_mfr_v1(info, config):
                pred_sent.append(mfr[key])
            else:
                pred_sent.append(tok)

        predictions.append(pred_sent)

    return predictions
