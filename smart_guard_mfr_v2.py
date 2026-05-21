"""MFR(Most Frequent Replacement) 정규화 모듈 — 최종 파이프라인의 MFR 단계.

predict_smart_guarded_mfr_v2(samples, mfr_stats): protected 토큰을 보호하면서
mfr_stats 기반 token 단위 최빈 정규화.

protect 헬퍼 2종:
  find_protected_indices        — 강한 protect (v2 통합 common_prompt). MFR / hard-case gating용.
  find_protected_indices_simple — 약한 protect (@/#/URL/순수기호/숫자). trigram override 전용.
"""

import re
import sys
from pathlib import Path


def find_protected_indices_simple(tokens):
    """약한 protect — @/#/URL/순수기호/숫자포함 토큰만 보호. trigram override 전용.

    trigram은 강한 protect를 쓰면 성능이 떨어져 의도적으로 이 약한 버전을 쓴다.
    """
    protected = []
    for i, tok in enumerate(tokens):
        tok = str(tok)
        if tok.startswith("@") or tok.startswith("#"):
            protected.append(i)
        elif "http" in tok.lower() or "www." in tok.lower():
            protected.append(i)
        elif re.fullmatch(r"[\W_]+", tok):
            protected.append(i)
        elif re.search(r"\d", tok):
            protected.append(i)
    return protected


# 강한 protect — v2 통합 common_prompt의 find_protected_indices.
# MFR(predict_smart_guarded_mfr_v2)과 hard-case gating(mine/build)이 사용한다.
# adapter 경유 시엔 sys.modules["prompts.common_prompt"]가 이미 v2로 등록돼 있고,
# mine/build처럼 단독 실행 시엔 common_prompt_v2_package에서 절대경로로 로드한다.
try:
    from prompts.common_prompt import find_protected_indices
except Exception:
    _CP_DIR = Path(__file__).parent / "prompt_mfr_dictionary" / "common_prompt_v2_package" / "prompts"
    if str(_CP_DIR) not in sys.path:
        sys.path.insert(0, str(_CP_DIR))
    from common_prompt import find_protected_indices


def normalize_lang_code(lang):
    lang = str(lang).lower()
    if lang in {"id-en", "id_en", "iden"}:
        return "iden"
    if lang in {"tr-de", "tr_de", "trde"}:
        return "trde"
    return lang


def get_lang(sample):
    return str(sample.get("lang", sample.get("language", ""))).lower()


def is_japanese_punctuation_like(tok: str) -> bool:
    tok = str(tok)
    ja_punct_set = {
        "...", "…", "〜", "～",
        "。", "、",
        "！", "？", "!", "?",
        "・", "･",
        "ー",
        "（", "）", "(", ")",
        "「", "」", "『", "』",
    }
    return tok in ja_punct_set


def is_indonesian_reduplication_2(tok: str) -> bool:
    tok = str(tok)
    return bool(re.search(r"[A-Za-zÀ-ÿ]+2(?:[A-Za-zÀ-ÿ]+)?$", tok))


def is_south_slavic_digit_suffix(tok: str) -> bool:
    tok = str(tok)
    return bool(re.search(r"\d", tok) and re.search(r"[A-Za-zÀ-ÿ]", tok))


def is_time_like_normalization(tok: str, top_norm: str) -> bool:
    tok = str(tok)
    top_norm = str(top_norm)
    if not re.fullmatch(r"\d{3,4}", tok):
        return False
    return bool(re.search(r"[.:]", top_norm))


def is_mention_normalization(tok: str, top_norm: str) -> bool:
    tok = str(tok)
    top_norm = str(top_norm)
    return tok.startswith("@") and top_norm == "[mention]"


def should_allow_mfr_for_protected_v2(
    lang,
    tok,
    info,
    protected_threshold=0.8,
    *,
    japanese_punct_threshold=0.95,
    digit_suffix_threshold=0.5,
    time_threshold=0.5,
):
    lang = normalize_lang_code(lang)
    tok = str(tok)

    if info is None:
        return False

    top_norm = str(info["top_norm"])
    conf = float(info["confidence"])

    # 1. mention normalization 허용
    if is_mention_normalization(tok, top_norm):
        return True

    # 2. Indonesian / ID-EN word+2 reduplication 허용
    if lang in {"id", "iden"} and is_indonesian_reduplication_2(tok):
        return True

    # 3. South Slavic digit suffix 허용
    if lang in {"hr", "sl", "sr"} and is_south_slavic_digit_suffix(tok):
        return conf >= digit_suffix_threshold

    # 4. 시간 표현 허용
    if is_time_like_normalization(tok, top_norm):
        return conf >= time_threshold

    # 5. 일본어 punctuation은 보수적으로 raw 유지
    if lang == "ja" and is_japanese_punctuation_like(tok):
        return conf >= japanese_punct_threshold

    # 6. 기본 protected threshold
    return conf >= protected_threshold


def predict_smart_guarded_mfr_v2(
    samples,
    mfr_stats,
    *,
    protected_threshold=0.8,
    normal_threshold=0.0,
    japanese_punct_threshold=0.95,
    digit_suffix_threshold=0.5,
    time_threshold=0.5,
):
    preds = []

    for sample in samples:
        lang = normalize_lang_code(get_lang(sample))
        raw_tokens = [str(tok) for tok in sample["raw"]]

        protected = set(find_protected_indices(raw_tokens))
        sent_pred = []

        for i, tok in enumerate(raw_tokens):
            info = mfr_stats.get(lang, {}).get(tok)

            if info is None:
                sent_pred.append(tok)
                continue

            if i in protected:
                if should_allow_mfr_for_protected_v2(
                    lang,
                    tok,
                    info,
                    protected_threshold=protected_threshold,
                    japanese_punct_threshold=japanese_punct_threshold,
                    digit_suffix_threshold=digit_suffix_threshold,
                    time_threshold=time_threshold,
                ):
                    sent_pred.append(str(info["top_norm"]))
                else:
                    sent_pred.append(tok)
            else:
                if float(info["confidence"]) >= normal_threshold:
                    sent_pred.append(str(info["top_norm"]))
                else:
                    sent_pred.append(tok)

        preds.append(sent_pred)

    return preds
