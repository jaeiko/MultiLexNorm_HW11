#!/usr/bin/env python3
"""
LLM correction with the OLD Stage-3 prompt builder.

- Reuses prompt_mfr_adapter.PromptMFRResources.build_normalization_prompt()
- Single-target per call (one hard token per API call)
- Output schema: {"index": i, "raw": "...", "normalized": "..."}
- Compatible parser: prompt_mfr_dictionary.common_prompt.extract_first_json_object
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict
import concurrent.futures

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_HERE = Path(__file__).parent
_MFR = _HERE.parent / "mfr_first_package"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_MFR))

from prompt_mfr_adapter import PromptMFRResources  # noqa
# Need the JSON parser from the common package; import via prompt_mfr_dictionary directly
sys.path.insert(0, str(_MFR / "prompt_mfr_dictionary" / "common_prompt_v2_package" / "prompts"))
from common_prompt import extract_first_json_object  # noqa


def load_env(env_path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ[k] = v


def call_gpt(client, model, system, user, max_retries=3, reasoning_effort=None):
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'response_format': {'type': 'json_object'},
            }
            if reasoning_effort:
                kwargs['reasoning_effort'] = reasoning_effort
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise last_err


def normalize_lang_for_resources(lang):
    """The old code expected: 'da','de','en','es','hr','id','iden','it','ja','ko','nl','sl','sr','th','tr','trde','vi'.
    We pass it through.
    """
    return lang


def process_one(args):
    """Top-level helper for concurrent.futures (must be picklable when used with processes;
    we'll use threads since OpenAI calls are I/O-bound)."""
    client, model, system, user, meta, reasoning_effort = args
    try:
        content = call_gpt(client, model, system, user, reasoning_effort=reasoning_effort)
    except Exception as e:
        return meta, None, str(e)
    return meta, content, None


def main():
    print("[LLM correction OLD-PROMPT] GPT-5-mini, single-target, no dynamic fewshot")
    t0 = time.time()
    base = _HERE
    repo = base.parent

    load_env(base / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY missing"); return 1
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    reasoning_effort = os.environ.get("REASONING_EFFORT")  # None | 'minimal' | 'low' | 'medium' | 'high'
    out_filename = os.environ.get("OUTPUT_FILE", "llm_corrections_oldprompt.jsonl")
    print(f"  model: {model_name}  reasoning_effort: {reasoning_effort or 'default'}  out: {out_filename}")

    # Prompt resources (old MFR adapter)
    # Must set cwd so PromptMFRResources finds the dictionary by relative path.
    old_cwd = os.getcwd()
    os.chdir(_MFR)
    try:
        resources = PromptMFRResources("prompt_mfr_dictionary")
        # Optional dynamic fewshot (token-level: positive=changed examples only, negative=0)
        fewshot_obj = None
        fewshot_k = int(os.environ.get("FEWSHOT_K", "0"))
        if fewshot_k > 0:
            from normalization_fewshot import NormalizationFewshot
            fewshot_obj = NormalizationFewshot(
                'data/train_internal_v1.parquet',
                default_positive_k=fewshot_k,
                default_negative_k=0,
            )
            print(f"  fewshot loaded (k={fewshot_k}, changed examples only)")
    finally:
        os.chdir(old_cwd)
    print(f"  prompt resources loaded")

    # Hard cases (overridable via HARD_CASES_FILE env)
    hc_filename = os.environ.get("HARD_CASES_FILE", "hard_cases_mini.jsonl")
    hc_path = base / "outputs" / hc_filename
    print(f"  hard cases input: {hc_filename}")
    hard = [json.loads(l) for l in hc_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"  hard cases: {len(hard)}")

    # Resume
    out_path = base / "outputs" / out_filename
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text(encoding='utf-8').splitlines():
            try:
                obj = json.loads(line)
                done_keys.add((obj['mini_row_idx'], obj['tok_idx']))
            except Exception:
                pass
        print(f"  resume: {len(done_keys)} already done")

    todo = [r for r in hard if (r['mini_row_idx'], r['tok_idx']) not in done_keys]
    print(f"  to process: {len(todo)}")

    SYSTEM = "You are a careful lexical normalization model. Output exactly one JSON object as instructed."

    # Build prompts upfront (so we know lengths)
    prompts = []
    skipped = 0
    for r in todo:
        try:
            fs_pairs = None
            if fewshot_obj is not None:
                fs_pairs = fewshot_obj.retrieve(
                    target_token=r['token'], lang=r['lang'],
                    full_sentence_tokens=r['raw_sentence']
                )
            user = resources.build_normalization_prompt(
                tokens=r['raw_sentence'],
                target_index=r['tok_idx'],
                lang=normalize_lang_for_resources(r['lang']),
                dynamic_fewshot_pairs=fs_pairs,
            )
        except Exception as e:
            print(f"  build error mini_row={r['mini_row_idx']} idx={r['tok_idx']} lang={r['lang']}: {e}")
            skipped += 1
            continue
        prompts.append((r, user))

    if skipped:
        print(f"  skipped {skipped} (prompt build failure)")

    if prompts:
        sample_len = len(prompts[0][1])
        print(f"  sample prompt length: {sample_len} chars (~{sample_len//4} tokens estimate)")

    # Concurrent call (threads, 8 workers — IO bound)
    n_done = 0
    n_err = 0
    n_parse_fail = 0
    t_loop = time.time()
    with open(out_path, "a", encoding='utf-8') as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = []
        for r, user in prompts:
            meta = {
                'mini_row_idx': r['mini_row_idx'],
                'tok_idx': r['tok_idx'],
                'lang': r['lang'],
                'token': r['token'],
                'gt': r['gt'],
                'cat': r['cat'],
            }
            futures.append(ex.submit(process_one, (client, model_name, SYSTEM, user, meta, reasoning_effort)))

        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            meta, content, err = fut.result()
            if err:
                n_err += 1
                fout.write(json.dumps({**meta, 'raw_response': None, 'error': err}, ensure_ascii=False) + "\n")
                fout.flush()
                continue
            obj = extract_first_json_object(content)
            norm = None
            if obj is not None:
                norm = obj.get('normalized')
                if not isinstance(norm, str):
                    norm = None
            if norm is None:
                n_parse_fail += 1
            n_done += 1
            fout.write(json.dumps({**meta, 'raw_response': content, 'normalized': norm}, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t_loop
                eta = el / (i + 1) * (len(futures) - i - 1)
                print(f"  [{i+1}/{len(futures)}] elapsed={el:.1f}s ETA={eta:.1f}s  done={n_done} err={n_err} parse_fail={n_parse_fail}")

    print(f"\n[done] done={n_done} err={n_err} parse_fail={n_parse_fail}")
    print(f"  saved → {out_path}")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
