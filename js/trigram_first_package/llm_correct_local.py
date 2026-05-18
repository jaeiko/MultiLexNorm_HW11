#!/usr/bin/env python3
"""
LLM correction via local ollama (OpenAI-compatible endpoint).
Reuses the OLD Stage-3 prompt builder.

Usage:
  python llm_correct_local.py --model gemma4:4b --output llm_corrections_gemma4_4b.jsonl
  python llm_correct_local.py --model qwen2.5:7b-instruct --output llm_corrections_qwen25_7b.jsonl
  python llm_correct_local.py --model gemma4:26b --output llm_corrections_gemma4_26b.jsonl
"""

import argparse
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
sys.path.insert(0, str(_MFR / "prompt_mfr_dictionary" / "common_prompt_v2_package" / "prompts"))
from common_prompt import extract_first_json_object  # noqa


def call_ollama(client, model, system, user, max_retries=3, use_json_format=True, disable_thinking=True):
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
            }
            if use_json_format:
                kwargs['response_format'] = {'type': 'json_object'}
            if disable_thinking:
                # ollama-specific: turn off Gemma/Qwen3 thinking mode
                kwargs['extra_body'] = {'think': False}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise last_err


def process_one(args):
    client, model, system, user, meta, use_json = args
    try:
        content = call_ollama(client, model, system, user, use_json_format=use_json)
    except Exception as e:
        return meta, None, str(e)
    return meta, content, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='ollama model name')
    ap.add_argument('--output', required=True, help='output jsonl filename (under outputs/)')
    ap.add_argument('--base-url', default='http://localhost:11434/v1')
    ap.add_argument('--workers', type=int, default=2, help='concurrent request workers')
    ap.add_argument('--no-json-format', action='store_true', help='omit response_format json_object (if model rejects)')
    ap.add_argument('--fewshot', action='store_true', help='enable dynamic retrieve-based few-shot pairs')
    ap.add_argument('--pos-k', type=int, default=5, help='positive (changed) fewshot k')
    ap.add_argument('--neg-k', type=int, default=5, help='negative (unchanged) fewshot k')
    args = ap.parse_args()

    print(f"[LLM correction LOCAL] model={args.model} workers={args.workers}")
    t0 = time.time()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package missing"); return 1
    client = OpenAI(base_url=args.base_url, api_key='ollama')

    # Prompt resources
    old_cwd = os.getcwd()
    os.chdir(_MFR)
    try:
        resources = PromptMFRResources("prompt_mfr_dictionary")
        fewshot_obj = None
        if args.fewshot:
            from normalization_fewshot import NormalizationFewshot
            fewshot_obj = NormalizationFewshot(
                'data/train_internal_v1.parquet',
                default_positive_k=args.pos_k,
                default_negative_k=args.neg_k,
                lang_overrides={
                    'th': {'positive_k': 2, 'negative_k': 2},
                    'ja': {'positive_k': 3, 'negative_k': 3},
                    'ko': {'positive_k': 3, 'negative_k': 3},
                },
            )
            print(f"  fewshot loaded (pos={args.pos_k} neg={args.neg_k})")
    finally:
        os.chdir(old_cwd)
    print(f"  prompt resources loaded")

    hc_filename = os.environ.get("HARD_CASES_FILE", "hard_cases_mini.jsonl")
    hc_path = _HERE / "outputs" / hc_filename
    print(f"  hard cases input: {hc_filename}")
    hard = [json.loads(l) for l in hc_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f"  hard cases: {len(hard)}")

    out_path = _HERE / "outputs" / args.output
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

    SYSTEM = ("You are a careful lexical normalization model. "
              "Reply DIRECTLY with the requested JSON object — do not output any reasoning, thinking, explanation, or extra text. "
              "Output exactly one JSON object as instructed.")
    USE_JSON = not args.no_json_format

    def build_prompt_for(r):
        fs_pairs = None
        if fewshot_obj is not None:
            fs_pairs = fewshot_obj.retrieve(
                target_token=r['token'], lang=r['lang'],
                full_sentence_tokens=r['raw_sentence']
            )
        return resources.build_normalization_prompt(
            tokens=r['raw_sentence'], target_index=r['tok_idx'], lang=r['lang'],
            dynamic_fewshot_pairs=fs_pairs,
        )

    # Sanity check 1 call before bulk
    if todo:
        r = todo[0]
        user = build_prompt_for(r)
        try:
            content = call_ollama(client, args.model, SYSTEM, user, use_json_format=USE_JSON)
            obj = extract_first_json_object(content)
            print(f"  sanity OK. response head: {(content or '')[:200]!r}")
            print(f"  parsed: {obj}")
        except Exception as e:
            print(f"  sanity FAIL ({type(e).__name__}): {e}")
            if USE_JSON:
                print("  retrying without response_format ...")
                USE_JSON = False
                try:
                    content = call_ollama(client, args.model, SYSTEM, user, use_json_format=False)
                    print(f"  sanity OK (no json fmt). head: {(content or '')[:200]!r}")
                except Exception as e2:
                    print(f"  sanity FAIL again: {e2}")
                    return 1
            else:
                return 1

    n_done = 0
    n_err = 0
    n_parse_fail = 0
    t_loop = time.time()
    with open(out_path, "a", encoding='utf-8') as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for r in todo:
            try:
                user = build_prompt_for(r)
            except Exception as e:
                fout.write(json.dumps({
                    'mini_row_idx': r['mini_row_idx'], 'tok_idx': r['tok_idx'],
                    'lang': r['lang'], 'token': r['token'], 'gt': r['gt'], 'cat': r['cat'],
                    'raw_response': None, 'normalized': None, 'error': f'build_failed: {e}',
                }, ensure_ascii=False) + "\n")
                continue
            meta = {
                'mini_row_idx': r['mini_row_idx'], 'tok_idx': r['tok_idx'],
                'lang': r['lang'], 'token': r['token'], 'gt': r['gt'], 'cat': r['cat'],
            }
            futures.append(ex.submit(process_one, (client, args.model, SYSTEM, user, meta, USE_JSON)))

        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            meta, content, err = fut.result()
            if err:
                n_err += 1
                fout.write(json.dumps({**meta, 'raw_response': None, 'normalized': None, 'error': err}, ensure_ascii=False) + "\n")
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
                print(f"  [{i+1}/{len(futures)}] elapsed={el:.1f}s ETA={eta:.1f}s done={n_done} err={n_err} parse_fail={n_parse_fail}")

    print(f"\n[done] done={n_done} err={n_err} parse_fail={n_parse_fail}")
    print(f"  saved → {out_path}")
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
