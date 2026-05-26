"""LLM Correction Pipeline Stage via local Ollama.

This module acts as Stage 3 of our lexical normalization pipeline.
It extracts target tokens flagged as hard cases, queries a local Ollama
instance (compatible with OpenAI's Chat Completion API), and generates
candidate normalization updates using dynamic few-shot template prompting.
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import time
from pathlib import Path
import concurrent.futures
from typing import Any, Dict, List, Tuple, Set

# Centralized paths configuration
import paths_config
paths_config.setup_imports()

from prompt_mfr_adapter import PromptMFRResources
from common_prompt import extract_first_json_object


def call_ollama(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_retries: int = 3,
    use_json_format: bool = True,
    disable_thinking: bool = True,
) -> str:
    """Queries the local Ollama instance with retry loops and thinking filters.

    Args:
        client: OpenAI API compatible client object.
        model: Name of the Ollama model.
        system: System instructions prompt string.
        user: User message prompt string.
        max_retries: Number of request attempts before raising error.
        use_json_format: Set to True to enforce json response structure.
        disable_thinking: Set to True to filter out model internal thinking tags.

    Returns:
        str: Response message content from the model.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'temperature': 0,
                'seed': 42,
            }
            if use_json_format:
                kwargs['response_format'] = {'type': 'json_object'}
            if disable_thinking:
                kwargs['extra_body'] = {'think': False}
                
            resp = client.chat.completions.create(**kwargs)
            return str(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise last_err


def process_one(args: Tuple[Any, str, str, str, Dict[str, Any], bool]) -> Tuple[Dict[str, Any], str | None, str | None]:
    """Processes a single target token prediction asynchronously.

    Args:
        args: A tuple package containing (client, model, system, user, meta_dict, use_json_bool).

    Returns:
        Tuple[Dict[str, Any], str | None, str | None]: (meta, content, error_msg).
    """
    client, model, system, user, meta, use_json = args
    try:
        content = call_ollama(client, model, system, user, use_json_format=use_json)
        return meta, content, None
    except Exception as e:
        return meta, None, str(e)


def main() -> int:
    """Executes the LLM Correction Stage via command line."""
    ap = argparse.ArgumentParser(description="LLM Correction Stage via local Ollama endpoint.")
    ap.add_argument('--model', required=True, help='Ollama model identifier name')
    ap.add_argument('--output', required=True, help='Output JSONL filename under outputs/')
    ap.add_argument('--base-url', default='http://localhost:11434/v1')
    ap.add_argument('--workers', type=int, default=2, help='Concurrent worker threads count')
    ap.add_argument('--no-json-format', action='store_true', help='Omit json response enforcement')
    ap.add_argument('--fewshot', action='store_true', help='Enable dynamic similarity-based few-shot retrieval')
    ap.add_argument('--pos-k', type=int, default=5, help='Count of positive few-shot items')
    ap.add_argument('--neg-k', type=int, default=5, help='Count of negative few-shot items')
    args = ap.parse_args()

    print(f"[LLM correction LOCAL] model={args.model} workers={args.workers}")
    t0 = time.time()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai library is missing in this environment.")
        return 1
        
    client = OpenAI(base_url=args.base_url, api_key='ollama')

    # Load prompt resources using paths_config
    resources = PromptMFRResources(str(paths_config.PROMPT_MFR_DICT_DIR))
    fewshot_obj = None
    if args.fewshot:
        from normalization_fewshot import NormalizationFewshot
        train_path = paths_config.DATASET_DIR / "train-00000-of-00001.parquet"
        fewshot_obj = NormalizationFewshot(
            train_path,
            default_positive_k=args.pos_k,
            default_negative_k=args.neg_k,
        )
        print(f"  Dynamic few-shot statistics loaded from {train_path.name}")
    print(f"  Prompt resources loaded successfully from {paths_config.PROMPT_MFR_DICT_DIR.name}")

    # Load mined hard cases
    hc_filename = os.environ.get("HARD_CASES_FILE", "hard_cases_val.jsonl")
    hc_path = paths_config.ROOT_DIR / "outputs" / hc_filename
    print(f"  Mined hard cases file: {hc_path.name}")
    if not hc_path.exists():
        print(f"ERROR: Hard cases file does not exist at {hc_path}. Run mine_hard_cases_dev.py first.")
        return 1
        
    hard = [json.loads(line) for line in hc_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    print(f"  Total hard cases to process: {len(hard)}")

    out_path = paths_config.ROOT_DIR / "outputs" / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys: Set[Tuple[int, int]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding='utf-8').splitlines():
            try:
                obj = json.loads(line)
                done_keys.add((int(obj['dev_row_idx']), int(obj['tok_idx'])))
            except Exception:
                pass
        print(f"  Resuming prediction: {len(done_keys)} items already processed.")

    todo = [r for r in hard if (int(r['dev_row_idx']), int(r['tok_idx'])) not in done_keys]
    print(f"  Remaining items to process: {len(todo)}")

    SYSTEM = ("You are a careful lexical normalization model. "
              "Reply DIRECTLY with the requested JSON object — do not output any reasoning, thinking, explanation, or extra text. "
              "Output exactly one JSON object as instructed.")
    USE_JSON = not args.no_json_format

    def build_prompt_for(r: Dict[str, Any]) -> str:
        fs_pairs = None
        if fewshot_obj is not None:
            fs_pairs = fewshot_obj.retrieve(
                target_token=r['token'], lang=r['lang'],
                full_sentence_tokens=r['raw_sentence']
            )
        return str(resources.build_normalization_prompt(
            tokens=r['raw_sentence'], target_index=r['tok_idx'], lang=r['lang'],
            dynamic_fewshot_pairs=fs_pairs,
        ))

    # Performs a quick 1-call sanity check to verify Ollama status and formatting
    if todo:
        r = todo[0]
        user = build_prompt_for(r)
        try:
            content = call_ollama(client, args.model, SYSTEM, user, use_json_format=USE_JSON)
            obj = extract_first_json_object(content)
            print(f"  Sanity check OK. response prefix: {(content or '')[:200]!r}")
            print(f"  Parsed JSON object: {obj}")
        except Exception as e:
            print(f"  Sanity check FAILED ({type(e).__name__}): {e}")
            if USE_JSON:
                print("  Retrying request without strict JSON formatting...")
                USE_JSON = False
                try:
                    content = call_ollama(client, args.model, SYSTEM, user, use_json_format=False)
                    print(f"  Sanity check OK (no JSON constraint). prefix: {(content or '')[:200]!r}")
                except Exception as e2:
                    print(f"  Critical check FAILED again: {e2}")
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
                    'dev_row_idx': r['dev_row_idx'], 'tok_idx': r['tok_idx'],
                    'lang': r['lang'], 'token': r['token'], 'gt': r['gt'], 'cat': r['cat'],
                    'raw_response': None, 'normalized': None, 'error': f'build_failed: {e}',
                }, ensure_ascii=False) + "\n")
                continue
                
            meta = {
                'dev_row_idx': r['dev_row_idx'], 'tok_idx': r['tok_idx'],
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
                print(f"  [{i+1}/{len(futures)}] elapsed={el:.1f}s ETA={eta:.1f}s processed={n_done} errors={n_err} parse_fails={n_parse_fail}")

    print(f"\n[LLM Correction Completed] processed={n_done} errors={n_err} parse_fails={n_parse_fail}")
    print(f"  Results saved -> {out_path}")
    print(f"  Execution completed in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
