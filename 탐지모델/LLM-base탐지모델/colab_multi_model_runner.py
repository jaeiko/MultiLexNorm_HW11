
import fnmatch
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path


def load_base_runner(project_dir):
    """기존 run_llm_experiment.py를 모듈로 불러온다.

    input:
        project_dir: LLM-base탐지모델 폴더 경로
    output:
        module: 기존 실험 러너 모듈
    """
    module_path = Path(project_dir) / "run_llm_experiment.py"
    spec = importlib.util.spec_from_file_location("llm_detection_base_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_PROFILES = {
    "mistral24b": {
        "experiment_name": "exp_014_mistral_small_24b_k20",
        "provider": "llama_cpp",
        "model": "Mistral-Small-3.2-24B-Instruct-2506",
        "hf_repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "filename_patterns": [
            "*Q3_K_M*.gguf",
            "*q3_k_m*.gguf",
            "*Q3_K_S*.gguf",
            "*q3_k_s*.gguf",
            "*UD-Q3*.gguf",
            "*Q4_K_M*.gguf",
            "*q4_k_m*.gguf",
        ],
        "quantization_preference": "Q3_K_M -> Q3_K_S -> Q4_K_M",
        "num_ctx": 16384,
        "num_predict": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "n_gpu_layers": -1,
        "n_batch": 512,
        "flash_attn": True,
        "think": False,
        "timeout": 120,
        "sleep": 0,
    },
    "qwen30b": {
        "experiment_name": "exp_015_qwen3_30b_a3b_k20",
        "provider": "llama_cpp",
        "model": "Qwen3-30B-A3B",
        "hf_repo": "tensorblock/Qwen_Qwen3-30B-A3B-GGUF",
        "filename_patterns": [
            "*Q3_K_S.gguf",
            "*q3_k_s.gguf",
            "*Q3_K_M.gguf",
            "*q3_k_m.gguf",
            "*Q2_K.gguf",
            "*q2_k.gguf",
        ],
        "quantization_preference": "Q3_K_S -> Q3_K_M -> Q2_K",
        "num_ctx": 16384,
        "num_predict": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "n_gpu_layers": -1,
        "n_batch": 512,
        "flash_attn": True,
        "think": False,
        "no_think_suffix": " /no_think",
        "timeout": 120,
        "sleep": 0,
    },
    "phi4": {
        "experiment_name": "exp_016_phi4_14b_k20",
        "provider": "llama_cpp",
        "model": "microsoft/phi-4",
        "hf_repo": "microsoft/phi-4-gguf",
        "filename_patterns": [
            "*Q4_K_M*.gguf",
            "*q4_k_m*.gguf",
            "*Q4_K_S*.gguf",
            "*q4_k_s*.gguf",
            "*Q3_K_M*.gguf",
            "*q3_k_m*.gguf",
        ],
        "quantization_preference": "Q4_K_M -> Q4_K_S -> Q3_K_M",
        "num_ctx": 16384,
        "num_predict": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "n_gpu_layers": -1,
        "n_batch": 512,
        "flash_attn": True,
        "think": False,
        "timeout": 120,
        "sleep": 0,
    },
}


def get_model_profile(profile_name):
    """모델 프로필 이름으로 실험 설정 dict를 복사해서 가져온다.

    input:
        profile_name: mistral24b, qwen30b, phi4 중 하나
    output:
        dict: 수정 가능한 모델 프로필 복사본
    """
    if profile_name not in MODEL_PROFILES:
        raise KeyError(f"Unknown MODEL_PROFILE: {profile_name}. Choose one of {list(MODEL_PROFILES)}")
    return dict(MODEL_PROFILES[profile_name])


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_experiment_configs(project_dir, profile, split="validation", lang=None, limit=None, offset=0):
    """모델 프로필을 기존 실험 폴더 형식의 config 3종으로 저장한다.

    input:
        project_dir: LLM-base탐지모델 폴더 경로
        profile: get_model_profile()로 가져온 모델 설정
        split: validation 또는 test
        lang: 특정 언어만 돌릴 때의 언어 코드. None이면 전체
        limit: smoke test용 row 제한. None이면 전체
        offset: 앞에서 건너뛸 row 수
    output:
        Path: 생성된 experiment_dir
    """
    project_dir = Path(project_dir)
    experiment_dir = project_dir / "experiments" / profile["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"{split}-00000-of-00001.parquet"
    dataset_path = f"../../multilexnorm2026-dataset/data/{dataset_name}"
    should_evaluate = split != "test"

    model_config = {
        "provider": profile["provider"],
        "model": profile["model"],
        "hf_repo": profile["hf_repo"],
        "filename_patterns": profile["filename_patterns"],
        "quantization_preference": profile.get("quantization_preference"),
        "temperature": profile.get("temperature", 0),
        "top_p": profile.get("top_p", 1.0),
        "num_ctx": profile.get("num_ctx", 16384),
        "num_predict": profile.get("num_predict", 1024),
        "think": profile.get("think", False),
        "no_think_suffix": profile.get("no_think_suffix", ""),
        "n_gpu_layers": profile.get("n_gpu_layers", -1),
        "n_batch": profile.get("n_batch", 512),
        "flash_attn": profile.get("flash_attn", True),
        "timeout": profile.get("timeout", 120),
        "sleep": profile.get("sleep", 0),
        "notes": "Gemma 4 26B k=20 config와 최대한 유사하게 맞춘 Colab GGUF 실행 설정",
    }
    prompt_config = {
        "prompt_template": "prompts/prompt_v1.txt",
        "fewshot": {
            "enabled": True,
            "strategy": "same_lang_normalized_edit_distance_label_balance",
            "train_dataset": "../../multilexnorm2026-dataset/data/train-00000-of-00001.parquet",
            "positive_k": 20,
            "negative_k": 20,
        },
    }
    run_config = {
        "dataset": dataset_path,
        "split": split,
        "lang": lang,
        "limit": limit,
        "offset": offset,
        "fallback": "1",
        "evaluate": should_evaluate,
        "resume": True,
        "flush_every": 1,
        "output_files": {
            "predictions": "predictions.txt",
            "debug": "debug.jsonl",
            "answer_labels": "answer_labels.txt",
            "benchmark_json": "benchmark.json",
            "benchmark_txt": "benchmark.txt",
            "summary": "summary.md",
        },
    }

    _write_json(experiment_dir / "LLM_model_config.json", model_config)
    _write_json(experiment_dir / "prompt_config.json", prompt_config)
    _write_json(experiment_dir / "run_config.json", run_config)
    return experiment_dir


def select_hf_file(repo_id, filename_patterns, token=None):
    """Hugging Face repo 안에서 선호 quantization 파일을 고른다.

    input:
        repo_id: Hugging Face repo id
        filename_patterns: 우선순위 순서의 glob 패턴 목록
        token: HF_TOKEN. public 모델이면 None이어도 됨
    output:
        str: 다운로드할 GGUF 파일명
    """
    from huggingface_hub import list_repo_files

    files = [name for name in list_repo_files(repo_id, token=token) if name.lower().endswith(".gguf")]
    if not files:
        raise FileNotFoundError(f"No GGUF file found in {repo_id}")

    for pattern in filename_patterns:
        matches = [name for name in files if fnmatch.fnmatch(name, pattern)]
        if matches:
            return sorted(matches, key=lambda name: (len(name), name))[0]

    raise FileNotFoundError(
        f"No GGUF file matched {filename_patterns} in {repo_id}. "
        f"Available examples: {files[:10]}"
    )


def download_gguf(profile, cache_dir):
    """프로필에 맞는 GGUF 파일을 Colab 로컬 디스크로 다운로드한다.

    input:
        profile: 모델 프로필 dict
        cache_dir: 모델 파일 저장 폴더
    output:
        Path: 다운로드된 GGUF 파일 경로
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = select_hf_file(profile["hf_repo"], profile["filename_patterns"], token=token)
    print(f"Selected GGUF: {profile['hf_repo']} / {filename}")
    return Path(
        hf_hub_download(
            repo_id=profile["hf_repo"],
            filename=filename,
            local_dir=cache_dir / profile["experiment_name"],
            local_dir_use_symlinks=False,
            token=token,
        )
    )


class LlamaCppClient:
    """llama-cpp-python으로 GGUF 모델을 호출하는 얇은 래퍼."""

    def __init__(self, profile, model_path):
        """GGUF 모델을 로드한다.

        input:
            profile: 모델 실행 옵션 dict
            model_path: GGUF 파일 경로
        output:
            None
        """
        from llama_cpp import Llama

        self.profile = profile
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=int(profile.get("num_ctx", 16384)),
            n_gpu_layers=int(profile.get("n_gpu_layers", -1)),
            n_batch=int(profile.get("n_batch", 512)),
            flash_attn=bool(profile.get("flash_attn", True)),
            verbose=False,
        )

    def generate(self, prompt):
        """단일 프롬프트에 대해 탐지 라벨 응답을 생성한다.

        input:
            prompt: 최종 조립된 탐지 프롬프트
        output:
            tuple[str, int]: 모델 응답 문자열, 생성 토큰 수
        """
        suffix = self.profile.get("no_think_suffix", "")
        if suffix:
            prompt = prompt.rstrip() + suffix

        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(self.profile.get("temperature", 0)),
            top_p=float(self.profile.get("top_p", 1.0)),
            max_tokens=int(self.profile.get("num_predict", 1024)),
        )
        content = response["choices"][0]["message"]["content"].strip()
        token_count = response.get("usage", {}).get("completion_tokens", 0)
        return content, int(token_count or 0)


def build_client(profile, cache_dir="/content/model_cache"):
    """프로필 provider에 맞는 LLM client를 만든다.

    input:
        profile: 모델 프로필 dict
        cache_dir: GGUF 파일 저장 폴더
    output:
        object: generate(prompt) 메서드를 가진 client
    """
    if profile.get("provider") != "llama_cpp":
        raise ValueError(f"Unsupported provider in Colab runner: {profile.get('provider')}")
    model_path = download_gguf(profile, cache_dir=cache_dir)
    return LlamaCppClient(profile, model_path)


def _read_lines(path):
    if not Path(path).exists():
        return []
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _truncate_lines(path, keep_count):
    """resume 중 파일별 줄 수가 어긋났을 때 안전한 공통 길이로 맞춘다."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()[:keep_count]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _append_line(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(str(value) + "\n")
        file.flush()


def _copy_if_exists(src, dst):
    """src 파일이 있으면 dst 위치로 복사한다."""
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_mirror_dir(experiment_dir, mirror_dir, output_names):
    """Drive mirror 폴더를 준비하고, 남아 있는 결과를 local experiment로 복원한다.

    input:
        experiment_dir: Colab 로컬 실험 폴더
        mirror_dir: Drive에 저장할 실험 폴더
        output_names: 복원/동기화할 output 파일명 목록
    output:
        Path | None: 준비된 mirror_dir
    """
    if mirror_dir is None:
        return None

    experiment_dir = Path(experiment_dir)
    mirror_dir = Path(mirror_dir)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    for name in ["LLM_model_config.json", "prompt_config.json", "run_config.json"]:
        _copy_if_exists(experiment_dir / name, mirror_dir / name)

    for name in output_names + ["progress.json"]:
        mirror_path = mirror_dir / name
        local_path = experiment_dir / name
        if mirror_path.exists() and not local_path.exists():
            _copy_if_exists(mirror_path, local_path)

    return mirror_dir


def _mirror_line(mirror_dir, file_name, value):
    """Drive mirror에도 같은 한 줄을 append한다."""
    if mirror_dir is not None:
        _append_line(Path(mirror_dir) / file_name, value)


def _mirror_file(experiment_dir, mirror_dir, file_name):
    """실험 결과 파일 하나를 Drive mirror로 복사한다."""
    if mirror_dir is not None:
        _copy_if_exists(Path(experiment_dir) / file_name, Path(mirror_dir) / file_name)


def run_experiment_streaming(project_dir, experiment_dir, client, resume=True, mirror_dir=None):
    """한 row 처리 직후마다 결과를 저장하면서 LLM 탐지 실험을 실행한다.

    input:
        project_dir: LLM-base탐지모델 폴더 경로
        experiment_dir: config와 output이 들어갈 실험 폴더
        client: generate(prompt) 메서드를 가진 모델 client
        resume: True면 기존 predictions/debug를 읽고 다음 row부터 이어서 실행
        mirror_dir: Drive에도 함께 저장할 실험 폴더. None이면 local에만 저장
    output:
        dict | None: validation이면 benchmark metrics, test면 None
    """
    runner = load_base_runner(project_dir)
    project_dir = Path(project_dir)
    experiment_dir = Path(experiment_dir)

    model_config = runner.read_json(experiment_dir / runner.CONFIG_MODEL)
    prompt_config = runner.read_json(experiment_dir / runner.CONFIG_PROMPT)
    run_config = runner.read_json(experiment_dir / runner.CONFIG_RUN)
    should_evaluate = bool(run_config.get("evaluate", run_config.get("split") != "test"))

    dataset_path = runner.resolve_path(run_config["dataset"], experiment_dir, project_dir)
    prompt_path = runner.resolve_path(prompt_config["prompt_template"], experiment_dir, project_dir)
    prompt_template = runner.read_text(prompt_path)
    rows = runner.load_eval_rows(
        dataset_path,
        lang=run_config.get("lang"),
        limit=run_config.get("limit"),
        offset=run_config.get("offset", 0),
    )

    fewshot_config = prompt_config.get("fewshot", {})
    fewshot_strategy = fewshot_config.get("strategy")
    train_examples = []
    if fewshot_config.get("enabled") and fewshot_strategy == "same_lang_normalized_edit_distance_label_balance":
        train_dataset_path = runner.resolve_path(fewshot_config["train_dataset"], experiment_dir, project_dir)
        train_examples = runner.load_train_examples(train_dataset_path)

    predictions_path = runner.output_path(experiment_dir, run_config, "predictions")
    debug_path = runner.output_path(experiment_dir, run_config, "debug")
    answer_path = runner.output_path(experiment_dir, run_config, "answer_labels")
    output_names = [
        run_config["output_files"]["predictions"],
        run_config["output_files"]["debug"],
        run_config["output_files"]["answer_labels"],
        run_config["output_files"]["benchmark_json"],
        run_config["output_files"]["benchmark_txt"],
        run_config["output_files"]["summary"],
    ]
    mirror_dir = _prepare_mirror_dir(experiment_dir, mirror_dir, output_names)

    if resume:
        predictions = _read_lines(predictions_path)
        answer_labels = _read_lines(answer_path) if should_evaluate else []
        debug_count = len(_read_lines(debug_path))
        processed_count = min(len(predictions), debug_count)
        if should_evaluate:
            processed_count = min(processed_count, len(answer_labels))
        if processed_count < len(predictions) or processed_count < debug_count:
            print(f"Resume files are misaligned. Truncating to {processed_count} completed rows.")
            _truncate_lines(predictions_path, processed_count)
            _truncate_lines(debug_path, processed_count)
            if should_evaluate:
                _truncate_lines(answer_path, processed_count)
            _mirror_file(experiment_dir, mirror_dir, predictions_path.name)
            _mirror_file(experiment_dir, mirror_dir, debug_path.name)
            if should_evaluate:
                _mirror_file(experiment_dir, mirror_dir, answer_path.name)
            predictions = predictions[:processed_count]
            answer_labels = answer_labels[:processed_count]
    else:
        processed_count = 0
        predictions = []
        answer_labels = []
        for path in [predictions_path, debug_path, answer_path]:
            Path(path).unlink(missing_ok=True)
            if mirror_dir is not None:
                (Path(mirror_dir) / Path(path).name).unlink(missing_ok=True)

    parse_failures = 0
    total_tokens = 0
    experiment_start = time.time()

    if processed_count:
        print(f"Resuming from row {processed_count + 1}/{len(rows)}")

    for position, row in enumerate(rows, start=1):
        if position <= processed_count:
            continue

        fewshot_examples = []
        fewshot_block = ""
        if fewshot_config.get("enabled") and fewshot_strategy == "same_lang_normalized_edit_distance_label_balance":
            fewshot_examples = runner.retrieve_fewshot_examples(
                row,
                train_examples,
                positive_k=int(fewshot_config.get("positive_k", 3)),
                negative_k=int(fewshot_config.get("negative_k", 3)),
            )
            fewshot_block = runner.format_fewshot_block(fewshot_examples)
        elif fewshot_config.get("enabled") and fewshot_strategy == "static_by_lang":
            fewshot_block = runner.load_static_fewshot_block(row, fewshot_config, experiment_dir, project_dir)

        prompt = runner.build_prompt(prompt_template, row, fewshot_block)
        response, eval_count = client.generate(prompt)
        total_tokens += eval_count
        if model_config.get("sleep", 0):
            time.sleep(float(model_config["sleep"]))

        label, reason, strict_parse = runner.parse_response(response, fallback=run_config.get("fallback", "1"))
        if not strict_parse:
            parse_failures += 1

        predictions.append(label)
        _append_line(predictions_path, label)
        _mirror_line(mirror_dir, predictions_path.name, label)

        if should_evaluate:
            answer_labels.append(row["answer_label"])
            _append_line(answer_path, row["answer_label"])
            _mirror_line(mirror_dir, answer_path.name, row["answer_label"])

        debug_record = {
            "position": position,
            "row_index": row["row_index"],
            "lang": row["lang"],
            "raw_text": row["raw_text"],
            "answer_label": row["answer_label"],
            "prediction": label,
            "reason": reason,
            "raw_response": response,
            "strict_parse": strict_parse,
            "fewshot_examples": fewshot_examples,
            "fewshot_strategy": fewshot_strategy,
            "fewshot_block": fewshot_block,
        }
        _append_line(debug_path, json.dumps(debug_record, ensure_ascii=False))
        _mirror_line(mirror_dir, debug_path.name, json.dumps(debug_record, ensure_ascii=False))
        progress = {
            "processed": position,
            "total": len(rows),
            "last_row_index": row["row_index"],
            "parse_failures_in_current_session": parse_failures,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        runner.write_json(experiment_dir / "progress.json", progress)
        if mirror_dir is not None:
            runner.write_json(Path(mirror_dir) / "progress.json", progress)
        print(f"[{position}/{len(rows)}] row={row['row_index']} label={label}", flush=True)

    elapsed = time.time() - experiment_start
    metrics = runner.benchmark_metrics(answer_labels, predictions) if should_evaluate else None
    if should_evaluate:
        runner.write_json(runner.output_path(experiment_dir, run_config, "benchmark_json"), metrics)
        runner.write_benchmark_text(runner.output_path(experiment_dir, run_config, "benchmark_txt"), metrics)
        _mirror_file(experiment_dir, mirror_dir, run_config["output_files"]["benchmark_json"])
        _mirror_file(experiment_dir, mirror_dir, run_config["output_files"]["benchmark_txt"])

    runner.write_summary(
        runner.output_path(experiment_dir, run_config, "summary"),
        experiment_dir,
        model_config,
        prompt_config,
        run_config,
        metrics,
        parse_failures,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        total_predictions=len(predictions),
    )
    _mirror_file(experiment_dir, mirror_dir, run_config["output_files"]["summary"])
    print(f"\nSaved experiment: {experiment_dir}")
    if mirror_dir is not None:
        print(f"Mirrored experiment: {mirror_dir}")
    if metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return metrics
