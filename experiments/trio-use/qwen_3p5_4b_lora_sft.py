"""GSM8K LoRA SFT：让模型学会用 \\boxed{} 输出最终答案。

prompt 使用 Qwen chat template（enable_thinking=False），后续 GRPO rollout 应复用
``build_prompt_text``，保证训练和采样格式一致。

运行（从仓库根目录）：
    uv run python experiments/trio-use/trio_forward.py --dry-run
    uv run python experiments/trio-use/trio_forward.py --max-steps 20
    uv run python experiments/trio-use/trio_forward.py   # 跑满一个 epoch

训练结束后默认在全部 1319 条 test 上依次评测 base 和 SFT；也可以不训练、单独评测：
    uv run python experiments/trio-use/trio_forward.py --eval-only   # 仅基模
    uv run python experiments/trio-use/trio_forward.py --eval-only \
        --model-path 'trio://RUN_ID/sampler_weights/gsm8k-boxed-sft-final-weights'
    # 指定 --model-path 时，会先评测 base，再评测该 checkpoint。
"""

import argparse
import json
import random
import re
import signal
from pathlib import Path

from datasets import load_dataset
import pytrio as trio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "database/gsm8k/main/train-*.parquet"
DEFAULT_TEST_FILE = REPO_ROOT / "database/gsm8k/main/test-*.parquet"
DEFAULT_EVAL_DIR = REPO_ROOT / "experiments/trio-use/eval-results"
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
CALC_ANNOTATION_RE = re.compile(r"<<.*?>>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K LoRA SFT for \\boxed{} format")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=8, help="每个 step 的样本数")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="超过该 token 数的样本会被跳过，避免截断 \\boxed{} 和 EOS",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="最多跑多少 step；0 表示跑满一个 epoch",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每多少 step 保存一次 state + sampler 权重；0 表示只在结束时保存",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=0,
        help="评测 test split 的前 N 条；0 表示全部 1319 条",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=32,
        help="评测时同时提交的采样请求数",
    )
    parser.add_argument(
        "--eval-print",
        type=int,
        default=3,
        help="终端打印前 N 条评测样例，其余只写入结果文件",
    )
    parser.add_argument(
        "--eval-output-dir",
        default=str(DEFAULT_EVAL_DIR),
        help="逐条评测结果 JSONL 和汇总 JSON 的输出目录",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，直接评测 --model-path（不传则评测基模）",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="--eval-only 时评测的 sampler 权重，如 trio://.../sampler_weights/...",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-name", default="gsm8k-boxed-sft")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--no-eval", action="store_true", help="训练后不评测 base 和 SFT")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计一个 epoch 的 step 和 token 数，不连接 PyTRIO",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.eval_examples < 0:
        raise ValueError("--eval-examples must be >= 0")
    if args.eval_concurrency <= 0:
        raise ValueError("--eval-concurrency must be > 0")
    if args.model_path and not args.eval_only:
        raise ValueError("--model-path is only used with --eval-only")
    return args


def load_gsm8k(path: str, split: str):
    """从本地 ModelScope 镜像 parquet 文件加载 GSM8K。

    ``data_files`` 用 {split: path} 显式命名，否则 datasets 会把单文件命名为 train。
    """
    return load_dataset("parquet", data_files={split: path}, split=split)


def load_local_tokenizer(model_id: str):
    """dry-run 时使用本地 tokenizer，不创建远端训练任务。"""
    try:
        from modelscope import AutoTokenizer
    except ImportError:
        from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def build_prompt_text(tokenizer, question: str) -> str:
    """构造与后续 GRPO rollout 完全一致的 prompt。"""
    messages = [{"role": "user", "content": question.strip() + QUESTION_SUFFIX}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def split_gsm8k_answer(answer: str) -> tuple[str, str]:
    """把 'reasoning ... #### 72' 拆成解题过程和最终答案。"""
    reasoning, separator, final_answer = answer.rpartition("####")
    if not separator:
        raise ValueError(f"GSM8K answer missing '####': {answer!r}")
    return reasoning, final_answer.strip()


def clean_reasoning(reasoning: str) -> str:
    """去掉 GSM8K 的 <<48/2=24>> 计算器标注并整理空白。"""
    reasoning = CALC_ANNOTATION_RE.sub("", reasoning)
    reasoning = re.sub(r"[ \t]+\n", "\n", reasoning)
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    return reasoning.strip()


def build_target_text(answer: str) -> str:
    """生成训练目标：解题过程 + 最后一行 \\boxed{答案}。"""
    reasoning, final_answer = split_gsm8k_answer(answer)
    reasoning = clean_reasoning(reasoning)
    return f"{reasoning}\n\n\\boxed{{{final_answer}}}"


def process_example(example, tokenizer, max_seq_len: int) -> trio.Datum | None:
    """把单条 GSM8K 样本转成 cross_entropy 所需的 Datum。

    超长样本直接跳过，绝不从尾部截断，否则会把 \\boxed{} 和 EOS 切掉。
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    prompt_text = build_prompt_text(tokenizer, example["question"])
    target_text = build_target_text(example["answer"])

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(target_text, add_special_tokens=False)
    completion_ids.append(tokenizer.eos_token_id)

    tokens = prompt_ids + completion_ids
    weights = [0] * len(prompt_ids) + [1] * len(completion_ids)

    if len(tokens) > max_seq_len:
        return None

    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    weights = weights[1:]

    if not (len(input_ids) == len(target_ids) == len(weights)):
        raise ValueError("Datum fields must have the same token length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": target_ids,
            "weights": weights,
        },
    )


def prepare_datums(
    dataset,
    tokenizer,
    max_seq_len: int,
    seed: int,
) -> tuple[list[trio.Datum], int]:
    """打乱全集并转成 Datum；返回 (datums, 因超长跳过的样本数)。"""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    datums: list[trio.Datum] = []
    skipped = 0
    for index in tqdm(indices, desc="Tokenizing", unit="example"):
        datum = process_example(dataset[index], tokenizer, max_seq_len)
        if datum is None:
            skipped += 1
            continue
        datums.append(datum)
    return datums, skipped


def describe_dataset(datums: list[trio.Datum], skipped: int, batch_size: int) -> None:
    """打印一个 epoch 的 step 数和 token 统计。"""
    num_examples = len(datums)
    steps_per_epoch = num_examples // batch_size
    dropped = num_examples % batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            f"Only {num_examples} usable examples, fewer than batch_size={batch_size}"
        )

    compute_tokens = sum(len(datum.model_input) for datum in datums)
    loss_tokens = sum(
        sum(datum.loss_fn_inputs["weights"].tolist()) for datum in datums
    )
    prompt_tokens = compute_tokens - loss_tokens + num_examples

    print(f"Usable examples: {num_examples} (skipped {skipped} over max_seq_len)")
    print(
        f"Tokens: prompt={prompt_tokens}, completion/loss={loss_tokens}, "
        f"compute positions={compute_tokens}"
    )
    print(
        f"One epoch @ batch_size={batch_size}: {steps_per_epoch} steps "
        f"(drop last {dropped} examples, "
        f"~{compute_tokens // num_examples * batch_size} tokens/step)"
    )


def save_checkpoint(training_client, name: str) -> tuple[str, str]:
    """同时保存可续训的 state 和可推理的 sampler 权重，返回两者路径。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    tqdm.write(f"Saved state [{name}]: {state.path}")
    tqdm.write(f"Saved sampler weights [{name}]: {weights.path}")
    return state.path, weights.path


def raise_keyboard_interrupt(signum, frame) -> None:
    """把 kill 发来的 SIGTERM 转成 KeyboardInterrupt，以便走中断保存逻辑。"""
    raise KeyboardInterrupt(f"received signal {signum}")


def train_one_epoch(
    training_client,
    datums: list[trio.Datum],
    args: argparse.Namespace,
) -> tuple[dict[str, float], int, int, int]:
    """跑一个 epoch（或前 --max-steps 步）。

    Returns:
        (最后一步 metrics, 累计 loss tokens, 实际运行步数, 一个 epoch 总步数)
    """
    steps_per_epoch = len(datums) // args.batch_size
    steps_to_run = (
        steps_per_epoch if args.max_steps == 0 else min(args.max_steps, steps_per_epoch)
    )
    if steps_to_run < steps_per_epoch:
        print(f"Running {steps_to_run}/{steps_per_epoch} steps (--max-steps)")

    adam_params = trio.AdamParams(learning_rate=args.learning_rate)
    total_tokens = 0
    last_metrics: dict[str, float] = {}

    completed_steps = 0
    last_saved_step = 0

    progress = tqdm(range(steps_to_run), desc="SFT epoch 1", unit="step")
    try:
        for step in progress:
            start = step * args.batch_size
            batch = datums[start : start + args.batch_size]

            fwdbwd_future = training_client.forward_backward(
                batch,
                loss_fn="cross_entropy",
            )
            optim_future = training_client.optim_step(adam_params)

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()
            completed_steps = step + 1

            # 直接使用 PyTRIO 聚合后的 metrics，不再手动拼接 logprobs/weights。
            metrics = fwdbwd_result.metrics
            last_metrics = {
                "loss_sum": float(metrics["loss_sum"]),
                "loss_mean": float(metrics["loss_mean"]),
                "token_count": float(metrics["token_count"]),
            }
            token_count = int(last_metrics["token_count"])
            total_tokens += token_count

            progress.set_postfix(
                loss_mean=f"{last_metrics['loss_mean']:.4f}",
                tokens=token_count,
                total_tokens=total_tokens,
            )

            if args.save_every > 0 and completed_steps % args.save_every == 0:
                save_checkpoint(
                    training_client,
                    f"{args.save_name}-step-{completed_steps}",
                )
                last_saved_step = completed_steps
    except BaseException as error:
        # 异常 / Ctrl+C / SIGTERM：保存最后一个完整 step 之后的进度再退出。
        progress.close()
        print(f"Training interrupted after {completed_steps} steps: {error!r}")
        if completed_steps > last_saved_step:
            try:
                save_checkpoint(
                    training_client,
                    f"{args.save_name}-interrupted-step-{completed_steps}",
                )
            except Exception as save_error:  # noqa: BLE001 - 保存失败不应掩盖原始异常
                print(f"Failed to save interrupted checkpoint: {save_error!r}")
        raise

    return last_metrics, total_tokens, steps_to_run, steps_per_epoch


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def score_response(example, text: str) -> dict:
    """对单条回答打分：是否有 \\boxed{} 格式、答案是否与标准答案一致。"""
    prediction = extract_boxed(text)
    _, gold_answer = split_gsm8k_answer(example["answer"])
    has_format = prediction is not None
    is_correct = has_format and (
        normalize_answer(prediction) == normalize_answer(gold_answer)
    )
    return {
        "gold": gold_answer,
        "pred": prediction,
        "format_ok": has_format,
        "correct": is_correct,
    }


def evaluate(
    service_client,
    tokenizer,
    weights_path: str | None,
    args: argparse.Namespace,
    run_name: str,
) -> dict[str, float]:
    """在 GSM8K test split 上评测（默认全部 1319 条），贪心解码。

    采样请求按 --eval-concurrency 分批并发提交：sample() 提交后立即返回 future，
    整批提交完再统一 .result()。逐条结果写入 JSONL，汇总指标写入 JSON 并返回。
    weights_path 为 None 时评测基模。
    """
    dataset = load_gsm8k(args.test_file, split="test")
    num_examples = (
        len(dataset)
        if args.eval_examples == 0
        else min(args.eval_examples, len(dataset))
    )

    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=weights_path,
    )
    stop = [tokenizer.eos_token] if tokenizer.eos_token else None
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        stop=stop,
    )

    records: list[dict] = []
    progress = tqdm(total=num_examples, desc="Eval", unit="example")
    for chunk_start in range(0, num_examples, args.eval_concurrency):
        chunk_end = min(chunk_start + args.eval_concurrency, num_examples)
        futures = []
        for index in range(chunk_start, chunk_end):
            prompt = trio.ModelInput.from_ints(
                tokenizer.encode(
                    build_prompt_text(tokenizer, dataset[index]["question"]),
                    add_special_tokens=False,
                )
            )
            futures.append(
                (
                    index,
                    sampling_client.sample(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                )
            )

        for index, future in futures:
            sequence = future.result().sequences[0]
            text = sequence.text or ""
            example = dataset[index]
            record = {
                "index": index,
                "question": example["question"],
                **score_response(example, text),
                "output_tokens": len(sequence.tokens),
                "hit_max_tokens": len(sequence.tokens) >= args.max_new_tokens,
                "stop_reason": sequence.stop_reason,
                "text": text,
            }
            records.append(record)
            progress.update(1)
            progress.set_postfix(
                acc=f"{sum(r['correct'] for r in records) / len(records):.3f}"
            )
    progress.close()

    for record in records[: args.eval_print]:
        print(f"[eval {record['index']}] {record['question']}")
        print(f"  gold: {record['gold']}  pred: {record['pred']!r}")
        print(f"  text: {record['text']!r}")

    summary = {
        "run_name": run_name,
        "model_path": weights_path or "base",
        "num_examples": num_examples,
        "accuracy": sum(r["correct"] for r in records) / num_examples,
        "format_rate": sum(r["format_ok"] for r in records) / num_examples,
        "num_correct": sum(r["correct"] for r in records),
        "num_format_ok": sum(r["format_ok"] for r in records),
        "num_hit_max_tokens": sum(r["hit_max_tokens"] for r in records),
        "avg_output_tokens": sum(r["output_tokens"] for r in records) / num_examples,
    }

    output_dir = Path(args.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{run_name}-test.jsonl"
    summary_path = output_dir / f"{run_name}-test-summary.json"
    with records_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Eval [{run_name}] on {num_examples} test examples: "
        f"accuracy {summary['num_correct']}/{num_examples} "
        f"({summary['accuracy']:.2%}), "
        f"boxed format {summary['num_format_ok']}/{num_examples} "
        f"({summary['format_rate']:.2%}), "
        f"hit max_tokens {summary['num_hit_max_tokens']}, "
        f"avg output tokens {summary['avg_output_tokens']:.1f}"
    )
    print(f"Per-example results: {records_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()

    if args.eval_only:
        # 不创建训练任务：本地 tokenizer + 远端采样，评测基模或已有 checkpoint。
        tokenizer = load_local_tokenizer(args.base_model)
        service_client = trio.ServiceClient()
        if args.model_path:
            evaluate(
                service_client=service_client,
                tokenizer=tokenizer,
                weights_path=None,
                args=args,
                run_name="base",
            )
        evaluate(
            service_client=service_client,
            tokenizer=tokenizer,
            weights_path=args.model_path,
            args=args,
            run_name=args.save_name if args.model_path else "base",
        )
        return

    print(f"Loading GSM8K from: {args.train_file}")
    train_dataset = load_gsm8k(args.train_file, split="train")
    print(f"Loaded {len(train_dataset)} training examples")

    if args.dry_run:
        tokenizer = load_local_tokenizer(args.base_model)
        datums, skipped = prepare_datums(
            dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        )
        describe_dataset(datums, skipped, args.batch_size)
        return

    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
    )
    tokenizer = training_client.get_tokenizer()

    datums, skipped = prepare_datums(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    describe_dataset(datums, skipped, args.batch_size)

    first_datum = datums[0]
    first_target = first_datum.loss_fn_inputs["target_tokens"].tolist()
    first_weights = first_datum.loss_fn_inputs["weights"].tolist()
    first_train_index = first_weights.index(1)
    print(
        f"First datum: {len(first_datum.model_input)} positions, "
        f"{sum(first_weights)} train tokens"
    )
    print(
        "First target preview: "
        f"{tokenizer.decode(first_target[first_train_index:])[:300]!r}"
    )

    last_metrics, total_tokens, steps_run, steps_per_epoch = train_one_epoch(
        training_client,
        datums,
        args,
    )
    print(
        f"Training finished | ran {steps_run}/{steps_per_epoch} steps | "
        f"last step: loss_sum={last_metrics['loss_sum']:.4f}, "
        f"loss_mean={last_metrics['loss_mean']:.4f}, "
        f"token_count={int(last_metrics['token_count'])} | "
        f"total loss tokens={total_tokens}"
    )

    _, weights_path = save_checkpoint(training_client, f"{args.save_name}-final")

    if not args.no_eval:
        evaluate(
            service_client=service_client,
            tokenizer=tokenizer,
            weights_path=None,
            args=args,
            run_name="base",
        )
        evaluate(
            service_client=service_client,
            tokenizer=tokenizer,
            weights_path=weights_path,
            args=args,
            run_name=f"{args.save_name}-final",
        )


if __name__ == "__main__":
    main()


"""
PYTHONUNBUFFERED=1 nohup uv run python experiments/trio-use/trio_forward.py \
    --batch-size 16 \
    --save-name gsm8k-boxed-sft \
    > experiments/trio-use/training.log 2>&1 &
echo $! > experiments/trio-use/training.pid
"""
