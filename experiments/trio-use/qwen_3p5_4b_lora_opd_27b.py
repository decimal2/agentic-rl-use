"""GSM8K OPD：Qwen3.5-4B LoRA student ← Qwen3.8-27B teacher。

和 SFT 的区别：
- SFT 直接用数据集里的 ``####`` 标准答案做 token 级监督；
- OPD 让 student 先按当前策略采样，再由 teacher 对 student 实际生成的 token
  计算 logprob，用 reverse KL 作为逐 token advantage。

每个 step 的数据流：
    student 采样 completion + old logprob
        -> teacher 对同一串 prompt + completion 计算 logprob
        -> reverse_kl = student_lp - teacher_lp
        -> advantage = -kl_penalty_coef * reverse_kl
        -> importance_sampling Datum -> forward_backward -> optim_step

运行（从仓库根目录）：
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py --dry-run
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py --max-steps 20
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py   # 跑满一个 epoch

nohup 示例：
    PYTHONUNBUFFERED=1 nohup uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py \
        --batch-size 8 --group-size 4 --save-name gsm8k-opd-27b \
        > experiments/trio-use/opd.log 2>&1 &
    echo $! > experiments/trio-use/opd.pid

训练完成后用 SFT 脚本评测（会同时输出 base 和 OPD checkpoint）：
    uv run python experiments/trio-use/qwen_3p5_4b_lora_sft.py --eval-only \
        --save-name gsm8k-opd-27b-final \
        --model-path 'trio://RUN_ID/sampler_weights/gsm8k-opd-27b-final-weights'
"""

import argparse
import random
import signal
import time
from pathlib import Path

from datasets import load_dataset
import numpy as np
import pytrio as trio
from tqdm import tqdm

# teacher 打 logprob 是额外的远程请求，给请求超时留足余量。
trio.configure(request_timeout_seconds=600)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "database/gsm8k/main/train-*.parquet"
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K OPD: Qwen3.5-4B LoRA student <- Qwen3.8-27B teacher"
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="student 基础模型")
    parser.add_argument(
        "--teacher-base-model",
        default="Qwen/Qwen3.8-27B",
        help="teacher 基础模型；必须和 student 使用同一 tokenizer",
    )
    parser.add_argument(
        "--teacher-model-path",
        default=None,
        help="可选：用已训练的 sampler 权重当 teacher；不传则用 teacher base model",
    )
    parser.add_argument("--rank", type=int, default=32, help="student LoRA rank")
    parser.add_argument("--batch-size", type=int, default=8, help="每个 step 的 prompt 数")
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="每个 prompt 采样的 completion 数",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="最多跑多少 step；0 表示跑满一个 epoch",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="每条 completion 的 token 上限")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0, help="reverse KL advantage 系数")
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--sampler-refresh-steps",
        type=int,
        default=1,
        help="每隔多少 step 用最新 student 权重刷新 sampler",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每多少 step 保存一次 state + sampler 权重；0 表示只在结束时保存",
    )
    parser.add_argument("--save-name", default="gsm8k-opd-27b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计 prompt/step/rollout 数，不连接 PyTRIO",
    )
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 记录")
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-opd")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.sampler_refresh_steps <= 0:
        raise ValueError("--sampler-refresh-steps must be > 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.kl_penalty_coef <= 0:
        raise ValueError("--kl-penalty-coef must be > 0")
    return args


def load_gsm8k_questions(path: str) -> list[str]:
    """只读 question；OPD 不需要数据集里的标准答案。"""
    dataset = load_dataset("parquet", data_files={"train": path}, split="train")
    questions = [str(row["question"]).strip() for row in dataset]
    if not questions:
        raise ValueError(f"No questions loaded from {path}")
    return questions


def load_local_tokenizer(model_id: str):
    """dry-run 时使用本地 tokenizer，不创建远端训练任务。"""
    try:
        from modelscope import AutoTokenizer
    except ImportError:
        from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def build_prompt_text(tokenizer, question: str) -> str:
    """必须和 SFT / 后续评测使用完全相同的 prompt。"""
    messages = [{"role": "user", "content": question.strip() + QUESTION_SUFFIX}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_prompt_tokens(tokenizer, question: str) -> list[int]:
    prompt_ids = tokenizer.encode(
        build_prompt_text(tokenizer, question),
        add_special_tokens=False,
    )
    if not prompt_ids:
        raise ValueError("Prompt tokens are empty")
    return prompt_ids


def completion_teacher_logprobs(
    teacher_client,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """teacher 对 student 实际生成的 completion 逐 token 打分。

    compute_logprobs 返回与输入对齐的 logprob，第一个 token 没有前文，可能是 None。
    只取 completion 区间，长度必须和 completion_ids 一致。
    """
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()
    completion_logprobs = all_logprobs[len(prompt_ids) :]
    if len(completion_logprobs) != len(completion_ids) or any(
        value is None for value in completion_logprobs
    ):
        raise ValueError(
            "Invalid teacher logprobs for completion tokens: "
            f"prompt={len(prompt_ids)}, completion={len(completion_ids)}, "
            f"teacher={len(completion_logprobs)}"
        )
    return [float(value) for value in completion_logprobs]


def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: np.ndarray,
) -> trio.Datum:
    """构造 importance_sampling 所需的右移 Datum。

    右移后第一个 completion token 由最后一个 prompt token 预测，所以 prompt 区间
    占 ``len(prompt_ids) - 1`` 个位置；这些位置的 logprob/advantage 都置 0。
    """
    prompt_loss_len = len(prompt_ids) - 1
    input_ids = prompt_ids + completion_ids[:-1]
    target_ids = [0] * prompt_loss_len + completion_ids
    padded_logprobs = [0.0] * prompt_loss_len + list(old_logprobs)
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)
    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("OPD datum fields must have the same length")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


def run_prompt_rollout(
    student_sampler,
    teacher_client,
    tokenizer,
    question: str,
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> tuple[list[trio.Datum], list[float], list[int], int]:
    """对一个 prompt 采样 group_size 条 completion，并转成 OPD Datum。"""
    prompt_ids = build_prompt_tokens(tokenizer, question)
    result = student_sampler.sample(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    ).result()

    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    completion_token_counts: list[int] = []
    hit_max_tokens = 0

    for sequence in result.sequences:
        completion_ids = list(sequence.tokens)
        if not completion_ids:
            continue
        student_logprobs = [float(value) for value in sequence.logprobs]
        if len(student_logprobs) != len(completion_ids):
            raise ValueError("Student token/logprob length mismatch")

        teacher_logprobs = completion_teacher_logprobs(
            teacher_client=teacher_client,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
        )
        reverse_kl = np.asarray(student_logprobs) - np.asarray(teacher_logprobs)
        advantages = -args.kl_penalty_coef * reverse_kl

        datums.append(
            build_opd_datum(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                old_logprobs=student_logprobs,
                advantages=advantages,
            )
        )
        reverse_kls.extend(reverse_kl.tolist())
        completion_token_counts.append(len(completion_ids))
        if len(completion_ids) >= args.max_tokens:
            hit_max_tokens += 1

    return datums, reverse_kls, completion_token_counts, hit_max_tokens


def save_checkpoint(training_client, name: str) -> tuple[str, str]:
    """同时保存可续训的 state 和可推理的 sampler 权重。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    tqdm.write(f"Saved state [{name}]: {state.path}")
    tqdm.write(f"Saved sampler weights [{name}]: {weights.path}")
    return state.path, weights.path


def raise_keyboard_interrupt(signum, frame) -> None:
    """把 kill 发来的 SIGTERM 转成 KeyboardInterrupt，走中断保存逻辑。"""
    raise KeyboardInterrupt(f"received signal {signum}")


def start_swanlab(args: argparse.Namespace, dataset_size: int, total_steps: int):
    if not args.swanlab:
        return None
    import swanlab

    config = vars(args).copy()
    config["dataset_size"] = dataset_size
    config["total_steps"] = total_steps
    return swanlab.init(
        project=args.swanlab_project,
        name=args.save_name,
        mode=args.swanlab_mode,
        config=config,
        tags=["PyTRIO", "OPD", "GSM8K", "Qwen3.5-4B", "Qwen3.8-27B"],
        log_dir=str(Path(__file__).resolve().parent / "swanlog"),
    )


def train_one_epoch(
    training_client,
    teacher_client,
    tokenizer,
    questions: list[str],
    args: argparse.Namespace,
    swanlab_run=None,
) -> tuple[int, int, int]:
    """跑一个 epoch（或前 --max-steps 步）。

    Returns:
        (实际运行步数, 一个 epoch 总步数, 累计 completion token 数)
    """
    steps_per_epoch = (len(questions) + args.batch_size - 1) // args.batch_size
    steps_to_run = (
        steps_per_epoch if args.max_steps == 0 else min(args.max_steps, steps_per_epoch)
    )
    if steps_to_run < steps_per_epoch:
        print(f"Running {steps_to_run}/{steps_per_epoch} steps (--max-steps)")

    adam = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    stop = list(dict.fromkeys(t for t in (tokenizer.eos_token, "<|im_end|>") if t))
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=stop,
    )

    student_sampler = None
    completed_steps = 0
    last_saved_step = 0
    total_completion_tokens = 0
    progress = tqdm(range(steps_to_run), desc="OPD epoch 1", unit="step")

    try:
        for step in progress:
            step_start = time.time()
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                student_sampler = training_client.save_weights_and_get_sampling_client()

            start = step * args.batch_size
            batch = questions[start : start + args.batch_size]
            datums: list[trio.Datum] = []
            reverse_kls: list[float] = []
            completion_token_counts: list[int] = []
            hit_max_tokens = 0

            for question in batch:
                rollout = run_prompt_rollout(
                    student_sampler=student_sampler,
                    teacher_client=teacher_client,
                    tokenizer=tokenizer,
                    question=question,
                    args=args,
                    sampling_params=sampling_params,
                )
                prompt_datums, prompt_kls, prompt_counts, prompt_hit_max = rollout
                datums.extend(prompt_datums)
                reverse_kls.extend(prompt_kls)
                completion_token_counts.extend(prompt_counts)
                hit_max_tokens += prompt_hit_max

            if not datums:
                raise RuntimeError("No OPD datums were built; all completions were empty")

            fwd_bwd_future = training_client.forward_backward(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = training_client.optim_step(adam)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()
            completed_steps = step + 1

            elapsed = time.time() - step_start
            completion_tokens_total = int(sum(completion_token_counts))
            total_completion_tokens += completion_tokens_total
            metrics = {
                "data/prompts": len(batch),
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_token_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "opd/hit_max_tokens": hit_max_tokens,
                "step/completion_tokens_per_second": completion_tokens_total
                / max(elapsed, 1e-6),
            }
            metrics.update(
                {
                    f"trainer/{key}": float(value)
                    for key, value in dict(fwd_bwd_result.metrics).items()
                }
            )
            if swanlab_run is not None:
                import swanlab

                swanlab.log(metrics, step=completed_steps)

            loss_mean = float(fwd_bwd_result.metrics.get("loss_mean", 0.0))
            progress.set_postfix(
                reverse_kl=f"{metrics['opd/reverse_kl_mean']:.4f}",
                loss=f"{loss_mean:.4f}",
                tokens=completion_tokens_total,
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
        print(f"OPD interrupted after {completed_steps} steps: {error!r}")
        if completed_steps > last_saved_step:
            try:
                save_checkpoint(
                    training_client,
                    f"{args.save_name}-interrupted-step-{completed_steps}",
                )
            except Exception as save_error:  # noqa: BLE001 - 保存失败不应掩盖原始异常
                print(f"Failed to save interrupted checkpoint: {save_error!r}")
        raise

    return steps_to_run, steps_per_epoch, total_completion_tokens


def ensure_models_available(
    service_client,
    student_model: str,
    teacher_model: str,
) -> None:
    """创建 client 前检查模型目录和 runner 可用性，避免 remote 404。"""
    capabilities = service_client.get_server_capabilities()
    models = {model.model_name: model for model in capabilities.supported_models}
    for role, model_name in (("student", student_model), ("teacher", teacher_model)):
        model = models.get(model_name)
        if model is None:
            raise ValueError(
                f"{role} model {model_name!r} is not supported; "
                f"available: {sorted(models)}"
            )
        capability = model.training if role == "student" else model.sampling
        if not capability.available:
            raise ValueError(
                f"{role} model {model_name!r} is listed but currently unavailable "
                f"(training.available={model.training.available}, "
                f"sampling.available={model.sampling.available})"
            )


def dry_run(args: argparse.Namespace, questions: list[str]) -> None:
    """只做本地统计：prompt 长度、step 数、rollout 数。"""
    tokenizer = load_local_tokenizer(args.base_model)
    lengths = [
        len(build_prompt_tokens(tokenizer, question))
        for question in tqdm(questions, desc="Tokenizing", unit="prompt")
    ]
    steps_per_epoch = (len(questions) + args.batch_size - 1) // args.batch_size
    rollouts_per_step = args.batch_size * args.group_size
    print(f"Usable prompts: {len(questions)}")
    print(
        f"Prompt tokens: mean={np.mean(lengths):.1f}, "
        f"max={max(lengths)}, min={min(lengths)}"
    )
    print(
        f"One epoch @ batch_size={args.batch_size}: {steps_per_epoch} steps, "
        f"{rollouts_per_step} rollouts/step, "
        f"{len(questions) * args.group_size} rollouts total"
    )
    print(
        f"Every rollout: 1 student sample + 1 teacher compute_logprobs, "
        f"max_tokens={args.max_tokens}"
    )


def main() -> None:
    args = parse_args()
    questions = load_gsm8k_questions(args.train_file)
    random.Random(args.seed).shuffle(questions)
    print(f"Loaded {len(questions)} GSM8K training prompts")

    if args.dry_run:
        dry_run(args, questions)
        return

    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    steps_per_epoch = (len(questions) + args.batch_size - 1) // args.batch_size
    total_steps = (
        steps_per_epoch if args.max_steps == 0 else min(args.max_steps, steps_per_epoch)
    )
    swanlab_run = start_swanlab(args, len(questions), total_steps)

    try:
        service_client = trio.ServiceClient()
        ensure_models_available(
            service_client,
            student_model=args.base_model,
            teacher_model=args.teacher_base_model,
        )
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.rank,
            seed=args.seed,
        )
        teacher_client = service_client.create_sampling_client(
            base_model=args.teacher_base_model,
            model_path=args.teacher_model_path,
        )
        tokenizer = training_client.get_tokenizer()

        # student 和 teacher 的 token id 必须逐 token 对齐，否则 teacher logprob
        # 会悄悄对应到不同的 token，训练信号完全错误。
        teacher_tokenizer = teacher_client.get_tokenizer()
        probe = "PyTRIO OPD tokenizer alignment check"
        if teacher_tokenizer.encode(probe, add_special_tokens=False) != tokenizer.encode(
            probe, add_special_tokens=False
        ):
            raise ValueError(
                "Teacher and student tokenizers disagree; token ids cannot be aligned"
            )

        print(f"Student: {args.base_model} (LoRA rank {args.rank})")
        print(
            f"Teacher: {args.teacher_base_model}"
            + (
                f" @ {args.teacher_model_path}"
                if args.teacher_model_path
                else " (base model)"
            )
        )
        print(
            f"Training: {total_steps} steps, batch={args.batch_size} prompts, "
            f"group={args.group_size} completions/prompt"
        )

        steps_run, steps_total, total_tokens = train_one_epoch(
            training_client=training_client,
            teacher_client=teacher_client,
            tokenizer=tokenizer,
            questions=questions,
            args=args,
            swanlab_run=swanlab_run,
        )
        print(
            f"OPD finished | ran {steps_run}/{steps_total} steps | "
            f"total completion tokens={total_tokens}"
        )

        save_checkpoint(training_client, f"{args.save_name}-final")
    finally:
        if swanlab_run is not None:
            import swanlab

            swanlab.finish()


if __name__ == "__main__":
    main()
