"""GSM8K OPD：Qwen3.5-4B LoRA student ← Qwen3.8-27B teacher。

和 SFT 的区别：
- SFT 直接用数据集里的 ``####`` 标准答案做 token 级监督；
- OPD 让 student 先按当前策略采样，再由 teacher 对 student 实际生成的 token
  计算 logprob，用 reverse KL 作为逐 token advantage。

每个 step 的数据流：
    student 采样 completion + old logprob
        -> teacher 对同一串 prompt + completion 计算 logprob
        -> reverse_kl = student_lp - teacher_lp
        -> advantage = -reverse_kl
        -> importance_sampling Datum -> forward_backward -> optim_step

固定使用 temperature=1、无 top-p/top-k 截断，并在每步更新前刷新 student sampler。

运行（从仓库根目录）：
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py --dry-run
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py --max-steps 20
    uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py   # 跑满一个 epoch

nohup 示例：
    PYTHONUNBUFFERED=1 nohup uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py \
        --batch-size 16 --group-size 1 --save-name gsm8k-opd-27b \
        > experiments/trio-use/opd.log 2>&1 &
    echo $! > experiments/trio-use/opd.pid

每 100 步（--save-every，默认 100）保存一次 state + sampler 权重，并在本地
``experiments/trio-use/checkpoints/`` 写入同名 JSON 元数据（step、data_offset、
batch_size、seed、数据集指纹）。续训时自动读取元数据，从断点对应的数据和全局
step 继续：
    PYTHONUNBUFFERED=1 nohup uv run python experiments/trio-use/qwen_3p5_4b_lora_opd_27b.py \
        --batch-size 16 \
        --group-size 1 \
        --resume-from 'trio://run_6rfsab589psb/weights/gsm8k-opd-27b-step-300-state' \
        --max-steps 468 \
        --save-name gsm8k-opd-27b-v2 > experiments/trio-use/opd-v2.log 2>&1 & 
    echo $! > experiments/trio-use/opd-v2.pid

``--max-steps`` 是全局 step 上限；0 表示跑到本 epoch 数据结束。没有元数据的旧
checkpoint 会用 ``--resume-step`` 或 checkpoint 名里的 step 推算数据偏移。
若训练更新已提交但结果未确认就中断，不保存未知状态；请从最近的可靠 checkpoint 续训。

训练完成后用 SFT 脚本评测（会同时输出 base 和 OPD checkpoint）：
    uv run python experiments/trio-use/qwen_3p5_4b_lora_sft.py --eval-only \
        --save-name gsm8k-opd-27b-final \
        --model-path 'trio://RUN_ID/sampler_weights/gsm8k-opd-27b-final-weights'
"""

import argparse
import hashlib
import json
import random
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytrio as trio
from datasets import load_dataset
from tqdm import tqdm

# teacher 打 logprob 是额外的远程请求，给请求超时留足余量。
trio.configure(request_timeout_seconds=600)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "database/gsm8k/main/train-*.parquet"
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
CHECKPOINTS_DIR = REPO_ROOT / "experiments/trio-use/checkpoints"
RESUME_STEP_RE = re.compile(r"-step-(\d+)-state$")


@dataclass(frozen=True)
class RunPlan:
    """本次运行的全局 step 和数据游标。"""

    resume_step: int
    data_offset: int
    steps_to_run: int
    questions_sha256: str
    metadata_found: bool

    @property
    def end_step(self) -> int:
        return self.resume_step + self.steps_to_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K OPD: Qwen3.5-4B LoRA student <- Qwen3.8-27B teacher"
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.5-4B", help="student 基础模型"
    )
    parser.add_argument(
        "--teacher-base-model",
        default="Qwen/Qwen3.8-27B",
        help="teacher 基础模型；必须和 student 使用同一 tokenizer",
    )
    parser.add_argument("--rank", type=int, default=32, help="student LoRA rank")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="每个 step 的 prompt 数"
    )
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
        help="跑到全局第多少 step 为止；0 表示跑到本 epoch 数据结束",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="每条 completion 的 token 上限"
    )
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="每多少 step 保存一次可续训 state + sampler 权重；0 表示只在结束时保存",
    )
    parser.add_argument("--save-name", default="gsm8k-opd-27b")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="从 save_state 的 checkpoint URI 继续训练（恢复权重和 optimizer）",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="断点已完成 step 数；不传时优先读本地元数据，其次从 checkpoint 名解析",
    )
    parser.add_argument(
        "--skip-prompts",
        type=int,
        default=None,
        help="直接指定数据偏移；用于断点 batch_size 与当前不一致的情况",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计 prompt/step/rollout 数，不连接 PyTRIO",
    )
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 记录")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.skip_prompts is not None and args.skip_prompts < 0:
        raise ValueError("--skip-prompts must be >= 0")
    if not args.resume_from and (
        args.resume_step is not None or args.skip_prompts is not None
    ):
        raise ValueError("--resume-step/--skip-prompts require --resume-from")
    return args


def load_gsm8k_questions(path: str) -> list[str]:
    """只读 question；OPD 不需要数据集里的标准答案。"""
    dataset = load_dataset("parquet", data_files={"train": path}, split="train")
    questions = [str(row["question"]).strip() for row in dataset]
    if not questions:
        raise ValueError(f"No questions loaded from {path}")
    return questions


def sha256_of_questions(questions: list[str]) -> str:
    """对打乱后的 question 列表做指纹，用于 resume 时校验数据顺序。"""
    return hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()


def parse_step_from_uri(uri: str) -> int | None:
    match = RESUME_STEP_RE.search(uri.rstrip("/"))
    return int(match.group(1)) if match else None


def find_checkpoint_metadata(state_uri: str) -> dict | None:
    """按 state URI 查找本地 checkpoint 元数据。"""
    matches = []
    for path in CHECKPOINTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("state_uri") == state_uri:
            matches.append(data)
    if len(matches) > 1:
        raise ValueError(f"Multiple checkpoint metadata files match {state_uri}")
    return matches[0] if matches else None


def resolve_run_plan(
    args: argparse.Namespace,
    questions: list[str],
    questions_sha256: str,
) -> RunPlan:
    """解析 resume step 和数据偏移，返回本次运行的全局 step 范围。"""
    num_questions = len(questions)
    metadata_found = False

    if not args.resume_from:
        resume_step = 0
        data_offset = 0
    else:
        metadata = find_checkpoint_metadata(args.resume_from)
        if metadata is not None:
            metadata_found = True
            resume_step = int(metadata["checkpoint_step"])
            if args.resume_step is not None and args.resume_step != resume_step:
                raise ValueError(
                    f"--resume-step {args.resume_step} does not match checkpoint "
                    f"metadata step {resume_step}"
                )
            for key, current in (
                ("seed", args.seed),
                ("num_questions", num_questions),
                ("questions_sha256", questions_sha256),
            ):
                if metadata.get(key) != current:
                    raise ValueError(
                        f"Checkpoint {key}={metadata.get(key)!r} does not match "
                        f"current {current!r}; use the original dataset and --seed"
                    )
            data_offset = int(metadata["data_offset"])
        else:
            resume_step = args.resume_step
            if resume_step is None:
                resume_step = parse_step_from_uri(args.resume_from)
            if resume_step is None:
                raise ValueError(
                    f"No checkpoint metadata for {args.resume_from} and cannot parse "
                    "a step number from its name; pass --resume-step"
                )
            data_offset = resume_step * args.batch_size

        if args.skip_prompts is not None:
            data_offset = args.skip_prompts
        if metadata is not None and metadata.get("batch_size") != args.batch_size:
            print(
                f"Warning: checkpoint batch_size={metadata.get('batch_size')}; "
                f"continuing with {args.batch_size} from data_offset={data_offset}"
            )

    if not 0 <= data_offset < num_questions:
        raise ValueError(
            f"data_offset {data_offset} is outside [0, {num_questions}); "
            "the epoch is already finished or the offset is invalid"
        )

    remaining_steps = (
        num_questions - data_offset + args.batch_size - 1
    ) // args.batch_size
    if args.max_steps and args.max_steps <= resume_step:
        raise ValueError(
            f"--max-steps {args.max_steps} must exceed resume step {resume_step}"
        )
    steps_to_run = (
        min(args.max_steps - resume_step, remaining_steps)
        if args.max_steps
        else remaining_steps
    )
    end_step = resume_step + steps_to_run
    if args.max_steps > end_step:
        print(
            f"--max-steps {args.max_steps} is beyond the epoch end; "
            f"stopping at global step {end_step}"
        )
    return RunPlan(
        resume_step=resume_step,
        data_offset=data_offset,
        steps_to_run=steps_to_run,
        questions_sha256=questions_sha256,
        metadata_found=metadata_found,
    )


def build_checkpoint_metadata(
    args: argparse.Namespace,
    plan: RunPlan,
    num_questions: int,
    global_step: int,
) -> dict:
    """生成写入本地 JSON 的 checkpoint 元数据。"""
    data_offset = min(
        plan.data_offset + (global_step - plan.resume_step) * args.batch_size,
        num_questions,
    )
    return {
        "run_name": args.save_name,
        "checkpoint_step": global_step,
        "data_offset": data_offset,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "seed": args.seed,
        "num_questions": num_questions,
        "questions_sha256": plan.questions_sha256,
    }


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


def ensure_tokenizers_aligned(student_tokenizer, teacher_tokenizer) -> None:
    """teacher 接收 student token ID，必须校验完整词表和特殊 token 配置。"""
    if (
        student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
        or student_tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map
    ):
        raise ValueError(
            "Teacher and student tokenizers disagree; token ids cannot be aligned"
        )


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
        advantages = -reverse_kl

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


def save_checkpoint(
    training_client,
    name: str,
    metadata: dict,
) -> None:
    """同时保存可续训的 state、可推理的 sampler 权重和本地元数据。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    tqdm.write(f"Saved state [{name}]: {state.path}")
    tqdm.write(f"Saved sampler weights [{name}]: {weights.path}")
    payload = {
        **metadata,
        "checkpoint_name": name,
        "state_uri": state.path,
        "weights_uri": weights.path,
    }
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = CHECKPOINTS_DIR / f"{name}.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tqdm.write(f"Saved checkpoint metadata [{name}]: {metadata_path}")


def raise_keyboard_interrupt(signum, frame) -> None:
    """把 kill 发来的 SIGTERM 转成 KeyboardInterrupt，走中断保存逻辑。"""
    raise KeyboardInterrupt(f"received signal {signum}")


def start_swanlab(args: argparse.Namespace, dataset_size: int, plan: RunPlan):
    if not args.swanlab:
        return None
    import swanlab

    config = vars(args).copy()
    config["dataset_size"] = dataset_size
    config["total_steps"] = plan.end_step
    config["resume_step"] = plan.resume_step
    config["data_offset"] = plan.data_offset
    config["steps_to_run"] = plan.steps_to_run
    config["metadata_found"] = plan.metadata_found
    return swanlab.init(
        project="agentic-rl-lab-opd",
        name=args.save_name,
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
    plan: RunPlan,
    swanlab_run=None,
) -> tuple[int, int]:
    """跑 RunPlan 指定的全局 step 范围。

    Returns:
        (实际运行步数, 累计 completion token 数)
    """
    adam = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
    )
    stop = list(dict.fromkeys(t for t in (tokenizer.eos_token, "<|im_end|>") if t))
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        stop=stop,
    )

    completed_steps = 0
    update_pending = False
    last_saved_step = plan.resume_step
    total_completion_tokens = 0
    progress = tqdm(
        range(plan.steps_to_run),
        desc=f"OPD step {plan.resume_step + 1}-{plan.end_step}",
        unit="step",
    )

    try:
        for local_step in progress:
            step_start = time.time()
            global_step = plan.resume_step + local_step + 1
            student_sampler = training_client.save_weights_and_get_sampling_client()

            start = plan.data_offset + local_step * args.batch_size
            batch = questions[start : start + args.batch_size]
            if not batch:
                raise RuntimeError(f"No prompts at data offset {start}")
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
                raise RuntimeError(
                    "No OPD datums were built; all completions were empty"
                )

            # 从提交开始到本地确认完成，远端状态可能已改变，异常时不能标记为旧 step。
            update_pending = True
            fwd_bwd_future = training_client.forward_backward(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = training_client.optim_step(adam)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()
            completed_steps = local_step + 1
            update_pending = False

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

                swanlab.log(metrics, step=global_step)

            loss_mean = float(fwd_bwd_result.metrics.get("loss_mean", 0.0))
            progress.set_postfix(
                step=f"{global_step}/{plan.end_step}",
                reverse_kl=f"{metrics['opd/reverse_kl_mean']:.4f}",
                loss=f"{loss_mean:.4f}",
                tokens=completion_tokens_total,
            )

            if (
                args.save_every > 0
                and global_step % args.save_every == 0
                and global_step < plan.end_step
            ):
                save_checkpoint(
                    training_client,
                    f"{args.save_name}-step-{global_step}",
                    metadata=build_checkpoint_metadata(
                        args, plan, len(questions), global_step
                    ),
                )
                last_saved_step = global_step
    except BaseException as error:
        progress.close()
        global_step = plan.resume_step + completed_steps
        print(
            f"OPD interrupted after {completed_steps} steps "
            f"(global step {global_step}): {error!r}"
        )
        if update_pending:
            print(
                "Remote update status is uncertain; skipping interrupted checkpoint. "
                f"Resume from the last reliable checkpoint (step {last_saved_step}), "
                "or restart if none exists."
            )
        elif global_step > last_saved_step:
            try:
                save_checkpoint(
                    training_client,
                    f"{args.save_name}-interrupted-step-{global_step}",
                    metadata=build_checkpoint_metadata(
                        args, plan, len(questions), global_step
                    ),
                )
            except Exception as save_error:  # noqa: BLE001 - 保存失败不应掩盖原始异常
                print(f"Failed to save interrupted checkpoint: {save_error!r}")
        raise

    return completed_steps, total_completion_tokens


def ensure_models_available(
    service_client,
    student_model: str | None,
    teacher_model: str,
) -> None:
    """创建 client 前检查模型目录和 runner 可用性，避免 remote 404。"""
    capabilities = service_client.get_server_capabilities()
    models = {model.model_name: model for model in capabilities.supported_models}
    for role, model_name in (("student", student_model), ("teacher", teacher_model)):
        # resume 时 student 的 base model 由 checkpoint 元数据决定，这里只检查 teacher。
        if model_name is None:
            continue
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


def dry_run(
    args: argparse.Namespace,
    questions: list[str],
    plan: RunPlan,
) -> None:
    """只做本地统计：prompt 长度、step 范围、rollout 数。"""
    tokenizer = load_local_tokenizer(args.base_model)
    lengths = [
        len(build_prompt_tokens(tokenizer, question))
        for question in tqdm(questions, desc="Tokenizing", unit="prompt")
    ]
    rollouts_per_step = args.batch_size * args.group_size
    print(f"Usable prompts: {len(questions)}")
    print(
        f"Resume: step={plan.resume_step}, "
        f"data_offset={plan.data_offset}/{len(questions)}, "
        f"metadata_found={plan.metadata_found}"
    )
    print(
        f"Run plan: {plan.steps_to_run} steps, "
        f"global {plan.resume_step + 1}..{plan.end_step}, "
        f"{rollouts_per_step} rollouts/step"
    )
    print(
        f"Prompt tokens: mean={np.mean(lengths):.1f}, "
        f"max={max(lengths)}, min={min(lengths)}"
    )
    print(
        f"Each prompt: 1 student sample request + up to {args.group_size} teacher requests, "
        f"max_tokens={args.max_tokens}"
    )


def main() -> None:
    args = parse_args()
    questions = load_gsm8k_questions(args.train_file)
    random.Random(args.seed).shuffle(questions)
    questions_sha256 = sha256_of_questions(questions)
    print(f"Loaded {len(questions)} GSM8K training prompts")

    plan = resolve_run_plan(args, questions, questions_sha256)

    if args.dry_run:
        dry_run(args, questions, plan)
        return

    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    swanlab_run = start_swanlab(args, len(questions), plan)

    try:
        service_client = trio.ServiceClient()
        ensure_models_available(
            service_client,
            student_model=None if args.resume_from else args.base_model,
            teacher_model=args.teacher_base_model,
        )
        if args.resume_from:
            print(f"Resuming student from state: {args.resume_from}")
            training_client = (
                service_client.create_training_client_from_state_with_optimizer(
                    args.resume_from
                )
            )
        else:
            training_client = service_client.create_lora_training_client(
                base_model=args.base_model,
                rank=args.rank,
                seed=args.seed,
            )
        teacher_client = service_client.create_sampling_client(
            base_model=args.teacher_base_model,
        )
        tokenizer = training_client.get_tokenizer()

        ensure_tokenizers_aligned(tokenizer, teacher_client.get_tokenizer())

        if args.resume_from:
            print(f"Student: resumed from {args.resume_from}")
        else:
            print(f"Student: {args.base_model} (LoRA rank {args.rank})")
        print(f"Teacher: {args.teacher_base_model} (base model)")
        print(
            f"Training: global steps {plan.resume_step + 1}..{plan.end_step} "
            f"({plan.steps_to_run} steps), batch={args.batch_size} prompts, "
            f"group={args.group_size} completions/prompt"
        )

        steps_run, total_tokens = train_one_epoch(
            training_client=training_client,
            teacher_client=teacher_client,
            tokenizer=tokenizer,
            questions=questions,
            args=args,
            plan=plan,
            swanlab_run=swanlab_run,
        )
        final_step = plan.resume_step + steps_run
        print(
            f"OPD finished | ran {steps_run} steps "
            f"(global {plan.resume_step + 1}..{final_step}) | "
            f"total completion tokens={total_tokens}"
        )

        save_checkpoint(
            training_client,
            f"{args.save_name}-final",
            metadata=build_checkpoint_metadata(args, plan, len(questions), final_step),
        )
    finally:
        if swanlab_run is not None:
            import swanlab

            swanlab.finish()


if __name__ == "__main__":
    main()
