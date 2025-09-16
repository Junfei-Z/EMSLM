# 作用：一键顺序跑 6 个 (模型, 量化) 动作；按动作建目录；逐题逐 turn 推理；
#       保存 answers.jsonl 和 segments.csv（UTC 时间，便于功率对齐）。
# 依赖：pip install llama-cpp-python pyyaml
# 建议：HWiNFO 独立运行采样功率（100ms），用 segments.csv 对齐时间段。

import os
import csv
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import yaml

from llama_cpp import Llama  # pip install llama-cpp-python

# ========= 你的 6 个动作（模型路径，按需调整） =========
ACTIONS = [
    r"E:\SLM\Q6\Qwen3-4B-Q6_K.gguf",
    r"E:\SLM\Q6\dolly-v2-3b.Q6_K.gguf",
    r"E:\SLM\Q6\gemma-2-2b-it-Q6_K.gguf",
    r"E:\SLM\Q8\Qwen3-4B-Q8_0.gguf",
    r"E:\SLM\Q8\dolly-v2-3b.Q8_0.gguf",
    r"E:\SLM\Q8\gemma-2-2b-it-Q8_0.gguf",
]

# ========= 输入题库（JSONL：question_id, turns[list[str]]） =========
QUESTION_FILE = r"E:\SLM\questionset.jsonl"

# ========= 推理固定参数（与你论文一致） =========
N_GPU_LAYERS = 48
MAX_TOKENS = 250
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 40
N_THREADS = None          # None=自动；或设置为 CPU 物理核数
N_CTX = 4096
SAMPLE_INTERVAL_MS = 100  # 记录用：功率采样 100ms（HWiNFO）

# ========= 输出根目录 =========
OUT_ROOT = Path(r"E:\SLM\runs_mtbench")

# ========= 工具函数 =========
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_action_name(model_path: str) -> str:
    """用模型文件名（去扩展名）作为目录名；替换掉点号。"""
    stem = Path(model_path).stem
    return stem.replace(".", "-")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_questions_jsonl(path: str):
    """读取 JSONL；返回 list[dict]，每个元素含 question_id(str), turns(list[str])。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            turns = obj.get("turns", [])
            if qid is None or not isinstance(turns, list) or len(turns) == 0:
                continue
            items.append({"question_id": str(qid), "turns": turns})
    return items

def build_llama(model_path: str) -> Llama:
    """
    创建 Llama 实例。
    若安装了 GPU 版 llama-cpp-python，n_gpu_layers>0 会把前 L 层放到 GPU 上。
    """
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        logits_all=False,
        verbose=False,
    )
    return llm

def infer_one(llm: Llama, prompt: str) -> dict:
    """
    单次推理。为稳健起见，既兼容 chat 模式也兼容 completion 模式。
    返回：answer/tokens_in/tokens_out/latency_s（tokens_* 若不可得则为 None）
    """
    # 优先走 chat 接口（常见指令微调模型）
    try:
        start = time.time()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
        )
        end = time.time()
        answer = result["choices"][0]["message"]["content"]
        # 尝试从 usage 里拿 token 统计（不一定都有）
        usage = result.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        return {
            "answer": answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": end - start,
        }
    except Exception:
        # 回退到普通 completion
        start = time.time()
        result = llm(
            prompt=prompt,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            echo=False,
        )
        end = time.time()
        answer = result["choices"][0]["text"]
        usage = result.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        return {
            "answer": answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": end - start,
        }

# ========= 主流程 =========
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    questions = load_questions_jsonl(QUESTION_FILE)
    if not questions:
        raise RuntimeError(f"No questions loaded from {QUESTION_FILE}")

    print(f"[loader] Loaded {len(questions)} questions from {QUESTION_FILE}")

    for model_path in ACTIONS:
        action_name = parse_action_name(model_path)
        run_dir = OUT_ROOT / action_name
        ensure_dir(run_dir)

        # 记录元数据
        meta = {
            "action_name": action_name,
            "model_path": model_path,
            "n_gpu_layers": N_GPU_LAYERS,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "n_threads": N_THREADS,
            "n_ctx": N_CTX,
            "sample_interval_ms": SAMPLE_INTERVAL_MS,
            "started_at_utc": now_iso(),
            "num_questions": len(questions),
            "notes": "Each question's turns are evaluated sequentially.",
        }
        with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True)

        # 保存题目快照
        with (run_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 初始化模型（每个 action 重新加载，保证隔离）
        print(f"[model] Loading {model_path} ...")
        llm = build_llama(model_path)
        print(f"[model] Ready: {action_name}")

        # 输出文件
        answers_path = run_dir / "answers.jsonl"
        segments_csv = run_dir / "segments.csv"
        with answers_path.open("w", encoding="utf-8") as f_ans, \
             segments_csv.open("w", newline="", encoding="utf-8") as f_seg:

            seg_writer = csv.writer(f_seg)
            seg_writer.writerow(["qid", "turn_idx", "prompt_sha1", "start_utc", "end_utc", "elapsed_s"])

            # 逐题，逐 turn
            for qi, q in enumerate(questions, 1):
                qid = q["question_id"]
                for ti, prompt in enumerate(q["turns"], 1):
                    prompt_hash = sha1(prompt)

                    # 开始时间（对齐 HWiNFO 功率）
                    t0 = time.time()
                    t0_iso = now_iso()

                    out = infer_one(llm, prompt)

                    # 结束时间
                    t1 = time.time()
                    t1_iso = now_iso()
                    elapsed = t1 - t0

                    # 记录 segments（UTC）
                    seg_writer.writerow([qid, ti, prompt_hash, t0_iso, t1_iso, f"{elapsed:.3f}"])

                    # 写答案（JSONL 一行一条）
                    rec = {
                        "qid": qid,
                        "turn_idx": ti,
                        "prompt_sha1": prompt_hash,
                        "answer": out.get("answer", ""),
                        "tokens_in": out.get("tokens_in"),
                        "tokens_out": out.get("tokens_out"),
                        "latency_s": out.get("latency_s", elapsed),
                        "model_path": model_path,
                        "action_name": action_name,
                        "n_gpu_layers": N_GPU_LAYERS,
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                    }
                    f_ans.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_ans.flush()

                print(f"[{action_name}] question {qi}/{len(questions)} done")

        (run_dir / "DONE.flag").write_text(now_iso(), encoding="utf-8")
        print(f"[{action_name}] ALL DONE -> {run_dir}")

if __name__ == "__main__":
    main()
