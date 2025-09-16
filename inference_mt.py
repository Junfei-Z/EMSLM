# -*- coding: utf-8 -*-
"""
批量测 MT-Bench 单题推理能耗（HWiNFO64 Pro + llama.cpp 本地模型）
- 每题：以管理员方式启动 HWiNFO 日志(100ms)，执行一次推理，停止日志
- 仅提取 CPU 封装功率 与 GPU 功率；按真实时间戳做梯形积分
- 按题保存窗口CSV，并汇总 summary.csv

先决条件：
1) 已安装 HWiNFO64 Pro，能以“传感器模式”记录日志；首次需管理员打开并同意 EULA
2) HWiNFO 可被命令行启动日志： -log_format=1 -poll_rate=100 -l"<csv_path>"
3) 安装：pip install llama-cpp-python datasets pandas numpy
"""

import os, re, time, subprocess, ctypes, argparse, json
from ctypes import wintypes
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from datasets import load_dataset
from llama_cpp import Llama

# ====== 你可以改的默认配置 ======
HWINFO_EXE = r"C:\Program Files\HWiNFO64\HWiNFO64.exe"
HWINFO_DIR = r"C:\Program Files\HWiNFO64"
DATA_PATH = r"C:\Users\Administrator\Desktop\project\energy and inference quality modeling on device SLM\question.jsonl.txt"
POLL_MS    = 100          # 100ms 采样
WARMUP     = True         # 每题前做 1 token 预热
KEEP_TIME  = True         # 导出时保留 Timestamp 列
TURN_INDEX = 1            # 默认使用第 1 轮问题（human 第一次提问）

# 替换成你的实际模型路径
# MODEL_PATH = r"E:\SLM\Q6\Qwen3-4B-Q6_K.gguf"
# MODEL_PATH = r"E:\SLM\Q6\dolly-v2-3b.Q6_K.gguf"  
# MODEL_PATH = r"E:\SLM\Q6\gemma-2-2b-it-Q6_K.gguf"
# MODEL_PATH = r"E:\SLM\Q8\Qwen3-4B-Q8_0.gguf"
# MODEL_PATH = r"E:\SLM\Q8\dolly-v2-3b.Q8_0.gguf"
MODEL_PATH = r"E:\SLM\Q8\gemma-2-2b-it-Q8_0.gguf"

LLAMA_KW = dict(
    n_gpu_layers=48,
    n_ctx=2048,
    n_batch=256,
    n_ubatch=256,
    flash_attn=False,
    use_mmap=True,
    # n_threads=8,  # 可按物理核数设置
)
GEN_KW = dict(
    max_tokens=250,
    temperature=0.7,
    stop=None
)
OUT_DIR = r"C:\HWiNFO_logs\mtbench2"

# 关闭不稳定优化（按你之前的建议）
os.environ["LLAMA_CUDA_USE_GRAPHS"] = "0"
os.environ["GGML_CUDA_USE_VMM"] = "0"

# ========== 工具函数：提权/启动/停止 HWiNFO ==========
def _runas(exe_path: str, params: str, workdir: str):
    """以 UAC 管理员动词启动，不返回句柄"""
    ShellExecuteW = ctypes.windll.shell32.ShellExecuteW
    ShellExecuteW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR,
                              wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_int]
    ShellExecuteW.restype  = wintypes.HINSTANCE
    ret = ShellExecuteW(None, "runas", exe_path, params, workdir, 1)
    if ret <= 32:
        raise RuntimeError(f"ShellExecute(runAs) 失败: {ret}")

def _kill_hwinfo(graceful=True):
    """结束 HWiNFO；不 /F 为温和结束（有助于刷新 CSV）"""
    if graceful:
        subprocess.run(["taskkill","/IM","HWiNFO64.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["taskkill","/IM","HWiNFO64.exe","/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _wait_log_started(path: Path, timeout_s=36.0) -> bool:
    """等待日志文件开始写入（大小变化）"""
    t0 = time.time()
    last_size = -1
    while time.time() - t0 < timeout_s:
        if path.exists():
            sz = path.stat().st_size
            if sz > 0 and sz != last_size:
                return True
            last_size = sz
        time.sleep(0.1)
    return False

# ========== CSV 解析与能量计算 ==========
def _parse_timestamp(df: pd.DataFrame):
    """识别 Date/Time 或合并列，返回 Series[datetime64] 或 None"""
    lowers = {c.lower(): c for c in df.columns}
    date_col = lowers.get("date") or lowers.get("日期")
    time_col = lowers.get("time") or lowers.get("时间")
    combo = None
    for c in df.columns:
        cl = c.lower()
        if ("date" in cl and "time" in cl) or ("日期" in cl and "时间" in cl):
            combo = c; break
    if combo:
        ts = pd.to_datetime(df[combo], errors="coerce")
    elif date_col and time_col:
        ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                            errors="coerce", dayfirst=True)
        if ts.isna().all():
            ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                                errors="coerce", dayfirst=False)
    else:
        ts = None
    return ts

def _pick_col(cols, key_pairs):
    low=[c.lower() for c in cols]
    # 同时包含两个关键词优先；否则只要包含第二关键词
    for a,b in key_pairs:
        for i,c in enumerate(low):
            if a in c and b in c: return cols[i]
    for _,b in key_pairs:
        for i,c in enumerate(low):
            if b in c: return cols[i]
    return None

CPU_KEYS = [("cpu","package power"), ("cpu","封装功率")]
GPU_KEYS = [("gpu","power"), ("gpu","功率")]

def _trapz_cum(p: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)
    for i in range(1, len(p)):
        out[i] = out[i-1] + (p[i-1] + p[i]) * 0.5 * dt[i]
    return out

def _round_100ms(dt: datetime) -> datetime:
    """把时间点四舍五入到 100ms 边界"""
    # 100ms = 100_000 微秒
    us = dt.microsecond
    rem = us % 100_000
    if rem >= 50_000:
        dt = dt + timedelta(microseconds=(100_000 - rem))
    else:
        dt = dt - timedelta(microseconds=rem)
    # 规范为 100ms 粒度（避免 99999 微秒毛刺）
    us2 = (dt.microsecond // 100_000) * 100_000
    return dt.replace(microsecond=us2)

def sanitize(name: str) -> str:
    # Windows 文件名安全
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    name = re.sub(r"\s+", " ", name)
    return name[:80]

# ========== 主流程：单题测量 ==========
def run_one_question(llm: Llama, prompt_text: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "raw_hwinfo.csv"

    # 1) 温和清理旧进程 & 启动 HWiNFO 日志
    _kill_hwinfo(graceful=True)
    params = f'-log_format=1 -poll_rate={POLL_MS} -l"{str(raw_csv)}"'
    _runas(HWINFO_EXE, params, HWINFO_DIR)
    if not _wait_log_started(raw_csv, timeout_s=12):
        _kill_hwinfo(graceful=True)
        raise RuntimeError("HWiNFO 日志未开始写入，请检查权限/路径/传感器窗口设置。")

    # 2) 可选预热
    if WARMUP:
        try:
            llm("ping", max_tokens=1, temperature=0.0)
        except Exception:
            pass

    # 3) 推理窗口
    t_start = datetime.now()
    resp = llm(prompt_text, **GEN_KW)
    t_end   = datetime.now()
    t_start_r = _round_100ms(t_start)
    t_end_r   = _round_100ms(t_end)

    # 4) 结束 HWiNFO
    _kill_hwinfo(graceful=True)

    # 5) 读 CSV → 取 CPU/GPU 两列 → 截取窗口 → 积分
    df = pd.read_csv(raw_csv, engine="python", sep=None)
    ts = _parse_timestamp(df)
    cpu_col = _pick_col(list(df.columns), CPU_KEYS)
    gpu_col = _pick_col(list(df.columns), GPU_KEYS)
    if not cpu_col or not gpu_col:
        raise RuntimeError(f"找不到 CPU/GPU 功率列；示例列名：{list(df.columns)[:12]}")

    if ts is not None and ts.notna().any():
        df["_ts_"] = ts
        pad = pd.Timedelta(milliseconds=POLL_MS)
        win = df[(df["_ts_"] >= t_start - pad) & (df["_ts_"] <= t_end + pad)].copy()
        p_cpu = win[cpu_col].astype(float).to_numpy()
        p_gpu = win[gpu_col].astype(float).to_numpy()
        p_tot = p_cpu + p_gpu
        dt_s  = win["_ts_"].diff().dt.total_seconds().fillna(0.0).to_numpy()

        win["CPU_Package_W"] = p_cpu
        win["GPU_W"]         = p_gpu
        win["Total_Power_W"] = p_tot
        win["CPU_Energy_J"]  = _trapz_cum(p_cpu, dt_s)
        win["GPU_Energy_J"]  = _trapz_cum(p_gpu, dt_s)
        win["Energy_J"]      = _trapz_cum(p_tot, dt_s)

        if KEEP_TIME:
            win.rename(columns={"_ts_":"Timestamp"}, inplace=True)
        else:
            win.drop(columns=["_ts_"], inplace=True)
        out_df = win[["Timestamp","CPU_Package_W","GPU_W","Total_Power_W",
                      "CPU_Energy_J","GPU_Energy_J","Energy_J"]] if KEEP_TIME else \
                 win[["CPU_Package_W","GPU_W","Total_Power_W",
                      "CPU_Energy_J","GPU_Energy_J","Energy_J"]]
    else:
        # 没有可靠时间戳时退化为固定 dt
        dt = POLL_MS/1000.0
        p_cpu = df[cpu_col].astype(float).to_numpy()
        p_gpu = df[gpu_col].astype(float).to_numpy()
        p_tot = p_cpu + p_gpu
        def trapz_cum_fixed(p, dt):
            out = np.zeros_like(p, dtype=float)
            for i in range(1, len(p)):
                out[i] = out[i-1] + (p[i-1] + p[i]) * 0.5 * dt
            return out
        out_df = pd.DataFrame({
            "CPU_Package_W": p_cpu,
            "GPU_W": p_gpu,
            "Total_Power_W": p_tot,
            "CPU_Energy_J": trapz_cum_fixed(p_cpu, dt),
            "GPU_Energy_J": trapz_cum_fixed(p_gpu, dt),
            "Energy_J":     trapz_cum_fixed(p_tot, dt),
        })

    # 6) 汇总指标
    E_tot = float(out_df["Energy_J"].iloc[-1])
    E_cpu = float(out_df["CPU_Energy_J"].iloc[-1])
    E_gpu = float(out_df["GPU_Energy_J"].iloc[-1])
    avgP  = float(out_df["Total_Power_W"].mean())
    dur_s = (t_end - t_start).total_seconds()

    usage = resp.get("usage", {}) or {}
    prompt_tok     = usage.get("prompt_tokens")
    completion_tok = usage.get("completion_tokens")
    total_tok      = usage.get("total_tokens")
    if completion_tok is None and (total_tok is not None and prompt_tok is not None):
        completion_tok = total_tok - prompt_tok
    j_per_token = (E_tot / completion_tok) if completion_tok else None
    tok_per_s   = (completion_tok / dur_s) if (completion_tok and dur_s > 0) else None

    # 7) 输出文件
    out_df.to_csv(out_dir / "q_window_cpu_gpu.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "start_time": t_start.isoformat(),
            "end_time": t_end.isoformat(),
            "start_time_100ms": t_start_r.isoformat(),
            "end_time_100ms":   t_end_r.isoformat(),
            "duration_s": dur_s,
            "avg_total_power_W": avgP,
            "energy_total_J": E_tot,
            "energy_cpu_J": E_cpu,
            "energy_gpu_J": E_gpu,
            "completion_tokens": completion_tok,
            "tokens_per_s": tok_per_s,
            "J_per_token": j_per_token
        }, f, ensure_ascii=False, indent=2)

    return {
        "t_start": t_start, "t_end": t_end,
        "t_start_100ms": t_start_r, "t_end_100ms": t_end_r,
        "dur_s": dur_s, "avgP": avgP,
        "E_tot": E_tot, "E_cpu": E_cpu, "E_gpu": E_gpu,
        "comp_tokens": completion_tok, "tok_per_s": tok_per_s, "J_per_tok": j_per_token
    }

# ========== 主程序：遍历 MT-Bench ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="gguf 模型路径")
    ap.add_argument("--out",   type=str, default=OUT_DIR, help="输出根目录")
    ap.add_argument("--turn",  type=int, default=TURN_INDEX, help="使用 MT-Bench 的第几轮问题(从1开始)")
    ap.add_argument("--limit", type=int, default=80, help="最多测多少题(默认80)")
    args = ap.parse_args()

    out_root = Path(args.out)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"mtbench_{ts_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次运行配置
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model,
            "turn": args.turn,
            "limit": args.limit,
            "POLL_MS": POLL_MS,
            "KEEP_TIME": KEEP_TIME,
            "WARMUP": WARMUP
        }, f, ensure_ascii=False, indent=2)

    # 初始化 Llama
    llm = Llama(model_path=args.model, **LLAMA_KW)

    # 载入 MT-Bench（train split，含80问）
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    turn_idx0 = max(0, args.turn - 1)

    # 汇总表
    summary_rows = []

    for i, item in enumerate(ds):
        if i >= args.limit:
            break
        turns = item.get("turns") or []
        category = item.get("category") or "NA"
        q_text = turns[turn_idx0] if turn_idx0 < len(turns) else turns[0]
        short = sanitize(q_text.split("\n")[0]) or f"q{i+1}"
        q_dir = run_dir / f"q_{i+1:04d}_{sanitize(category)}_{short[:40]}"
        print(f"\n=== [{i+1}/{args.limit}] {category} ===")
        print(q_text[:200] + ("..." if len(q_text)>200 else ""))

        try:
            stats = run_one_question(llm, q_text, q_dir)
            summary_rows.append({
                "index": i+1,
                "category": category,
                "question_preview": q_text[:120],
                "start_100ms": stats["t_start_100ms"].isoformat(),
                "end_100ms":   stats["t_end_100ms"].isoformat(),
                "duration_s":  stats["dur_s"],
                "avg_total_power_W": stats["avgP"],
                "energy_total_J": stats["E_tot"],
                "energy_cpu_J": stats["E_cpu"],
                "energy_gpu_J": stats["E_gpu"],
                "completion_tokens": stats["comp_tokens"],
                "tokens_per_s": stats["tok_per_s"],
                "J_per_token":  stats["J_per_tok"],
            })
        except Exception as e:
            print(f"[跳过] 第{i+1}题失败：{e}")
            summary_rows.append({
                "index": i+1, "category": category,
                "question_preview": q_text[:120], "error": str(e)
            })

    # 写入汇总 CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n全部完成。输出目录：{run_dir}")

if __name__ == "__main__":
    main()

