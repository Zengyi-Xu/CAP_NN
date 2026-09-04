"""传输记录管理。

为每次测试生成唯一 ID，并保存关键参数和结果。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import uuid

import config


def generate_run_id() -> str:
    """生成唯一的测试 ID：时间戳 + 短 UUID。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    return f"{ts}_{short_uid}"


def save_record(run_id: str,
                record: Dict[str, Any],
                record_dir: Path = config.RECORD_DIR) -> Path:
    """将测试记录保存为 JSON 和文本摘要。

    Args:
        run_id: 本次测试的唯一 ID
        record: 记录内容字典
        record_dir: 保存记录的目录

    Returns:
        JSON 文件路径
    """
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    json_path = record_dir / f"record_{run_id}.json"
    txt_path = record_dir / f"record_{run_id}.txt"

    # 添加元数据
    full_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    full_record.update(record)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_record, f, indent=2, ensure_ascii=False, default=str)

    # 文本摘要
    lines = [
        f"运行 ID: {run_id}",
        f"时间戳: {full_record['timestamp']}",
        f"导频图案: {full_record.get('pilot_pattern', 'N/A')}",
        f"虚拟信道: {full_record.get('use_virtual_channel', False)}",
        f"  截止频率={full_record.get('virtual_channel_fc', 'N/A')} Hz",
        f"  信噪比={full_record.get('virtual_channel_snr_db', 'N/A')} dB",
        f"  非线性={full_record.get('virtual_channel_nonlinearity', 'N/A')}",
        f"  时延={full_record.get('virtual_channel_delay', 'N/A')}",
        f"  衰减={full_record.get('virtual_channel_attenuation', 'N/A')}",
        f"估计速率（来自 QPSK 探测）: {full_record.get('estimated_rate_gbps', 'N/A')} Gbps",
        f"最终速率（比特加载）: {full_record.get('final_rate_gbps', 'N/A')} Gbps",
        f"最终误码率: {full_record.get('final_ber', 'N/A')}",
        f"最终误符号率: {full_record.get('final_ser', 'N/A')}",
        f"平均恢复信噪比 (dB): {full_record.get('mean_recovered_snr_db', 'N/A')}",
        f"比率: {full_record.get('ratio', 'N/A')}",
        f"使用神经网络: {full_record.get('use_nn', False)}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path
