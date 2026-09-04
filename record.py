"""Transmission record management.

Generate a unique ID for each test and save key parameters and results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import uuid

import config


def generate_run_id() -> str:
    """Generate a unique test ID: timestamp + short UUID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    return f"{ts}_{short_uid}"


def save_record(run_id: str,
                record: Dict[str, Any],
                record_dir: Path = config.RECORD_DIR) -> Path:
    """Save the test record as JSON and a text summary.

    Args:
        run_id: unique ID for this test
        record: record content dict
        record_dir: directory to save records

    Returns:
        JSON file path
    """
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    json_path = record_dir / f"record_{run_id}.json"
    txt_path = record_dir / f"record_{run_id}.txt"

    # Add metadata
    full_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    full_record.update(record)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_record, f, indent=2, ensure_ascii=False, default=str)

    # Text summary
    lines = [
        f"Run ID: {run_id}",
        f"Timestamp: {full_record['timestamp']}",
        f"Pilot Pattern: {full_record.get('pilot_pattern', 'N/A')}",
        f"Virtual Channel: {full_record.get('use_virtual_channel', False)}",
        f"  FC={full_record.get('virtual_channel_fc', 'N/A')} Hz",
        f"  SNR={full_record.get('virtual_channel_snr_db', 'N/A')} dB",
        f"  Nonlinearity={full_record.get('virtual_channel_nonlinearity', 'N/A')}",
        f"  Delay={full_record.get('virtual_channel_delay', 'N/A')}",
        f"  Attenuation={full_record.get('virtual_channel_attenuation', 'N/A')}",
        f"Estimated Rate (from QPSK probing): {full_record.get('estimated_rate_gbps', 'N/A')} Gbps",
        f"Final Rate (bitloading): {full_record.get('final_rate_gbps', 'N/A')} Gbps",
        f"Final BER: {full_record.get('final_ber', 'N/A')}",
        f"Final SER: {full_record.get('final_ser', 'N/A')}",
        f"Mean Recovered SNR (dB): {full_record.get('mean_recovered_snr_db', 'N/A')}",
        f"Ratio: {full_record.get('ratio', 'N/A')}",
        f"Use NN: {full_record.get('use_nn', False)}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path
