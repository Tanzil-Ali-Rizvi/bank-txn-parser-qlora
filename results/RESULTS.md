# Results

Exact match = every one of the 9 fields correct for a transaction.

| System | Seen formats & merchants: exact match | field accuracy | Unseen formats or merchants: exact match | field accuracy | Valid JSON | Median latency |
|---|---|---|---|---|---|---|
| Regex + merchant dictionary | **80.0%** | 97.2% | **31.8%** | 88.0% | 100.0% | – |
| Qwen2.5-1.5B-Instruct, 4-shot prompt | **17.0%** | 78.9% | **17.0%** | 78.0% | 100.0% | 687 ms |
| Qwen2.5-1.5B + QLoRA (this repo) | **98.2%** | 99.8% | **72.2%** | 95.5% | 100.0% | 140 ms |

## Field accuracy: Seen formats & merchants

| Field | Regex + merchant dictionary | Qwen2.5-1.5B-Instruct, 4-shot prompt | Qwen2.5-1.5B + QLoRA (this repo) |
|---|---|---|---|
| direction | 97.0% | 83.8% | 100.0% |
| amount | 99.8% | 92.4% | 100.0% |
| channel | 91.2% | 78.2% | 99.8% |
| counterparty | 94.8% | 71.2% | 99.0% |
| category | 93.8% | 48.8% | 99.8% |
| reference | 100.0% | 71.4% | 100.0% |
| account_last4 | 98.0% | 79.0% | 100.0% |
| balance | 99.8% | 85.8% | 100.0% |
| date | 100.0% | 99.8% | 99.4% |

## Field accuracy: Unseen formats or merchants

| Field | Regex + merchant dictionary | Qwen2.5-1.5B-Instruct, 4-shot prompt | Qwen2.5-1.5B + QLoRA (this repo) |
|---|---|---|---|
| direction | 95.4% | 83.4% | 88.2% |
| amount | 100.0% | 95.8% | 100.0% |
| channel | 96.2% | 73.6% | 92.6% |
| counterparty | 53.4% | 70.6% | 92.8% |
| category | 49.0% | 42.6% | 89.4% |
| reference | 100.0% | 68.0% | 98.4% |
| account_last4 | 98.2% | 78.8% | 99.0% |
| balance | 100.0% | 90.2% | 100.0% |
| date | 100.0% | 99.0% | 99.0% |
