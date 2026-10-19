# Results

Exact match = every one of the 9 fields correct for a transaction.

| System | Seen formats & merchants: exact match | field accuracy | Unseen formats or merchants: exact match | field accuracy | Valid JSON | Median latency |
|---|---|---|---|---|---|---|
| Regex + merchant dictionary | **80.0%** | 97.2% | **31.8%** | 88.0% | 100.0% | – |

## Field accuracy: Seen formats & merchants

| Field | Regex + merchant dictionary |
|---|---|
| direction | 97.0% |
| amount | 99.8% |
| channel | 91.2% |
| counterparty | 94.8% |
| category | 93.8% |
| reference | 100.0% |
| account_last4 | 98.0% |
| balance | 99.8% |
| date | 100.0% |

## Field accuracy: Unseen formats or merchants

| Field | Regex + merchant dictionary |
|---|---|
| direction | 95.4% |
| amount | 100.0% |
| channel | 96.2% |
| counterparty | 53.4% |
| category | 49.0% |
| reference | 100.0% |
| account_last4 | 98.2% |
| balance | 100.0% |
| date | 100.0% |
