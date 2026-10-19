# bank-txn-parser-qlora

**A 1.5B-parameter model fine-tuned with QLoRA on a free GPU to turn Indian bank SMS alerts and statement lines into clean JSON.** Covers UPI, IMPS, NEFT, RTGS, card, ATM, NACH, cheque and bank entries.

> Status: training run in progress on Kaggle. Results, model link and demo will be added here.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Tanzil-Ali-Rizvi/bank-txn-parser-qlora/blob/main/notebooks/kaggle_train.ipynb)

## Baseline to beat

A hand-written regex parser with a dictionary of every merchant in the training set:

| | Seen formats & merchants | Unseen formats or merchants |
|---|---|---|
| Exact match (all 9 fields right) | 80.0% | 28.0% |

## Quickstart

```bash
pip install -e ".[dev]"
python -m txnparser.generate --out data      # regenerate the dataset (seeded, byte-identical)
python -m txnparser.rules                     # regex baseline
python -m txnparser.report                    # comparison table -> results/RESULTS.md
```

Training and evaluation run end to end in [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) on a free Kaggle T4.

## License

Apache-2.0 © 2026 Tanzil Ali Rizvi
