# PAITA: Performance-Adaptive Instruction-Template Assignment

Code accompanying the CIKM 2026 paper "Performance-Adaptive
Instruction-Template Assignment for Clinical Reasoning in Large Language
Models."

PAITA formulates instruction-template assignment during fine-tuning as a
training-time corpus-construction problem, using a UCB bandit to balance
exploration of under-sampled templates with exploitation of templates
associated with stronger validation performance. This repository provides
all three compared assignment strategies: Static, Random, and PAITA.

## Layout

```
PAITA/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── cikm2026.yaml
├── src/
│   ├── train.py
│   ├── paita.py
│   ├── prompt_managers.py
│   ├── templates.py
│   ├── data.py
│   └── evaluation.py
└── scripts/
    ├── run_static.sh
    ├── run_random.sh
    └── run_paita.sh
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py \
    --strategy paita \
    --model meta-llama/Meta-Llama-3-8B \
    --data_path /path/to/train.csv \
    --epochs 4 \
    --output_dir outputs/paita
```

`--data_path` expects a CSV with `task`, `input`, and `output` columns.
`task` values must be one of `Clinical Diagnosis`, `Clinical Reasoning`,
or `Medical Question Answering`. See `configs/cikm2026.yaml` for the full
hyperparameter set and `scripts/` for ready-to-edit run commands.

## Strategies

- **Static** — a single fixed template (T0) per task; no exploration.
- **Random** — a template is sampled uniformly at random from the
  ten-template pool each time a training instance is encountered;
  assignments are continually refreshed across epochs. Serves as the
  PAFT-style exploration-only baseline.
- **PAITA** — a UCB bandit assigns templates using
  `i* = argmax_i (s_i + c * sqrt(ln(t) / u_i))`. `c = 1.0`, exponential
  smoothing rate `0.15`, minimum warm-up of 20 assignments per template.

## Evaluation template attribution

Evaluation examples are pre-assigned a `template_id` (round-robin, see
`create_eval_dataset_with_template_ids` in `data.py`). The collator
formats these deterministically with exactly that template
(`format_with_template_id`), so a prediction's recorded `template_id`
always matches the template that generated it.

## Known limitations

- Evaluation is memory-intensive, so a small number of examples were used
  for validation during training..


## Citation

If you use this code, please cite the CIKM 2026 paper:

Henry Mukalazi Serugunda, Ouyang Jianquan, and Jacob Katende. 2026.
Performance-Adaptive Instruction-Template Assignment for Clinical Reasoning
in Large Language Models. In *Proceedings of the 35th ACM International
Conference on Information and Knowledge Management (CIKM '26)*,
November 7–11, 2026, Rome, Italy. ACM, New York, NY, USA, 3760–3769.
https://doi.org/10.1145/3799682.3840978

```bibtex
@inproceedings{serugunda2026paita,
  author    = {Serugunda, Henry Mukalazi and Ouyang, Jianquan and Katende, Jacob},
  title     = {Performance-Adaptive Instruction-Template Assignment for Clinical Reasoning in Large Language Models},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  pages     = {3760--3769},
  location  = {Rome, Italy},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3799682.3840978},
  url       = {https://doi.org/10.1145/3799682.3840978}
}
```
