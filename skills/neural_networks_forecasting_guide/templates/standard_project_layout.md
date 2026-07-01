# Standard Project Layout

Unless requested otherwise, generate the following project.

project/

├── config.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── transforms.py
│
├── models/
│   ├── model.py
│   ├── blocks.py
│   └── layers.py
│
├── trainers/
│   └── trainer.py
│
├── metrics/
│   └── metrics.py
│
├── utils/
│   ├── checkpoint.py
│   ├── plotting.py
│   ├── seed.py
│   └── logging.py
│
└── outputs/
    ├── checkpoints/
    ├── predictions/
    └── figures/

Keep responsibilities separated.

Avoid monolithic scripts.

Prefer reusable modules.