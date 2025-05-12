# multimodal_ag

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This project focuses on identifying toxic content in online comments. It uses automated machine learning techniques to predict whether a given comment is harmful, aiming to help create safer and more positive online environments by flagging potentially offensive language.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Citation](#citation)

---
## Core features

1. **Multimodal Data Handling**: The project demonstrates the use of AutoGluon for handling multimodal data, specifically combining tabular and text features to predict toxicity in online comments.
2. **AutoGluon Integration**: Leverages the AutoGluon library (version 0.5.2) for automated machine learning, including model training and evaluation with specified hyperparameters and presets.
3. **Toxicity Prediction**: The core task is to predict whether online social media comments are toxic, utilizing a dataset from the Jigsaw Unintended Bias in Toxicity Classification Kaggle competition.
4. **F1 Metric Evaluation**: The model's performance is evaluated using the F1 metric, which measures the harmonic mean of precision and recall. The script prints and returns this value.
5. **Data Loading & Preprocessing**: The project includes functionality to load data from CSV files (train and test sets) related to the Jigsaw competition, using pandas for data manipulation.

---

## Installation

Install multimodal_ag using one of the following methods:

**Build from source:**

1. Clone the multimodal_ag repository:
```sh
git clone https://github.com/andreygetmanov/multimodal_ag
```

2. Navigate to the project directory:
```sh
cd multimodal_ag
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Contributing

- **[Report Issues](https://github.com/andreygetmanov/multimodal_ag/issues)**: Submit bugs found or log feature requests for the project.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    andreygetmanov (2022). multimodal_ag repository [Computer software]. https://github.com/andreygetmanov/multimodal_ag

### BibTeX format:

    @misc{multimodal_ag,

        author = {andreygetmanov},

        title = {multimodal_ag repository},

        year = {2022},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/andreygetmanov/multimodal_ag.git}},

        url = {https://github.com/andreygetmanov/multimodal_ag.git}

    }

---
