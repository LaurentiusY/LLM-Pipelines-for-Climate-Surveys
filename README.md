# LLM Pipelines for Climate Surveys

This repository contains the code accompanying the paper:

> Li, Z., & Ji, F. (2026). *When simulations look right but causal effects go wrong: Large language models as behavioral simulators.* arXiv:2604.02458. https://doi.org/10.48550/arXiv.2604.02458

[![arXiv](https://img.shields.io/badge/arXiv-2604.02458-b31b1b.svg)](https://arxiv.org/abs/2604.02458)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Large language models (LLMs) allow researchers to describe a target population and an intervention in natural language and then simulate how people would respond. This project examines how far such simulations can be trusted to recover intervention effects, rather than only reproducing the overall pattern of responses.

The study evaluates three LLMs on 11 climate-psychology interventions, using data from 59,508 participants across 62 countries, and replicates the main analysis on two additional datasets covering 12 and 27 countries. The central finding is a divergence between descriptive fit and causal accuracy: LLMs can approximate observed attitudinal patterns reasonably well while still misestimating the effects of interventions, particularly for behavioral outcomes and for interventions that rely on evoking internal experience. Good descriptive fit alone is therefore not sufficient evidence that a simulation yields valid conclusions about intervention effects.

This repository provides the LLM pipelines used to run the simulations for two climate surveys.

## Repository Structure

```
llm-for-climate-surveys/
├── climate-belief-survey-llm/     # LLM pipeline for the climate belief survey
├── climate-misinfo-survey-llm/    # LLM pipeline for the climate misinformation survey
├── LICENSE
└── README.md
```

## Getting Started

Each sub-project is self-contained. Please refer to the `pipeline_walkthrough.md` file within each project directory for setup instructions, execution steps, and implementation details:

- [`climate-belief-survey-llm/pipeline_walkthrough.md`](climate-belief-survey-llm/pipeline_walkthrough.md)
- [`climate-misinfo-survey-llm/pipeline_walkthrough.md`](climate-misinfo-survey-llm/pipeline_walkthrough.md)

## Citation

If you use this code in your research, please cite the paper:

**BibTeX**

```bibtex
@misc{li2026simulations,
  title         = {When simulations look right but causal effects go wrong: Large language models as behavioral simulators},
  author        = {Li, Zonghan and Ji, Feng},
  year          = {2026},
  eprint        = {2604.02458},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CY},
  doi           = {10.48550/arXiv.2604.02458},
  url           = {https://arxiv.org/abs/2604.02458}
}
```

**APA**

Li, Z., & Ji, F. (2026). *When simulations look right but causal effects go wrong: Large language models as behavioral simulators.* arXiv:2604.02458. https://doi.org/10.48550/arXiv.2604.02458

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions about the paper, please contact the corresponding author, Feng Ji (f.ji@utoronto.ca), Department of Applied Psychology and Human Development, University of Toronto. For issues related to the code, please open an issue in this repository.
