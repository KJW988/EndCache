# Third-Party Notices

EndCache does not vendor the upstream repositories. It provides only minimal patches for the pinned revisions below. The licenses and terms of each upstream project continue to apply to its source code, model weights, and datasets.

| Component | Repository / revision | License | Influence on this repository |
|---|---|---|---|
| Diffusion Policy | https://github.com/real-stanford/diffusion_policy @ `5ba07ac6661db573af695b419a7947ecb704690f` | MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab | Integration points in the DDPM sampling loop and the epsilon-to-endpoint conversion |
| openpi | https://github.com/Physical-Intelligence/openpi @ `650c5b0283a49c42784fb5055a0507da2c6d347d` | Apache-2.0 | Velocity-hold integration point in the π0.5 Euler action-generation loop |
| Isaac-GR00T N1.7 | https://github.com/NVIDIA/Isaac-GR00T @ `3df8b3825d67f755e69141446f4315f281b9b7e6` | Apache-2.0 | Integration points in the GR00T-N1.7 action-head Euler loop and evaluation client |
| RDT | https://github.com/thu-ml/RoboticsDiffusionTransformer @ `cd79363a1387e8f81c7724d070ef7e45fd23150f` | MIT, Copyright (c) 2024 TSAIL group | Integration points in the DPM-Solver++ sample-prediction loop and ManiSkill evaluator |
| Isaac-GR00T N1.6 | https://github.com/NVIDIA/Isaac-GR00T @ `ead52833afbbf4243f8cd5e7664f48a94de03b19` | NVIDIA License | Integration points in the GR00T-N1.6 action-head Euler loop and RoboCasa evaluator |

The full text of the Apache License 2.0 is available at https://www.apache.org/licenses/LICENSE-2.0. For Diffusion Policy and RDT, refer to the `LICENSE` file in each pinned revision for the full MIT License text.

## GR00T-N1.6 Notice

Section 3.3 of the NVIDIA License in the N1.6 revision restricts use to non-commercial research and excludes military, surveillance, service of nuclear technology, and biometric processing purposes. Review the complete upstream license before using or redistributing the N1.6 patch. A copy is included at `third_party/licenses/NVIDIA-GR00T-N1.6.txt` for convenience.

## Project-Level License

The authors have not yet selected a top-level license for the newly written EndCache core. Therefore, this repository does not currently include a separate `LICENSE` file. Any project-level license selected later will not replace the terms that apply to upstream code, model weights, or datasets.
