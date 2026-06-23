# Curved Ewald factorization strategy source notes

Generated PDF: `benchmark_results\curved_ewald_factorization_strategy_ko.pdf`
Generated on: 2026-06-22

## Main strategy

- Main novelty: prepared forward/adjoint operatorization for rotationally structured curved Ewald/cap Fourier evaluation.
- Main evidence: WAXS for correctness/scaling validation; ODT for repeated inverse/backpropagation impact.
- High-NA role: prior-art-connected bridge, recovering known circular-harmonic/Fourier-Bessel reduction; adaptive/vectorial/GPU implementation as secondary practical demo.
- Claim boundary: circular harmonic identity itself is not novel; WAXS/ODT geometry-aware operatorization is the novelty.

## Web sources checked in this turn

- Boichenko 2022/2023: https://arxiv.org/abs/2212.10978 -- High-NA Richards-Wolf circularly polarized vortex-mode reduction; known precedent, not our novelty.
- Kirisits et al. 2024/2025: https://arxiv.org/abs/2407.01793 -- Generalized Fourier diffraction theorem and filtered backpropagation for diffraction tomography.
- Chen et al. 2021: https://arxiv.org/abs/2101.11709 -- Curved Ewald sphere problem in electron-microscopy reconstruction.
- Horstmeyer and Yang 2015: https://arxiv.org/abs/1510.08756 -- Fourier ptychographic diffraction tomography geometry and iterative reconstruction baseline.
- Zuo et al. 2019: https://arxiv.org/abs/1904.09386 -- High-throughput Fourier ptychographic diffraction tomography application context.
- Pratley et al. 2018/2019: https://arxiv.org/abs/1807.09239 -- Adjacent radio-interferometry example of radial/Hankel structure for wide-field correction.
- Zhao et al. 2014/2015: https://arxiv.org/abs/1412.0781 -- Adjacent Fourier-Bessel basis and NUFFT use in cryo-EM image analysis.

## Local context used

- Existing benchmark directories and previous PDF/report scripts were inspected for output convention.
- The new PDF is a strategy memo, not a fresh benchmark rerun.
