# ACFO NCS detector-aware WAXS 검증

## 판정

**정확도·메모리 gate PASS. Nq=512 local publication timing protocol PASS; 외부 머신 반복은 남아 있다.**

15.5 keV, 100 mm 거리의 EIGER2 X 4M rectangular active envelope에서 detector에 들어오는 partial-arc node만 FINUFFT가 계산하도록 비교했다. ACFO 시간에는 현재 구현이 계산하는 full ring 전체가 포함된다.

| Nq | dq (A^-1) | active/full targets | ACFO s | FINUFFT s | speedup | complex L2 | memory ratio | repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.02451 | 507,298/576,000 | 13.18 (cached) | 31.34 (cached) | 2.38x | 5.831e-07 | 5.37x | 2 |
| 512 | 0.01223 | 1,015,326/1,152,000 | 21.60 (cached) | 42.69 (cached) | 1.98x | 7.028e-07 | 5.14x | 30 |
| 1024 | 0.00611 | 2,031,482/2,304,000 | 35.20 (first) | 124.89 (first) | 3.55x | 7.860e-07 | 4.69x | 0 |

## Nq=512 publication protocol

- 10 warm-up + 30 measured calls per method
- AB/BA alternating order; first calls and memory profiling are separate
- ratio-of-medians speedup: 1.976x
- paired speedup median/p05/p95: 2.001x / 1.918x / 2.371x
- ACFO/FINUFFT coefficient of variation: 5.7% / 10.5%
- 이전 1-repeat 3.47x 값은 superseded하며 local claim은 약 2x로 제한한다.

## 제한

현재 mask는 detector 외곽 rectangle만 반영하며 module gap, bad-pixel mask, beamstop은 포함하지 않는다. Nq=256은 cached 2회, Nq=1024는 first-run probe이므로 Nq=512만 local publication protocol을 충족한다. 외부 머신 반복은 여전히 필요하다.

## 재현

```powershell
.\.venv\Scripts\python.exe scripts\summarize_waxs_detector_aware.py
```
