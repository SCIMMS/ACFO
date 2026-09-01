# WAXS direct-NDFT reference sweep

ACFO와 FINUFFT를 동일한 binned source 및 curved-manifold target의 complex128 direct NDFT와 각각 비교했다.

| case | q (Å⁻¹) | curvature | bin (nm) | Nphi | ACFO prod | ACFO full | FINUFFT 1e-6 | FINUFFT tight | binned-vs-atom intensity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_q_physical | 0.05–1 | 1.00 | 0.1 | 180 | 1.766e-07 | 2.531e-15 | 6.661e-07 | 4.038e-11 | 4.272e-04 |
| mid_q_physical | 2–4 | 1.00 | 0.1 | 500 | 3.269e-07 | 2.675e-14 | 9.795e-07 | 5.138e-11 | 5.516e-01 |
| high_q_physical | 5–6.3 | 1.00 | 0.1 | 750 | 9.111e-07 | 6.694e-14 | 1.187e-06 | 5.412e-11 | 7.765e-01 |
| high_q_planar | 5–6.3 | 0.00 | 0.1 | 750 | 1.946e-07 | 6.568e-14 | 4.249e-07 | 3.865e-11 | — |
| high_q_half_curvature | 5–6.3 | 0.50 | 0.1 | 750 | 4.486e-07 | 6.733e-14 | 1.131e-06 | 5.918e-11 | — |
| high_q_angular_half | 5–6.3 | 1.00 | 0.1 | 384 | 2.616e-06 | 6.198e-14 | 1.154e-06 | 5.384e-11 | 7.688e-01 |
| high_q_angular_double | 5–6.3 | 1.00 | 0.1 | 1500 | 8.967e-07 | 6.219e-14 | 1.179e-06 | 5.383e-11 | 7.695e-01 |
| high_q_bin_0p05nm | 5–6.3 | 1.00 | 0.05 | 750 | 8.029e-07 | 6.360e-14 | 1.135e-06 | 5.058e-11 | 5.085e-01 |
| high_q_bin_0p025nm | 5–6.3 | 1.00 | 0.025 | 750 | 7.991e-07 | 6.283e-14 | 1.088e-06 | 5.030e-11 | 3.121e-01 |
| high_q_bin_0p0125nm | 5–6.3 | 1.00 | 0.0125 | 750 | 8.140e-07 | 6.312e-14 | 1.119e-06 | 5.003e-11 | 2.334e-01 |
| high_q_bin_0p00625nm | 5–6.3 | 1.00 | 0.00625 | 750 | 8.513e-07 | 6.293e-14 | 1.101e-06 | 4.999e-11 | 2.108e-01 |
| high_q_bin_0p003125nm | 5–6.3 | 1.00 | 0.003125 | 750 | 8.699e-07 | 6.311e-14 | 1.094e-06 | 4.984e-11 | 2.031e-01 |
| high_q_bin_0p0015625nm | 5–6.3 | 1.00 | 0.0015625 | 750 | 8.516e-07 | 6.267e-14 | 1.093e-06 | 4.987e-11 | 2.017e-01 |
| high_q_bin_0p00078125nm | 5–6.3 | 1.00 | 0.00078125 | 750 | 8.686e-07 | 6.280e-14 | 1.092e-06 | 4.992e-11 | 2.017e-01 |

- operator/NUFFT gates: **PASS**
- Direct FFT는 curved nonuniform target의 동일 연산자가 아니므로 oracle로 사용하지 않았다.
- exact-atom 열은 operator 오차가 아니라 cylindrical source discretization까지 포함한 end-to-end 오차다.
- 시간은 이 소규모 correctness 실행의 부수 기록이며 production 성능 비교에 사용하지 않는다.
