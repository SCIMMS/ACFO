# ACFO NCS v14 WAXS detector n_phi=2250 보충 실행

## 목적

기존 외부 full run의 12개 성공 결과를 그대로 보존하고, frozen validator와 실행 조건이 달랐던 WAXS detector 단계만 `nphi_min=2250`으로 다시 실행한다. 원본 run directory는 수정하지 않으며, 모든 기존 산출물을 복사한 amended run directory와 새로운 return ZIP을 만든다.

## 배치

보충 ZIP의 폴더를 기존 `ACFO_NCS_validation_release_candidate_v14` release root 바로 아래에 둔다. 기존 `.venv`, `scripts`, `benchmark_results`가 유지되어 있어야 한다.

## 한 줄 실행

기존 release root에서 다음을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File ACFO_NCS_v14_waxs_detector_nphi2250_supplement\run_waxs_detector_only.ps1 -OriginalRunDir "benchmark_results\external_acfo_ncs_v14_20260714T091255Z_full"
```

실제 계산 없이 명령과 출력 위치만 확인하려면 끝에 `-DryRun`을 붙인다.

## 결과

다음 두 항목이 생성된다.

- `benchmark_results/external_acfo_ncs_v14_20260714T091255Z_full_waxs_detector_nphi2250_amended/`
- 같은 이름 뒤에 `_return_package.zip`이 붙은 반환 ZIP

amended 폴더는 원본 `n_phi=2160` detector 결과, 원래 validation/receipt/steps 파일, 새 `n_phi=2250` 결과, 변경 이유와 양쪽 SHA-256을 모두 보존한다. WAXS detector 외 계산은 다시 실행하지 않는다.
