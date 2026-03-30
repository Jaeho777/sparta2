# 유가 예측 제출 패키지 최종 점검

## 패키지 용도
- 이 폴더는 교수님께 바로 전달할 수 있도록 정리한 최종 패키지이다.
- 포함 항목은 `report/`, `ipynb/`, `csv/` 세 폴더이다.

## 제출 핵심 파일
- 보고서: [report/00_REPORT_oil_professor.md](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_final_package_oil_20260311/report/00_REPORT_oil_professor.md)
- 기준선 노트북: [ipynb/01_oil_nickel_style_hybrid.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_final_package_oil_20260311/ipynb/01_oil_nickel_style_hybrid.ipynb)
- Transformer 조합 탐색 노트북: [ipynb/02_oil_transformer_advanced.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_final_package_oil_20260311/ipynb/02_oil_transformer_advanced.ipynb)
- STL 단계 설명 노트북: [ipynb/03_oil_stl_residual_ror.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_final_package_oil_20260311/ipynb/03_oil_stl_residual_ror.ipynb)

## 최종 점검 결과
- 완료: 보고서가 최신 문장과 최신 수치 기준으로 갱신되어 있다.
- 완료: 노트북 3개가 모두 존재한다.
- 완료: 노트북 3개가 모두 실행본으로 저장되어 출력 셀이 포함되어 있다.
- 완료: 기준선 결과 CSV가 포함되어 있다.
- 완료: Transformer 실험 CSV가 포함되어 있다.
- 완료: STL 실험 CSV가 포함되어 있다.
- 완료: 추가 점검용 반복 평가 CSV가 포함되어 있다.
- 완료: 보고서와 노트북에서 `단일 분할 결과`와 `추가 점검용 반복 평가 결과`를 구분해서 설명한다.

## 현재 기준 핵심 수치
- 기준선 최고 성능: `Hybrid_TwoPointLinear0.7_GB0.3`, test RMSE `1.3440`
- Transformer 단일 분할 공식 선정: `PatchTST + Transformer + ElasticNet`, validation RMSE `1.2443`, test RMSE `2.3577`
- Transformer 단일 분할 test 최저: `PatchTST + iTransformer`, test RMSE `1.2308`
- STL 단일 분할 최종 구조: `Exponential Smoothing + NLinear + LightGBM`, test RMSE `1.2275`
- 추가 점검용 반복 평가 평균 test 최저: `1-step Random Walk`, mean test RMSE `2.2191`

## CSV 해석 순서
- 기준선: `csv/01_*`
- Transformer 조합 탐색: `csv/02_*`
- STL 단계 설명 실험: `csv/03_*`
- 추가 점검용 반복 평가: `csv/04_*`

## 제출 직전 확인 항목
- 보고서를 먼저 열어 상단 결론과 표가 정상 표시되는지 확인
- 노트북 3개가 오류 없이 열리는지 확인
- `02_oil_transformer_advanced.ipynb`에서 추가 점검용 반복 평가 표 출력이 보이는지 확인
- `03_oil_stl_residual_ror.ipynb`에서 추가 점검용 반복 평가 표 출력이 보이는지 확인
- 교수님이 코드 실행을 요청할 경우 `ipynb/` 폴더의 3개 노트북과 `csv/` 폴더를 함께 전달

## 해석상 주의사항
- `02_oil_transformer_advanced.ipynb`와 `03_oil_stl_residual_ror.ipynb`의 단일 분할 결과만으로 서로 다른 계열의 우열을 단정하지 않는다.
- 직접 비교는 `csv/04_common_protocol_rolling_origin_summary.csv` 기준으로 읽는다.
- 2차 잔차 보정은 일부 단일 분할에서 개선을 보였지만, 반복 검증 평균에서는 일관된 추가 개선으로 확인되지 않았다.
