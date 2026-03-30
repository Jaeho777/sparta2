# 유가 예측 제출용 폴더 점검

## 폴더 용도
- 이 폴더는 `ipynb/`와 `csv/`만 담은 제출용 폴더이다.
- 보고서까지 포함한 최종 패키지는 `professor_final_package_oil_20260311` 폴더를 사용한다.

## 포함 파일
- 기준선 노트북: [ipynb/01_oil_nickel_style_hybrid.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_submission_oil_20260311/ipynb/01_oil_nickel_style_hybrid.ipynb)
- Transformer 조합 탐색 노트북: [ipynb/02_oil_transformer_advanced.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_submission_oil_20260311/ipynb/02_oil_transformer_advanced.ipynb)
- STL 단계 설명 노트북: [ipynb/03_oil_stl_residual_ror.ipynb](/Users/jaeholee/Desktop/T-LAB/sparta_2/sparta2/professor_submission_oil_20260311/ipynb/03_oil_stl_residual_ror.ipynb)

## 점검 결과
- 완료: 노트북 3개가 모두 존재한다.
- 완료: 노트북 3개가 모두 실행본으로 저장되어 있다.
- 완료: 기준선, Transformer, STL, 추가 점검용 반복 평가 CSV가 모두 포함되어 있다.
- 완료: 노트북 2와 3에는 추가 점검용 반복 평가 결과 출력 셀이 포함되어 있다.

## 핵심 수치
- 기준선 최고 성능: `Hybrid_TwoPointLinear0.7_GB0.3`, test RMSE `1.3440`
- Transformer 단일 분할 test 최저: `PatchTST + iTransformer`, test RMSE `1.2308`
- STL 단일 분할 최종 구조: `Exponential Smoothing + NLinear + LightGBM`, test RMSE `1.2275`
- 추가 점검용 반복 평가 평균 test 최저: `1-step Random Walk`, mean test RMSE `2.2191`

## 전달 시 주의사항
- 보고서까지 함께 내야 하면 `professor_final_package_oil_20260311` 폴더를 전달한다.
- 코드와 결과만 요청받으면 이 폴더를 전달하면 된다.
