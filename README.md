---
title: Iris Streamlit Classifier
emoji: 🌸
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# 붓꽃 품종 분류기

scikit-learn Iris 데이터셋으로 학습한 로지스틱 회귀 모델을
Streamlit으로 제공하는 교육용 프로젝트입니다.

## 입력과 출력

-입력: 꽃받침·꽃잎의 길이와 너비(cm)
-출력: 예측 품종과 품종별 확률

## 모델과 데이터

-모델: StandardScaler + LogisticRegression Pipeline
-데이터: scikit-learn Iris dataset
-성능: 앱에 저장된 테스트 정확도 표시

## 실행

streamlit run app.py

## 한계

교육용 예제이며 실제 생물학적 판정에 사용하지 않습니다.

## 라이선스

배포 전에 코드·데이터·모델에 적용할 라이선스를 확인하고 기재합니다.