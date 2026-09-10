from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "artifacts" / "iris_model.joblib"

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
model = artifact["model"]
target_names = artifact["target_names"]

st.set_page_config(page_title="붓꽃 분류기", page_icon="🌸")
st.title("🌸 붓꽃 품종 분류기")
st.write("측정값을 입력하면 품종별 확률을 예측합니다.")
st.caption(f"저장 모델 테스트 정확도:{artifact['test_accuracy']:.3f}")

with st.form("prediction"):
    sl = st.number_input("꽃받침 길이(cm)", 0.1, 10.0, 5.1, 0.1)
    sw = st.number_input("꽃받침 너비(cm)", 0.1, 10.0, 3.5, 0.1)
    pl = st.number_input("꽃잎 길이(cm)", 0.1, 10.0, 1.4, 0.1)
    pw = st.number_input("꽃잎 너비(cm)", 0.1, 10.0, 0.2, 0.1)
    submitted = st.form_submit_button("품종 예측")

if submitted:
    probabilities = model.predict_proba([[sl, sw, pl, pw]])[0]
    predicted_index = int(probabilities.argmax())
    st.success(f"예측 품종:{target_names[predicted_index]}")

    result = pd.DataFrame({
        "품종": target_names,
        "예측 확률": probabilities,
    }).sort_values("예측 확률", ascending=False)
    st.dataframe(result.style.format({"예측 확률": "{:.2%}"}), hide_index=True)
    st.bar_chart(result.set_index("품종"))

with st.expander("모델 정보"):
    st.write("표준화 + 로지스틱 회귀")
    st.warning("교육용 예제이며 실제 생물학적 판정에 사용하지 않습니다.")