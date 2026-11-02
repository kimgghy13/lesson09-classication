import pandas as pd
import plotly.express as px
import streamlit as st

# ※ 첫 화면(메인 파일)에서 쓰는 주소와 똑같이 맞춰 주세요.
DATA_URL = "https://example.com/healthcare-dataset-stroke-data.csv"

st.set_page_config(page_title="탐색", layout="wide")
st.title("탐색")


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


df = load_data(DATA_URL)

GROUP_LABEL = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
df["뇌졸중 여부"] = df["stroke"].map(GROUP_LABEL)

# 1. 나이와 평균 혈당 분포
st.header("나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포",
                       labels={"age": "나이", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포",
                       labels={"avg_glucose_level": "평균 혈당"})
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)

# 2. 뇌졸중 여부별 상자그림과 평균값
st.header("뇌졸중 여부에 따른 나이와 평균 혈당")
order = {"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]}
col1, col2 = st.columns(2)
with col1:
    fig = px.box(df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
                 category_orders=order, title="나이",
                 labels={"age": "나이"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
                 category_orders=order, title="평균 혈당",
                 labels={"avg_glucose_level": "평균 혈당"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당의 평균"})
)
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True)

# 3. 고혈압·심장병 유무별 뇌졸중 비율
st.header("고혈압·심장병 유무별 뇌졸중 비율")
col1, col2 = st.columns(2)
for col, (name, title) in zip(
    (col1, col2),
    (("hypertension", "고혈압"), ("heart_disease", "심장병")),
):
    rate = (
        df.groupby(name)["stroke"].mean().mul(100).round(2).reset_index()
    )
    rate["구분"] = rate[name].map({0: f"{title} 없음", 1: f"{title} 있음"})
    fig = px.bar(rate, x="구분", y="stroke", text="stroke",
                 title=f"{title} 유무별 뇌졸중 비율",
                 labels={"stroke": "뇌졸중 비율(%)", "구분": ""})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    with col:
        st.plotly_chart(fig, use_container_width=True)

# 4. 체질량지수(bmi) 결측자의 뇌졸중 비율
st.header("체질량지수(bmi)가 비어 있는 사람")
bmi_missing = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "사람 수": [len(bmi_missing), len(df)],
        "뇌졸중 사람 수": [int(bmi_missing["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [
            round(bmi_missing["stroke"].mean() * 100, 2) if len(bmi_missing) else float("nan"),
            round(df["stroke"].mean() * 100, 2),
        ],
    },
    index=["bmi가 빈 사람", "전체"],
)
st.dataframe(bmi_table, use_container_width=True)

# 5. 흡연 상태별 사람 수
st.header("흡연 상태별 사람 수")
smoking = (
    df["smoking_status"].value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, use_container_width=True, hide_index=True)
