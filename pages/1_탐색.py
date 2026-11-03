import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("탐색")


def find_main_url():
    """첫 화면(pages 폴더 밖의 .py 파일)에서 데이터 주소를 찾는다."""
    root = Path(__file__).resolve().parent.parent
    found = []
    for path in sorted(root.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found += re.findall(r"https?://[^\s\"'()]+", text)
    csv_urls = [u for u in found if ".csv" in u.lower()]
    if csv_urls:
        return csv_urls[0]
    return found[0] if found else None


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


url = find_main_url()
if url is None:
    url = st.text_input("첫 화면에서 쓰는 데이터 주소(CSV)를 붙여 넣어 주세요.").strip()
    if not url:
        st.info("주소를 입력하면 아래에 분석 결과가 나타나요.")
        st.stop()
st.caption(f"데이터 주소: {url}")

try:
    df = load_data(url)
except Exception as e:
    st.error(f"데이터를 읽지 못했어요.\n\n주소: {url}\n\n오류: {e}")
    st.stop()

needed = ["age", "avg_glucose_level", "hypertension", "heart_disease",
          "bmi", "smoking_status", "stroke"]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    st.error(f"필요한 열이 없어요: {missing_cols}\n\n현재 열: {list(df.columns)}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})
order = {"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]}

# 1. 나이와 평균 혈당 분포
st.header("나이와 평균 혈당의 분포")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(fig)
with c2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포")
    fig.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수")
    st.plotly_chart(fig)

# 2. 뇌졸중 여부별 상자그림과 평균값
st.header("뇌졸중 여부에 따른 나이와 평균 혈당")
c1, c2 = st.columns(2)
with c1:
    fig = px.box(df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
                 category_orders=order, title="나이")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="나이")
    st.plotly_chart(fig)
with c2:
    fig = px.box(df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
                 category_orders=order, title="평균 혈당")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="평균 혈당")
    st.plotly_chart(fig)

mean_table = df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]].mean().round(2)
mean_table = mean_table.reindex(["뇌졸중 없음", "뇌졸중 있음"])
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table)

# 3. 고혈압·심장병 유무별 뇌졸중 비율
st.header("고혈압·심장병 유무별 뇌졸중 비율")
c1, c2 = st.columns(2)
for col, (name, label) in zip([c1, c2], [("hypertension", "고혈압"),
                                         ("heart_disease", "심장병")]):
    rate = df.groupby(name)["stroke"].mean().mul(100).round(2).reset_index()
    rate["구분"] = rate[name].map({0: f"{label} 없음", 1: f"{label} 있음"})
    rate = rate.rename(columns={"stroke": "뇌졸중 비율(%)"})
    fig = px.bar(rate, x="구분", y="뇌졸중 비율(%)", text="뇌졸중 비율(%)",
                 title=f"{label} 유무별 뇌졸중 비율")
    fig.update_layout(xaxis_title="")
    fig.update_traces(textposition="outside")
    with col:
        st.plotly_chart(fig)

# 4. 체질량지수(bmi) 결측자의 뇌졸중 비율
st.header("체질량지수(bmi)가 비어 있는 사람")
bm = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "사람 수": [len(bm), len(df)],
        "뇌졸중 사람 수": [int(bm["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [
            round(float(bm["stroke"].mean()) * 100, 2) if len(bm) else None,
            round(float(df["stroke"].mean()) * 100, 2),
        ],
    },
    index=["bmi가 빈 사람", "전체"],
)
st.dataframe(bmi_table)

# 5. 흡연 상태별 사람 수
st.header("흡연 상태별 사람 수")
smoking = df["smoking_status"].fillna("(비어 있음)").value_counts().reset_index()
smoking.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking)
