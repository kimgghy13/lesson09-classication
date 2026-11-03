import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title("채점표")
st.caption("stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중을 양성으로 두었어요. "
           "입력 속성은 '분류 모델' 페이지에서 처음 골라져 있는 나이·평균 혈당·고혈압·심장병 넷이에요.")

# '분류 모델' 페이지에서 처음 골라져 있는 속성 (체질량지수는 뺀 넷)
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
NO_VALUE = "계산할 수 없음"
MAX_ROWS = 20

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"


# ---------------------------------------------------------------- 데이터 읽기
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
        st.info("주소를 입력하면 아래에 결과가 나타나요.")
        st.stop()

try:
    df = load_data(url)
except Exception as e:
    st.error(f"데이터를 읽지 못했어요.\n\n주소: {url}\n\n오류: {e}")
    st.stop()

needed = ["id", "stroke"] + FEATURES
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    st.error(f"필요한 열이 없어요: {missing_cols}\n\n현재 열: {list(df.columns)}")
    st.stop()

# ---------------------------------------------------------------- 데이터 나누기
# 번호 순으로 정렬 → 열 명씩 묶음 → 각 묶음의 앞 세 명은 테스트용
df = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3

train_df = df.loc[~is_test].reset_index(drop=True)
test_df = df.loc[is_test].reset_index(drop=True)

X_train = train_df[FEATURES].astype(float)
X_test = test_df[FEATURES].astype(float)
y_train = train_df["stroke"].astype(int).to_numpy()
y_test = test_df["stroke"].astype(int).to_numpy()

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("입력 속성에 비어 있는 값이 있어 모델을 만들 수 없어요.")
    st.stop()

# ---------------------------------------------------------------- 모델 만들기
# 로지스틱 회귀: 크기 맞추기(표준화)를 훈련용으로만 배워서 적용
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logit.fit(X_train, y_train)

# 의사결정트리: 질문 세 번까지, 사람이 다섯 명 미만인 마디는 더 나누지 않음
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)

# 기준 모델: 입력을 보지 않고 훈련용에서 많은 쪽으로만 답함
base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

models = {LOGIT_NAME: logit, TREE_NAME: tree, BASE_NAME: base}
preds = {name: m.predict(X_test) for name, m in models.items()}

# ---------------------------------------------------------------- 혼동행렬
n_test = len(y_test)
n_stroke = int(y_test.sum())
st.write(f"테스트용 **{n_test:,}명** 가운데 뇌졸중인 사람은 **{n_stroke:,}명**"
         f"({n_stroke / n_test * 100:.2f}%)이에요.")

st.header("혼동행렬 (행은 실제, 열은 예측)")


def counts(y_true, y_pred):
    """(TN, FP, FN, TP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def matrix_table(tn, fp, fn, tp):
    table = pd.DataFrame(
        [
            [f"TP: {tp:,}명", f"FN: {fn:,}명", f"{tp + fn:,}명"],
            [f"FP: {fp:,}명", f"TN: {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fn + fp + tn:,}명"],
        ],
        index=["실제 뇌졸중", "실제 아님", "합계"],
        columns=["예측 뇌졸중", "예측 아님", "합계"],
    )
    table.index.name = "실제 \\ 예측"
    return table


cm = {name: counts(y_test, preds[name]) for name in (LOGIT_NAME, TREE_NAME)}

c1, c2 = st.columns(2)
for col, name in zip((c1, c2), (LOGIT_NAME, TREE_NAME)):
    with col:
        st.subheader(name)
        st.table(matrix_table(*cm[name]))
st.caption("TP: 뇌졸중인 사람을 뇌졸중이라고 맞힘 · FN: 뇌졸중인 사람을 아니라고 놓침 · "
           "FP: 아닌 사람을 뇌졸중이라고 헛짚음 · TN: 아닌 사람을 아니라고 맞힘")

# ---------------------------------------------------------------- 지표 표
st.header("네 가지 지표")


def pct(x):
    return NO_VALUE if x is None else f"{x * 100:.2f}%"


def four_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    actual_pos = tp + fn  # 재현율의 분모: 실제 뇌졸중인 사람 수
    pred_pos = tp + fp  # 정밀도의 분모: 뇌졸중이라 예측한 사람 수
    recall = tp / actual_pos if actual_pos else None
    precision = tp / pred_pos if pred_pos else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return [pct(accuracy), pct(recall), pct(precision), pct(f1)]


rows = {}
for name in (LOGIT_NAME, TREE_NAME):
    rows[name] = four_metrics(*cm[name])
rows[BASE_NAME] = four_metrics(*counts(y_test, preds[BASE_NAME]))

metric_table = pd.DataFrame(rows, index=["정확도", "재현율", "정밀도", "F1"]).T
metric_table.index.name = "모델"
st.table(metric_table)
st.caption("정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN) · 정밀도 = TP ÷ (TP+FP) · "
           "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)")

# ---------------------------------------------------------------- 놓친 환자와 헛짚은 사람
st.header("놓친 환자와 헛짚은 사람")
pick = st.selectbox("어느 모델의 목록을 볼까요?", [LOGIT_NAME, TREE_NAME])
pred = preds[pick]

missed_mask = (y_test == 1) & (pred == 0)  # 실제 뇌졸중인데 아니라고 예측 (FN)
wrong_mask = (y_test == 0) & (pred == 1)  # 아닌데 뇌졸중이라고 예측 (FP)


def person_list(mask):
    sub = test_df.loc[mask, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]].copy()
    sub.columns = ["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"]
    sub["고혈압 여부"] = sub["고혈압 여부"].astype(int).map({0: "없음", 1: "있음"})
    sub["심장병 여부"] = sub["심장병 여부"].astype(int).map({0: "없음", 1: "있음"})
    sub = sub.sort_values(["나이", "번호"], ascending=[False, True])
    return sub.set_index("번호")


def show_list(col, base_title, mask):
    sub = person_list(mask)
    total = len(sub)
    with col:
        if total > MAX_ROWS:
            st.subheader(f"{base_title} (전체 {total:,}명 중 앞 {MAX_ROWS}명)")
        else:
            st.subheader(f"{base_title} (전체 {total:,}명)")
        if total == 0:
            st.write("해당하는 사람이 없어요.")
        else:
            st.dataframe(sub.head(MAX_ROWS))


left, right = st.columns(2)
show_list(left, "놓친 환자: 실제 뇌졸중인데 아니라고 예측", missed_mask)
show_list(right, "헛짚은 사람: 아닌데 뇌졸중이라고 예측", wrong_mask)

_, fp_cm, fn_cm, _ = cm[pick]
n_missed, n_wrong = int(missed_mask.sum()), int(wrong_mask.sum())
if n_missed == fn_cm and n_wrong == fp_cm:
    st.write(f"두 목록의 사람 수(놓친 환자 {n_missed:,}명, 헛짚은 사람 {n_wrong:,}명)는 "
             f"위 혼동행렬의 FN 칸({fn_cm:,}명), FP 칸({fp_cm:,}명)과 같아요.")
else:
    st.write(f"두 목록의 사람 수(놓친 환자 {n_missed:,}명, 헛짚은 사람 {n_wrong:,}명)가 "
             f"혼동행렬의 FN 칸({fn_cm:,}명), FP 칸({fp_cm:,}명)과 달라요. 확인이 필요해요.")
