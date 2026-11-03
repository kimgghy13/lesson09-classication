import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.title("분류 모델")
st.caption("stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중을 양성으로 두었어요.")

NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURES = list(NAMES)
BINARY = ["hypertension", "heart_disease"]

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE, RED = "#1f77b4", "#d62728"
BLUE_LIGHT, RED_LIGHT = "rgba(31,119,180,0.18)", "rgba(214,39,40,0.18)"


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

# ---------------------------------------------------------------- 속성 고르기
st.header("입력으로 사용할 속성")
picked = st.multiselect(
    "입력으로 사용할 속성을 골라 주세요.",
    options=FEATURES,
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda f: NAMES[f],
)
chosen = [f for f in FEATURES if f in picked]
if len(chosen) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요. 그래야 아래 결과를 보여 드릴 수 있어요.")
    st.stop()

# ---------------------------------------------------------------- 데이터 나누기
# 번호 순으로 정렬 → 열 명씩 묶음 → 각 묶음의 앞 세 명은 테스트용
df = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3

X = df[chosen].astype(float)
y = df["stroke"].astype(int)

X_train = X.loc[~is_test].copy()
X_test = X.loc[is_test].copy()
y_train = y.loc[~is_test].to_numpy()
y_test = y.loc[is_test].to_numpy()

st.write(f"학습용 **{len(X_train):,}명**, 테스트용 **{len(X_test):,}명**으로 나누었어요. "
         "(번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정)")

# 체질량지수는 골랐을 때만, 훈련용의 중앙값으로 채운다.
if "bmi" in chosen:
    bmi_median = X_train["bmi"].median()
    n_fill_train = int(X_train["bmi"].isna().sum())
    n_fill_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수가 빈 사람(훈련용 {n_fill_train}명, 테스트용 {n_fill_test}명)은 "
               f"훈련용의 중앙값 {bmi_median:.1f}로 채웠어요.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("고른 속성에 비어 있는 값이 있어 모델을 만들 수 없어요.")
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

# ---------------------------------------------------------------- 정확도 카드
st.header("정확도")


def card(title, model):
    train_acc = model.score(X_train, y_train) * 100
    test_acc = model.score(X_test, y_test) * 100
    html = "".join([
        '<div style="border:1px solid rgba(128,128,128,0.45);border-radius:12px;',
        'padding:16px 12px;background:rgba(128,128,128,0.08);text-align:center;height:100%;">',
        f'<div style="font-size:0.9rem;font-weight:600;min-height:3.6em;">{title}</div>',
        f'<div style="font-size:2.8rem;font-weight:700;line-height:1.3;">{test_acc:.2f}%</div>',
        '<div style="font-size:0.8rem;opacity:0.75;">테스트 데이터의 정확도</div>',
        '<div style="display:flex;justify-content:space-around;margin-top:12px;font-size:0.78rem;">',
        f'<div>훈련 데이터<br><b>{train_acc:.2f}%</b></div>',
        f'<div>테스트 데이터<br><b>{test_acc:.2f}%</b></div>',
        '</div></div>',
    ])
    st.markdown(html, unsafe_allow_html=True)


c1, c2, c3 = st.columns(3)
with c1:
    card(LOGIT_NAME, logit)
with c2:
    card(TREE_NAME, tree)
with c3:
    card(BASE_NAME, base)

# ---------------------------------------------------------------- 산점도
st.header("테스트 데이터의 산점도와 두 모델의 경계")
sc1, sc2 = st.columns(2)
with sc1:
    x_feat = st.selectbox("가로축", chosen, index=0, format_func=lambda f: NAMES[f])
with sc2:
    y_options = [f for f in chosen if f != x_feat]
    y_feat = st.selectbox("세로축", y_options, index=0, format_func=lambda f: NAMES[f])


def axis_range(f):
    if f in BINARY:
        return -0.3, 1.3
    lo, hi = float(X_test[f].min()), float(X_test[f].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


x_lo, x_hi = axis_range(x_feat)
y_lo, y_hi = axis_range(y_feat)

G = 150
gx = np.linspace(x_lo, x_hi, G)
gy = np.linspace(y_lo, y_hi, G)
XX, YY = np.meshgrid(gx, gy)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다.
medians = X_test.median()
grid = pd.DataFrame({f: np.full(XX.size, float(medians[f])) for f in chosen})
grid[x_feat] = XX.ravel()
grid[y_feat] = YY.ravel()

prob = logit.predict_proba(grid)[:, 1].reshape(G, G)
tree_pred = tree.predict(grid).reshape(G, G)

fig = go.Figure()

# 의사결정트리가 나눈 칸(옅은 색)
fig.add_trace(go.Heatmap(
    x=gx, y=gy, z=tree_pred, zmin=0, zmax=1,
    colorscale=[[0, BLUE_LIGHT], [1, RED_LIGHT]],
    showscale=False, hoverinfo="skip", showlegend=False,
))
# 로지스틱 회귀가 0.5로 가르는 선
fig.add_trace(go.Contour(
    x=gx, y=gy, z=prob, showscale=False, hoverinfo="skip", showlegend=False,
    contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
    line=dict(color="black", width=3),
))

# 테스트 데이터 점 (이진 속성은 겹치지 않게 살짝 흩뿌려 그림)
rng = np.random.RandomState(0)


def jitter(values, feat):
    v = values.astype(float).copy()
    if feat in BINARY:
        v = v + rng.uniform(-0.06, 0.06, size=len(v))
    return v


for label, value, color in [("아님", 0, BLUE), ("뇌졸중(양성)", 1, RED)]:
    m = y_test == value
    fig.add_trace(go.Scatter(
        x=jitter(X_test[x_feat].to_numpy()[m], x_feat),
        y=jitter(X_test[y_feat].to_numpy()[m], y_feat),
        mode="markers", name=f"실제: {label}",
        marker=dict(color=color, size=6, opacity=0.7),
    ))

# 범례용 표시(선과 칠한 칸)
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="lines", name="로지스틱 회귀가 0.5로 가르는 선",
    line=dict(color="black", width=3),
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers", name="의사결정트리: 아님이라고 답하는 칸",
    marker=dict(color=BLUE_LIGHT, size=14, symbol="square", line=dict(color=BLUE, width=1)),
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers", name="의사결정트리: 뇌졸중이라고 답하는 칸",
    marker=dict(color=RED_LIGHT, size=14, symbol="square", line=dict(color=RED, width=1)),
))

fig.update_xaxes(title=NAMES[x_feat], range=[x_lo, x_hi])
fig.update_yaxes(title=NAMES[y_feat], range=[y_lo, y_hi])
for feat, update in ((x_feat, fig.update_xaxes), (y_feat, fig.update_yaxes)):
    if feat in BINARY:
        update(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(height=560, legend=dict(orientation="h", yanchor="top", y=-0.18))
st.plotly_chart(fig)

others = [f for f in chosen if f not in (x_feat, y_feat)]
if others:
    parts = []
    for f in others:
        v = float(medians[f])
        parts.append(f"{NAMES[f]} {v:.0f}" if f in BINARY else f"{NAMES[f]} {v:.1f}")
    note = "두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: " + ", ".join(parts) + "."
    if "bmi" in others:
        note += " (체질량지수는 빈 칸을 채운 뒤의 중앙값이에요.)"
    st.write(note)
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 값은 없어요.")

if prob.max() < 0.5:
    st.write(f"로지스틱 회귀가 0.5로 가르는 선은 이 그림 밖에 있어요. "
             f"그림 안에서는 뇌졸중일 확률이 어디서나 0.5보다 작아요(가장 큰 값 {prob.max():.3f}).")
elif prob.min() >= 0.5:
    st.write(f"로지스틱 회귀가 0.5로 가르는 선은 이 그림 밖에 있어요. "
             f"그림 안에서는 뇌졸중일 확률이 어디서나 0.5 이상이에요(가장 작은 값 {prob.min():.3f}).")
else:
    st.write("검은 선이 로지스틱 회귀가 뇌졸중 확률 0.5로 가르는 선이에요.")
if any(f in BINARY for f in (x_feat, y_feat)):
    st.caption("고혈압·심장병 축은 점이 겹치지 않도록 그림에서만 살짝 흩뿌렸어요.")

# ---------------------------------------------------------------- 의사결정트리 가지 그림
st.header("의사결정트리가 던진 질문")

t = tree.tree_
path = tree.decision_path(X_train)
n_node = np.asarray(path.sum(axis=0)).ravel().astype(int)
n_pos = np.asarray(path.T.dot(y_train)).ravel().astype(int)

dot = [
    "digraph Tree {",
    'graph [bgcolor="white"];',
    'node [shape=box, style="rounded,filled", fontcolor="black", color="#555555", '
    'width=2.2, margin="0.15,0.08"];',
    'edge [fontcolor="black", color="#555555"];',
]
n_leaves = 0
n_leaves_no = 0
for i in range(t.node_count):
    n, p = int(n_node[i]), int(n_pos[i])
    rate = p / n * 100 if n else 0.0
    stats = f"훈련용 {n}명\\n뇌졸중 {p}명 ({rate:.1f}%)"
    is_leaf = t.children_left[i] == -1
    if is_leaf:
        answer_pos = p > (n - p)
        n_leaves += 1
        n_leaves_no += 0 if answer_pos else 1
        answer = "뇌졸중" if answer_pos else "아님"
        fill = "#f4b6b6" if answer_pos else "#b6d4f0"
        dot.append(f'n{i} [label="답: {answer}\\n{stats}", fillcolor="{fill}"];')
    else:
        feat = chosen[t.feature[i]]
        if feat in BINARY:
            question = f"{NAMES[feat]}이 없나요?"
        else:
            question = f"{NAMES[feat]} ≤ {t.threshold[i]:.1f} ?"
        dot.append(f'n{i} [label="{question}\\n{stats}", fillcolor="#f2f2f2"];')
        dot.append(f'n{i} -> n{t.children_left[i]} [label="예"];')
        dot.append(f'n{i} -> n{t.children_right[i]} [label="아니요"];')
dot.append("}")
st.graphviz_chart("\n".join(dot))

st.write(f"답을 내는 마디는 모두 **{n_leaves}칸**이고, 그중 **{n_leaves_no}칸**이 '아님'이라고 답해요.")
asked = sorted({int(f) for f in t.feature if f >= 0})
if asked:
    asked_names = ", ".join(NAMES[chosen[i]] for i in asked)
    not_asked = [NAMES[f] for k, f in enumerate(chosen) if k not in asked]
    tail = f" (묻지 않은 속성: {', '.join(not_asked)})" if not_asked else ""
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 **{asked_names}**이에요.{tail}")
else:
    st.write("이 나무는 아무것도 묻지 않고 바로 답을 냈어요.")
