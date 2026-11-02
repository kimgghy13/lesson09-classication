import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
# 알려 준 주소는 404가 나서, 올바른 형태의 주소들을 차례로 시도한다.
DATA_URLS = [
    "https://raw.githubusercontent.com/kimgghy13/data/refs/heads/main/stroke.csv",
    "https://raw.githubusercontent.com/kimgghy13/data/main/stroke.csv",
    "https://raw.githubusercontent.com/kimgghy13/data/refs/main/stroke.csv",
]
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

# 브라우저 탭 제목과 아이콘
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 큰 숫자 카드 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 저장 / 불러오기 ----------
def load_notes() -> dict:
    """내_기록/기록.json 을 읽어 온다. 없거나 깨져 있으면 빈 기록을 돌려준다."""
    empty = {"우리말_뜻": {}, "출처": ""}
    if not SAVE_FILE.exists():
        return empty
    try:
        data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    if not isinstance(data, dict):
        return empty
    return {
        "우리말_뜻": data.get("우리말_뜻", {}) if isinstance(data.get("우리말_뜻"), dict) else {},
        "출처": data.get("출처", "") if isinstance(data.get("출처"), str) else "",
    }


def save_notes(notes: dict) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FILE.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    last_error = None
    for url in DATA_URLS:
        try:
            return pd.read_csv(url, encoding="utf-8")
        except Exception as e:
            last_error = e
    raise last_error


def describe_values(series: pd.Series) -> str:
    """열에 들어 있는 값의 종류를 짧은 글로 정리한다."""
    values = series.dropna()
    if values.empty:
        return "-"
    if pd.api.types.is_numeric_dtype(values):
        uniques = sorted(values.unique())
        if len(uniques) <= 2:
            return ", ".join(f"{v:g}" for v in uniques)
        return f"숫자 ({values.min():g} ~ {values.max():g})"
    return " / ".join(sorted(map(str, values.unique())))


try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 데이터 주소를 확인해 주세요.\n\n{e}")
    st.stop()

notes = load_notes()

# ---------- 화면 맨 위 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("뇌졸중(stroke) 여부와 나이, 혈당, 생활 습관 등을 담은 데이터를 살펴보는 첫 화면입니다.")

# ---------- 큰 숫자 카드 4개 ----------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.2f}%")

# ---------- 열 설명 표 ----------
st.subheader("열 설명 표")
st.write("**우리말 뜻** 칸은 교재를 보고 직접 채워 넣으세요. 다 적은 뒤 저장 버튼을 누르면 새로고침해도 남아 있어요.")

saved_meanings = notes["우리말_뜻"]
info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [saved_meanings.get(col, "") for col in df.columns],
        "값의 종류": [describe_values(df[col]) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        info,
        key="meaning_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻 ✏️", help="교재를 보고 직접 적어 보세요."
            ),
        },
    )
    save_meaning = st.form_submit_button("우리말 뜻 저장", type="primary")

if save_meaning:
    current = load_notes()
    current["우리말_뜻"] = {
        row["열 이름"]: (row["우리말 뜻"] or "").strip() for _, row in edited.iterrows()
    }
    save_notes(current)
    st.success("우리말 뜻을 저장했어요.")

# ---------- 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

# ---------- 데이터 출처 ----------
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "출처",
        value=notes["출처"],
        height=120,
        placeholder="교재에 나온 데이터 출처를 여기에 적어 주세요.",
        label_visibility="collapsed",
    )
    save_source = st.form_submit_button("출처 저장", type="primary")

if save_source:
    current = load_notes()
    current["출처"] = source_text.strip()
    save_notes(current)
    st.success("출처를 저장했어요.")
