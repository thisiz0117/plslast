import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import io
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 페이지 설정 ---
st.set_page_config(
    page_title="기후 위기-학업 영향 대시보드",
    page_icon="🌪️",
    layout="wide",
)

# --- 폰트 설정 ---
FONT_PATH = '/fonts/Pretendard-Bold.ttf'

def get_font_name(font_path):
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        # Matplotlib에 폰트 추가
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
        return font_name
    return None

font_name = get_font_name(FONT_PATH)

# --- 데이터 로드 및 전처리 (공식 데이터) ---
@st.cache_data(ttl=3600) # 1시간 동안 캐시
def load_public_data():
    """
    기상청 AWS S3에서 서울 월별 평균 기온 및 강수량 데이터 로드
    출처: 기상청 기상자료개방포털 (https://data.kma.go.kr/resources/AWS/since_2000_202312/CSV/MONTH/)
    데이터셋 설명: 2000년부터의 월별 기상 데이터
    """
    # 데이터 URL (2000년부터 2023년까지의 데이터 예시, 실제 운영시 최신 데이터 경로로 변경 필요)
    # 기상청 데이터는 연도별로 파일이 나뉘어 있어, 대표적인 파일 하나를 예시로 사용합니다.
    # 여러 연도를 합치려면 반복문으로 URL을 생성하여 로드해야 합니다.
    local_path = "data/gdp_data.csv"
    try:
        df = pd.read_csv(local_path, encoding='euc-kr')

        # gdp_data.csv에 맞는 데이터 전처리
        # 예시: 'Country Name', '1960', ..., '2022' 등 연도별 GDP 데이터
        # 대한민국 데이터만 추출
        df_korea = df[df['Country Name'] == 'Korea, Rep.'].copy()
        # 연도별 GDP 데이터만 추출
        years = [str(y) for y in range(1960, 2023)]
        gdp_values = df_korea[years].T.reset_index()
        gdp_values.columns = ['date', 'gdp']
        gdp_values['date'] = pd.to_datetime(gdp_values['date'], format='%Y')
        gdp_values['gdp'] = pd.to_numeric(gdp_values['gdp'], errors='coerce')
        gdp_values = gdp_values.dropna()
        gdp_values = gdp_values[gdp_values['date'] <= pd.to_datetime(datetime.now().date())]
        gdp_values.sort_values('date', inplace=True)
        return gdp_values, True

    except Exception as e:
        st.error(f"공식 데이터를 불러오는 데 실패했습니다: {e}. 예시 데이터로 대시보드를 표시합니다.")
        # 예시 데이터 생성
        dates = pd.to_datetime(pd.date_range(start="2022-01-01", end="2023-12-31", freq='MS'))
        # 오늘 이후 데이터 제거
        today = pd.to_datetime(datetime.now().date())
        dates = dates[dates <= today]
        
        data = {
            'date': dates,
            'value_temp': np.random.uniform(-5, 28, size=len(dates)) + np.sin(np.arange(len(dates)) * np.pi / 6) * 10,
            'value_rain': np.random.uniform(10, 350, size=len(dates)) * (np.sin(np.arange(len(dates)) * np.pi / 6) ** 2 + 0.1)
        }
        return pd.DataFrame(data), False

# --- 데이터 준비 (사용자 입력) ---
@st.cache_data
def load_user_data():
    """사용자 입력 텍스트를 기반으로 데이터프레임 생성"""
    data = {
        'event': [
            '태풍 카눈', '태풍 카눈', '태풍 카눈', '태풍 카눈',
            '전국 폭우', '전국 폭우', '전국 폭우', '전국 폭우', '전국 폭우', '전국 폭우',
            '충북 호우', '충북 호우', '충북 호우'
        ],
        'year': [
            2023, 2023, 2023, 2023,
            2025, 2025, 2025, 2025, 2025, 2025,
            2023, 2023, 2023
        ],
        'region': [
            '강원', '강원', '강원', '강원',
            '전국', '전국', '전국', '전국', '전국', '전국',
            '충북', '충북', '충북'
        ],
        'type': [
            '휴업', '등교시간 조정', '개학 연기', '원격수업',
            '학사일정 조정', '단축수업', '등교시간 조정', '휴업', '원격수업', '시설 피해',
            '피해 학교·유치원', '등교시간 조정', '원격수업'
        ],
        'value': [
            5, 1, 2, 2,
            247, 156, 59, 29, 3, 451,
            24, 7, 1
        ],
        'unit': ['곳'] * 13
    }
    df = pd.DataFrame(data)
    
    # 2025년 데이터는 가상의 데이터이므로, 오늘 날짜와 비교하여 필터링
    today = datetime.now()
    if today.year < 2025:
        # st.warning("2025년 데이터는 미래 시점의 가상 데이터입니다.")
        pass # 가상 데이터라도 일단 표시
    
    # 데이터 표준화: 'date' 컬럼 생성 (연도만 사용)
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    df.rename(columns={'type': 'group'}, inplace=True)
    return df

# --- 헬퍼 함수 ---
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- UI 그리기 ---
st.title("🌪️ 기후 위기와 학업 성취도 영향 대시보드")
st.write("이 대시보드는 기후 위기로 인한 자연재해가 학생들의 수업에 미치는 영향을 분석합니다.")

tab1, tab2 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력 데이터 대시보드"])

# --- 탭 1: 공식 공개 데이터 ---
with tab1:
    st.header("서울 월별 평균 기온 및 강수량 변화 (기상청)")
    
    df_public, data_loaded_successfully = load_public_data()

    if data_loaded_successfully:
        st.markdown("데이터 출처: [기상청 기상자료개방포털](https://data.kma.go.kr/resources/AWS/since_2000_202312/CSV/MONTH/) (예시: 2023년 데이터)")
    else:
        st.warning("공식 데이터 로드에 실패하여, 임의의 예시 데이터를 사용합니다.")

    # 사이드바 옵션
    st.sidebar.header("공식 데이터 옵션")
    selected_years = st.sidebar.slider(
        "연도 선택",
        min_value=df_public['date'].dt.year.min(),
        max_value=df_public['date'].dt.year.max(),
        value=(df_public['date'].dt.year.min(), df_public['date'].dt.year.max())
    )

    smoothing = st.sidebar.checkbox("이동 평균 보기 (3개월)")

    df_filtered = df_public[
        (df_public['date'].dt.year >= selected_years[0]) &
        (df_public['date'].dt.year <= selected_years[1])
    ]

    if smoothing:
        df_filtered['value_temp_smooth'] = df_filtered['value_temp'].rolling(window=3, min_periods=1).mean()
        df_filtered['value_rain_smooth'] = df_filtered['value_rain'].rolling(window=3, min_periods=1).mean()

    # 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 연도별 GDP 변화")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['gdp'], mode='lines+markers', name='GDP', marker_color='#ff6347'))
        # smoothing은 GDP에는 적용하지 않음
        fig_temp.update_layout(
            yaxis_title="GDP (current US$)",
            xaxis_title="연도",
            font=dict(family=font_name) if font_name else None
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        st.subheader("� 연도별 GDP 막대그래프")
        fig_gdp_bar = px.bar(df_filtered, x='date', y='gdp', title="", labels={'gdp': 'GDP (current US$)', 'date': '연도'})
        fig_gdp_bar.update_layout(
            yaxis_title="GDP (current US$)",
            xaxis_title="연도",
            font=dict(family=font_name) if font_name else None
        )
        st.plotly_chart(fig_gdp_bar, use_container_width=True)

    # 데이터 테이블 및 다운로드
    st.subheader("데이터 원본")
    st.dataframe(df_filtered)
    st.download_button(
        label="처리된 데이터 다운로드 (CSV)",
        data=to_csv(df_filtered),
        file_name="public_climate_data_processed.csv",
        mime="text/csv",
    )

# --- 탭 2: 사용자 입력 데이터 ---
with tab2:
    st.header("기상 이변으로 인한 학교 수업 차질 통계")
    st.markdown("사용자가 제공한 기사 및 연구 자료 기반 데이터 시각화입니다.")

    df_user = load_user_data()

    # 사이드바 옵션
    st.sidebar.header("사용자 데이터 옵션")
    selected_events = st.sidebar.multiselect(
        "재해 유형 선택",
        options=df_user['event'].unique(),
        default=df_user['event'].unique()
    )
    
    df_user_filtered = df_user[df_user['event'].isin(selected_events)]

    # 시각화
    st.subheader("📊 재해 유형별 학교 피해 현황")
    fig_user_bar = px.bar(
        df_user_filtered,
        x='event',
        y='value',
        color='group',
        barmode='group',
        title="주요 기상 재해별 학사일정 조정 및 피해 건수",
        labels={'value': '학교/피해 건수', 'event': '재해 유형', 'group': '조치 유형'}
    )
    fig_user_bar.update_layout(font=dict(family=font_name) if font_name else None)
    st.plotly_chart(fig_user_bar, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📍 지역별 피해 현황")
        # '전국' 제외
        df_region = df_user_filtered[df_user_filtered['region'] != '전국'].groupby('region')['value'].sum().reset_index()
        fig_region = px.pie(df_region, values='value', names='region', title="지역별 총 피해/조정 건수 (전국 제외)", hole=0.3)
        fig_region.update_traces(textposition='inside', textinfo='percent+label')
        fig_region.update_layout(font=dict(family=font_name) if font_name else None)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col4:
        st.subheader("🗓️ 2025년 전국 폭우 상세 분석")
        df_2025_rain = df_user_filtered[df_user_filtered['event'] == '전국 폭우']
        if not df_2025_rain.empty:
            fig_2025 = px.bar(
                df_2025_rain,
                x='group',
                y='value',
                color='group',
                title="2025년 전국 폭우 조치 유형별 상세",
                labels={'value': '학교/피해 건수', 'group': '조치 유형'}
            )
            fig_2025.update_layout(font=dict(family=font_name) if font_name else None, showlegend=False)
            st.plotly_chart(fig_2025, use_container_width=True)
        else:
            st.info("'전국 폭우' 데이터가 선택되지 않았습니다.")


    # 데이터 테이블 및 다운로드
    st.subheader("데이터 원본")
    st.dataframe(df_user_filtered)
    st.download_button(
        label="처리된 데이터 다운로드 (CSV)",
        data=to_csv(df_user_filtered),
        file_name="user_disaster_data_processed.csv",
        mime="text/csv",
    )
