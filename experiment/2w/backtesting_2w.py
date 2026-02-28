import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
PRICE_LABEL = {"open": "시가(Open)", "close": "종가(Close)", "vwap": "VWAP"}
KOSPI = "KS11"
KOSPI200 = "KS200"
KOACT = "441800"  # KoAct 배당성장액티브 ETF
RISK_FREE_ANNUAL = 0.03

GROUP_PERIODS = OrderedDict({
    "g1":  ("2025-01-02", "2025-01-15"),
    "g2":  ("2025-01-16", "2025-02-04"),
    "g3":  ("2025-02-05", "2025-02-18"),
    "g4":  ("2025-02-19", "2025-03-06"),
    "g5":  ("2025-03-07", "2025-03-20"),
    "g6":  ("2025-03-21", "2025-04-03"),
    "g7":  ("2025-04-04", "2025-04-17"),
    "g8":  ("2025-04-18", "2025-05-02"),
    "g9":  ("2025-05-07", "2025-05-20"),
    "g10": ("2025-05-21", "2025-06-02"),
    "g11": ("2025-06-04", "2025-06-18"),
    "g12": ("2025-06-19", "2025-07-02"),
    "g13": ("2025-07-03", "2025-07-16"),
    "g14": ("2025-07-17", "2025-07-30"),
    "g15": ("2025-07-31", "2025-08-13"),
    "g16": ("2025-08-14", "2025-08-29"),
    "g17": ("2025-09-01", "2025-09-12"),
    "g18": ("2025-09-15", "2025-09-26"),
    "g19": ("2025-09-29", "2025-10-17"),
    "g20": ("2025-10-20", "2025-10-31"),
    "g21": ("2025-11-03", "2025-11-14"),
    "g22": ("2025-11-17", "2025-11-28"),
    "g23": ("2025-12-01", "2025-12-12"),
    "g24": ("2025-12-15", "2025-12-29"),
    "g25": ("2025-12-30", "2026-01-14"),
})

GROUP_KEYS = list(GROUP_PERIODS.keys())


def get_invest_period(select_group):
    """GN 선정 → GN+1 기간 반환"""
    idx = GROUP_KEYS.index(select_group)
    if idx + 1 >= len(GROUP_KEYS):
        return None
    next_group = GROUP_KEYS[idx + 1]
    return next_group, GROUP_PERIODS[next_group]


# ─────────────────────────────────────────────
# 가격 및 수익률 계산
# ─────────────────────────────────────────────
def _get_entry_exit_price(df_price, method):
    if method == "open":
        return df_price['Open'].iloc[0], df_price['Open'].iloc[-1]
    elif method == "close":
        return df_price['Close'].iloc[0], df_price['Close'].iloc[-1]
    elif method == "vwap":
        typical = (df_price['High'] + df_price['Low'] + df_price['Close']) / 3
        return typical.iloc[0], typical.iloc[-1]
    raise ValueError(f"지원하지 않는 가격 기준: {method}")


def get_period_return(ticker, start, end, method="close"):
    try:
        df = fdr.DataReader(ticker, start, end)
        if df.empty or len(df) < 2:
            print(f"[FDR] {ticker} {start}~{end}: 빈 데이터 (len={len(df) if not df.empty else 0})")
            return 0
        entry, exit_ = _get_entry_exit_price(df, method)
        if entry == 0:
            print(f"[FDR] {ticker} {start}~{end}: 시가 0")
            return 0
        return (exit_ / entry) - 1
    except Exception as e:
        print(f"[FDR] {ticker} {start}~{end}: {type(e).__name__} — {e}")
        return 0


# ─────────────────────────────────────────────
# 비중 계산
# ─────────────────────────────────────────────
def calc_equal_weight(df):
    scores = df['비고'].apply(lambda x: 2 if '중복' in str(x) else 1)
    return scores / scores.sum()


def calc_score_weight(df):
    scores = df['최종점수'].clip(lower=0)
    total = scores.sum()
    if total == 0:
        return pd.Series(1 / len(df), index=df.index)
    return scores / total


# ─────────────────────────────────────────────
# 성과 지표
# ─────────────────────────────────────────────
def calc_sharpe(rets, rf_annual=RISK_FREE_ANNUAL, periods_per_year=26):
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = rets - rf_period
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year) if excess.std() != 0 else 0


def calc_mdd(rets):
    cum = (1 + rets).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def calc_ir(s_ret, b_ret, periods_per_year=26):
    excess = s_ret - b_ret
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year) if excess.std() != 0 else 0


def calc_win_rate(s_ret, b_ret):
    return (s_ret > b_ret).sum() / len(s_ret) if len(s_ret) > 0 else 0


def summarize(label, s_ret, b_ret):
    n = len(s_ret)
    return {
        '전략명': label,
        '총 수익률': f"{((1+s_ret).prod()-1)*100:.2f}%",
        'KOSPI 총 수익률': f"{((1+b_ret).prod()-1)*100:.2f}%",
        '초과수익률': f"{(((1+s_ret).prod()-1)-((1+b_ret).prod()-1))*100:.2f}%p",
        f'샤프 비율 (연율화, {n}기간)': f"{calc_sharpe(s_ret, periods_per_year=n):.3f}",
        'MDD': f"{calc_mdd(s_ret)*100:.2f}%",
        '정보 비율 (IR)': f"{calc_ir(s_ret, b_ret, periods_per_year=n):.3f}",
        '승률 (vs KOSPI)': f"{calc_win_rate(s_ret, b_ret)*100:.1f}% ({(s_ret>b_ret).sum()}/{n})",
        '기간 평균 수익률': f"{s_ret.mean()*100:.2f}%",
        '기간 변동성': f"{s_ret.std()*100:.2f}%",
    }


# ─────────────────────────────────────────────
# 백테스팅 메인 (base_dir을 매개변수로 받음)
# ─────────────────────────────────────────────
def run_backtest(base_dir, price_method="close", progress_callback=None):
    """
    base_dir: CSV 폴더 경로 (예: './data/rebal_2w_csv/외국인단독')
    progress_callback: (current, total, msg) -> None  (Streamlit 등에서 진행률 표시용)
    """
    available_csvs = sorted(
        [f.replace('.csv', '') for f in os.listdir(base_dir) if f.endswith('.csv')],
        key=lambda x: int(x.replace('g', ''))
    )

    investable = [g for g in available_csvs if get_invest_period(g) is not None]
    total = len(investable)
    results = []
    holdings_map = {}

    for idx, select_group in enumerate(investable):
        invest_group, (start_date, end_date) = get_invest_period(select_group)

        csv_path = os.path.join(base_dir, f"{select_group}.csv")
        df = pd.read_csv(csv_path)
        df['티커'] = df['티커'].astype(str).str.zfill(6)

        w_eq = calc_equal_weight(df)
        w_sc = calc_score_weight(df)

        if progress_callback:
            progress_callback(idx + 1, total,
                              f"{select_group} → {invest_group} ({start_date}~{end_date})")

        stock_rets = []
        for _, row in df.iterrows():
            ret = get_period_return(row['티커'], start_date, end_date, method=price_method)
            stock_rets.append(ret)
        stock_rets_arr = np.array(stock_rets)

        ret_eq = np.dot(stock_rets_arr, w_eq.values)
        ret_sc = np.dot(stock_rets_arr, w_sc.values)
        ret_bench = get_period_return(KOSPI, start_date, end_date, method=price_method)
        ret_k200 = get_period_return(KOSPI200, start_date, end_date, method=price_method)
        ret_koact = get_period_return(KOACT, start_date, end_date, method=price_method)

        results.append({
            'SelectGroup': select_group,
            'InvestGroup': invest_group,
            'Period': f"{start_date}~{end_date}",
            'StartDate': start_date,
            'EndDate': end_date,
            'EqualWeight': ret_eq,
            'ScoreWeight': ret_sc,
            'KOSPI': ret_bench,
            'KOSPI200': ret_k200,
            'KoAct': ret_koact,
        })

        # 보유종목 상세 저장
        detail = df[['티커', '종목명', '최종점수', '비고']].copy()
        detail['w_equal'] = w_eq.values
        detail['w_score'] = w_sc.values
        detail['return'] = stock_rets
        detail['contrib_eq'] = detail['return'] * detail['w_equal']
        detail['contrib_sc'] = detail['return'] * detail['w_score']
        holdings_map[invest_group] = detail

    res = pd.DataFrame(results)
    res['EW_Cum'] = (1 + res['EqualWeight']).cumprod() - 1
    res['SW_Cum'] = (1 + res['ScoreWeight']).cumprod() - 1
    res['KOSPI_Cum'] = (1 + res['KOSPI']).cumprod() - 1
    res['K200_Cum'] = (1 + res['KOSPI200']).cumprod() - 1
    res['KoAct_Cum'] = (1 + res['KoAct']).cumprod() - 1

    m_eq = summarize("동일비중 (중복2배)", res['EqualWeight'], res['KOSPI'])
    m_sc = summarize("점수비중 (최종점수)", res['ScoreWeight'], res['KOSPI'])
    m_ka = summarize("KoAct 배당성장", res['KoAct'], res['KOSPI'])

    return res, m_eq, m_sc, m_ka, holdings_map


# ─────────────────────────────────────────────
# CLI 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2주 리밸런싱 백테스팅 - 동일비중 vs 점수비중 비교")
    parser.add_argument("--signal", type=str, default="외국인단독",
                        choices=["외국인단독", "기관포함"],
                        help="시그널 유형 선택")
    parser.add_argument("--price", type=str, default="close",
                        choices=["open", "close", "vwap"],
                        help="수익률 계산 기준")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), f"../../data/file/rebal_2w_csv/{args.signal}")
    result, m_eq, m_sc, m_ka, _ = run_backtest(base_dir, price_method=args.price)

    print("\n" + "=" * 100)
    print(f"  2주 리밸런싱 백테스팅 성과 보고서")
    print(f"  시그널: {args.signal} | 가격: {PRICE_LABEL[args.price]} | 총 {len(result)}기간")
    print("=" * 100)

    disp = result[['InvestGroup', 'Period']].copy()
    disp['동일비중'] = result['EqualWeight'].apply(lambda x: f"{x*100:+.2f}%")
    disp['점수비중'] = result['ScoreWeight'].apply(lambda x: f"{x*100:+.2f}%")
    disp['KOSPI'] = result['KOSPI'].apply(lambda x: f"{x*100:+.2f}%")
    disp['KOSPI200'] = result['KOSPI200'].apply(lambda x: f"{x*100:+.2f}%")
    disp['KoAct'] = result['KoAct'].apply(lambda x: f"{x*100:+.2f}%")
    disp['동일(누적)'] = result['EW_Cum'].apply(lambda x: f"{x*100:+.2f}%")
    disp['점수(누적)'] = result['SW_Cum'].apply(lambda x: f"{x*100:+.2f}%")
    disp['KOSPI(누적)'] = result['KOSPI_Cum'].apply(lambda x: f"{x*100:+.2f}%")
    disp['K200(누적)'] = result['K200_Cum'].apply(lambda x: f"{x*100:+.2f}%")
    disp['KoAct(누적)'] = result['KoAct_Cum'].apply(lambda x: f"{x*100:+.2f}%")
    print(disp.to_string(index=False))

    print("\n" + "-" * 120)
    print("  [ 성과 지표 비교 ]")
    print("-" * 120)
    header = f"  {'지표':34s} | {'동일비중 (중복2배)':>18s} | {'점수비중 (최종점수)':>18s} | {'KoAct 배당성장':>18s}"
    print(header)
    print("  " + "-" * 96)
    for key in list(m_eq.keys())[1:]:
        print(f"  {key:34s} | {m_eq[key]:>18s} | {m_sc[key]:>18s} | {m_ka[key]:>18s}")
    print("-" * 120)

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'2주 리밸런싱: 동일비중 vs 점수비중  [{args.signal} / {PRICE_LABEL[args.price]}]',
                 fontsize=14, fontweight='bold')

    x_labels = result['InvestGroup']

    ax1 = axes[0]
    ax1.plot(x_labels, result['EW_Cum'] * 100,
             label='동일비중 (중복2배)', marker='o', linewidth=1.5, markersize=4, color='#2196F3')
    ax1.plot(x_labels, result['SW_Cum'] * 100,
             label='점수비중 (최종점수)', marker='s', linewidth=1.5, markersize=4, color='#FF9800')
    ax1.plot(x_labels, result['KOSPI_Cum'] * 100,
             label='KOSPI', linestyle='--', linewidth=1.5, color='#9E9E9E')
    ax1.plot(x_labels, result['K200_Cum'] * 100,
             label='KOSPI 200', linestyle='--', linewidth=1.5, color='#607D8B')
    ax1.plot(x_labels, result['KoAct_Cum'] * 100,
             label='KoAct 배당성장', linestyle='-.', linewidth=1.5, color='#8E24AA')
    ax1.set_ylabel('누적 수익률 (%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    ax2 = axes[1]
    diff = (result['EqualWeight'] - result['KOSPI']) * 100
    colors = ['#4CAF50' if x >= 0 else '#F44336' for x in diff]
    ax2.bar(x_labels, diff, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('동일비중 초과수익 (%p)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    output_file = f"result_2w_{args.signal}_{args.price}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n>> 완료! 그래프: '{output_file}'")
