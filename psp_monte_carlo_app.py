# Streamlit web app: Monte‑Carlo PSP Comparator (Quarter-level, No Dates)
# --------------------------------------------------------------
# How to run locally:
#   1) pip install -U streamlit pandas numpy plotly
#   2) streamlit run psp_monte_carlo_app.py
#   3) Open the local URL from Streamlit output
#
# Notes:
# - Works with your quarterly export (no dates needed).
# - Commissions: percent + fixed per successful transaction (no mins/tiers).
# - Real PSP is simulated via bootstrap of successful payments (preserves shape).
# - Custom PSP supports scalar or (low, high) ranges for all params.
# - Churn/Switch logic is included.
# - Sensitivity heatmaps included (optional).
# --------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="PSP Monte-Carlo Comparator", layout="wide")
st.title("PSP Monte‑Carlo Comparator (Quarter-level)")

# ---------------------------
# Helpers
# ---------------------------

def sample_param(value, size):
    """Scalar -> vector of same; (low, high) -> Uniform(low, high, size)."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return np.random.uniform(low, high, size)
    return np.full(size, float(value))


def lognormal_params_from_mean_std(mean, std):
    """
    Поддерживает скаляры И массивы.
    Возвращает mu, sigma, sigma2 той же формы, что и вход.
    """
    mean = np.asarray(mean, dtype=float)
    std  = np.asarray(std,  dtype=float)
    mean = np.maximum(mean, 1e-9)
    std  = np.maximum(std,  1e-9)
    sigma2 = np.log1p((std**2) / (mean**2))  # численно стабильнее
    sigma  = np.sqrt(sigma2)
    mu     = np.log(mean) - 0.5 * sigma2
    return mu, sigma, sigma2



# ---------------------------
# Preprocessing
# ---------------------------
@st.cache_data(show_spinner=False)
def load_default_csv():
    candidates = [
        ("sample.csv.gz", None, "gzip"),
        ("sample.csv",     "utf-16", None),
        ("monte (1).csv",  "utf-16", None),
        ("monte.csv",      "utf-16", None),
    ]
    for name, enc, comp in candidates:
        try:
            if comp == "gzip":
                return pd.read_csv(name, encoding=enc)  # pandas сам поймёт .gz
            return pd.read_csv(name, encoding=enc) if enc else pd.read_csv(name)
        except Exception:
            # вторая попытка без явной кодировки
            try:
                return pd.read_csv(name)
            except Exception:
                continue
    return None


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Business rules from your notebook
    if "order_amount" not in df.columns:
        df["order_amount"] = df.get("pay_total_money", 0).fillna(0)
    if "is_success" not in df.columns:
        df["is_success"] = ((df.get("order_state") == "purchased") & (df.get("pay_total_money", 0).fillna(0) > 0)).astype(int)
    # if "channel" in df.columns:
        df = df[df["channel"] == "Online"]
    keep = [c for c in ["customer_id", "cart_id", "order_id", "pay_type", "order_state", "order_amount", "is_success"] if c in df.columns]
    return df[keep].copy()


def get_payment_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("pay_type").agg(
        rows=("order_amount", "size"),
        success=("is_success", "sum"),
        fail=("is_success", lambda x: (1 - x).sum()),
        aov_mean=("order_amount", "mean"),
        aov_std=("order_amount", "std"),
    ).reset_index()
    stats["success_rate"] = (stats["success"] / stats["rows"]).round(4)
    stats["fail_rate"] = (1 - stats["success_rate"]).round(4)
    return stats


def extract_real_psp_empirics(df: pd.DataFrame, pay_type: str, count_by_cart_id: bool = False):
    d = df[df["pay_type"] == pay_type]
    if d.empty:
        raise ValueError(f"No rows for pay_type='{pay_type}'")
    if count_by_cart_id and "cart_id" in d.columns:
        orders_quarter = int(d["cart_id"].nunique())
        # approximate successes at order-level via mean success prob
        p_succ = float(d["is_success"].mean())
        success_quarter = int(round(orders_quarter * p_succ))
    else:
        orders_quarter = int(len(d))
        success_quarter = int(d["is_success"].sum())
    fail_quarter = int(orders_quarter - success_quarter)
    succ_amounts = d.loc[d["is_success"] == 1, "order_amount"].astype(float).values
    if succ_amounts.size == 0:
        succ_amounts = np.array([0.0])
    return dict(orders_quarter=orders_quarter, success_quarter=success_quarter, fail_quarter=fail_quarter, succ_amounts=succ_amounts)


# ---------------------------
# Simulators
# ---------------------------

def simulate_real_psp(
    emp,
    commission_percent=0.02,
    commission_fixed=0.0,
    *,
    churn_rate_after_fail=0.30,
    switch_rate=0.20,
    conversion_after_switch=0.50,
    switch_commission_percent=None,
    switch_commission_fixed=None,
    churn_loss_per_order=0.0,
    n_iter=10000,
):
    lam = max(emp["orders_quarter"], 0)
    alpha = emp["success_quarter"] + 1
    beta = emp["fail_quarter"] + 1
    succ_amounts = emp["succ_amounts"]

    if switch_commission_percent is None:
        switch_commission_percent = commission_percent
    if switch_commission_fixed is None:
        switch_commission_fixed = commission_fixed

    orders = np.random.poisson(lam=lam, size=n_iter)
    p_succ0 = np.random.beta(alpha, beta, size=n_iter)
    p_succ0 = np.clip(p_succ0, 0.0, 1.0)
    successes0 = np.random.binomial(n=orders, p=p_succ0)
    fails = orders - successes0

    churn_rate = np.clip(churn_rate_after_fail, 0.0, 1.0)
    churned = np.random.binomial(fails, churn_rate)
    remaining = fails - churned

    switch_rate = np.clip(switch_rate, 0.0, 1.0)
    switched = np.random.binomial(remaining, switch_rate)

    conv = np.clip(conversion_after_switch, 0.0, 1.0)
    recovered = np.random.binomial(switched, conv)

    gross0 = np.zeros(n_iter, dtype=float)
    gross_rec = np.zeros(n_iter, dtype=float)
    for i in range(n_iter):
        k0 = int(successes0[i])
        kr = int(recovered[i])
        if k0 > 0:
            gross0[i] = float(np.random.choice(succ_amounts, size=k0, replace=True).sum())
        if kr > 0:
            gross_rec[i] = float(np.random.choice(succ_amounts, size=kr, replace=True).sum())

    comm0 = gross0 * float(commission_percent) + successes0 * float(commission_fixed)
    commr = gross_rec * float(switch_commission_percent) + recovered * float(switch_commission_fixed)
    churn_loss = float(churn_loss_per_order) * churned

    net = (gross0 + gross_rec) - (comm0 + commr) - churn_loss
    return net


def simulate_custom_psp(params, n_iter=10000, aov_cap=None, tight=False):
    """
    Custom flow: prim. successes/fails -> churn -> switch -> conversion.
    AOV modeled via CLT for avg check per quarter (separately for primary and recovered).
    """
    lam_draw = np.clip(sample_param(params["orders_mu"], n_iter), 0, None)
    if tight:
        lam_fixed = int(np.round(np.mean(lam_draw)))
        orders = np.full(n_iter, lam_fixed)
        p_succ0 = 1.0 - float(np.mean(np.clip(sample_param(params["fail_rate"], n_iter), 0.0, 0.999999)))
        successes0 = np.random.binomial(n=orders, p=p_succ0)
    else:
        orders = np.random.poisson(lam=lam_draw)
        fail = np.clip(sample_param(params["fail_rate"], n_iter), 0.0, 0.999999)
        p_succ0 = 1.0 - fail
        successes0 = np.random.binomial(n=orders, p=p_succ0)
    fails = orders - successes0

    churn_rate = np.clip(sample_param(params.get("churn_rate_after_fail", 0.0), n_iter), 0.0, 1.0)
    churned = np.random.binomial(fails, churn_rate)
    remaining = fails - churned

    switch_rate = np.clip(sample_param(params.get("switch_rate", 0.0), n_iter), 0.0, 1.0)
    switched = np.random.binomial(remaining, switch_rate)

    conv = np.clip(sample_param(params.get("conversion_after_switch", 0.0), n_iter), 0.0, 1.0)
    recovered = np.random.binomial(switched, conv)

    # AOV CLT
    mean_i = np.asarray(sample_param(params["aov_mean"], n_iter), dtype=float)
    std_i = np.asarray(sample_param(params["aov_std"], n_iter), dtype=float)
    mean_i = np.maximum(mean_i, 1e-9)
    std_i = np.maximum(std_i, 1e-9)

    mu, sigma, sigma2 = lognormal_params_from_mean_std(mean_i, std_i)
    var_one = (np.exp(sigma2) - 1.0) * np.exp(2 * mu + sigma2)

    n0 = np.maximum(successes0, 1)
    nr = np.maximum(recovered, 1)

    aov_bar0 = np.random.normal(loc=mean_i, scale=np.sqrt(var_one / n0))
    aov_barr = np.random.normal(loc=mean_i, scale=np.sqrt(var_one / nr))
    aov_bar0 = np.clip(aov_bar0, 0.0, None)
    aov_barr = np.clip(aov_barr, 0.0, None)
    if aov_cap is not None:
        aov_bar0 = np.minimum(aov_bar0, aov_cap)
        aov_barr = np.minimum(aov_barr, aov_cap)

    gross0 = successes0 * aov_bar0
    grossr = recovered * aov_barr
    gross = gross0 + grossr

    c_pct = np.asarray(sample_param(params["commission_percent"], n_iter), dtype=float)
    c_fix = np.asarray(sample_param(params["commission_fixed"], n_iter), dtype=float)
    sc_pct = np.asarray(sample_param(params.get("switch_commission_percent", params["commission_percent"]), n_iter), dtype=float)
    sc_fix = np.asarray(sample_param(params.get("switch_commission_fixed", params["commission_fixed"]), n_iter), dtype=float)

    comm0 = gross0 * c_pct + successes0 * c_fix
    commr = grossr * sc_pct + recovered * sc_fix
    commission = comm0 + commr

    churn_loss_per_order = np.asarray(sample_param(params.get("churn_loss_per_order", 0.0), n_iter), dtype=float)
    churn_loss = churn_loss_per_order * churned

    net = gross - commission - churn_loss
    return net


# ---------------------------
# UI: Data input
# ---------------------------
with st.expander("1) Load data", expanded=True):
    up = st.file_uploader("Upload CSV (utf-16 or utf-8). If empty, app will try ./monte.csv if any.", type=["csv"]) 
    if up is not None:
        data = up.read()
        try:
            df_raw = pd.read_csv(io.BytesIO(data), encoding="utf-16")
        except Exception:
            df_raw = pd.read_csv(io.BytesIO(data))
    else:
        df_raw = load_default_csv()
        if df_raw is None:
            st.warning("No file uploaded and ./monte.csv not found. Upload a CSV to continue.")
            st.stop()

    df = prepare_df(df_raw)
    st.caption("Preview after basic preprocessing (Online only, essential columns (random 100 rows)):")
    st.dataframe(df.sample(100), use_container_width=True)

# Stats
stats = get_payment_stats(df)
st.subheader("Payment stats (quarter, empirical)")
st.dataframe(stats, use_container_width=True)

pay_types = sorted(df["pay_type"].dropna().unique().tolist())
if not pay_types:
    st.error("No pay_type values found.")
    st.stop()

# ---------------------------
# Sidebar params
# ---------------------------
st.sidebar.header("Simulation settings")
N_ITER = st.sidebar.slider("Iterations", min_value=1000, max_value=50000, value=10000, step=1000)
TIGHT = st.sidebar.checkbox("Tight mode (fix orders & p_success to means for Custom)", value=False)
TRIM_P995 = st.sidebar.checkbox("Trim Custom hist at P99.5 for plot", value=False)

st.sidebar.header("Real PSP")
real_pay = st.sidebar.selectbox("pay_type (Real)", pay_types, index=0)
count_by_cart_id = st.sidebar.checkbox("Count orders by unique cart_id (if present)", value=False)
REAL_COMM_PCT = st.sidebar.number_input("commission_percent (Real)", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%0.3f")
REAL_COMM_FIX = st.sidebar.number_input("commission_fixed (Real)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
REAL_CHURN = st.sidebar.number_input("churn_rate_after_fail (Real)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
REAL_SWITCH = st.sidebar.number_input("switch_rate (Real)", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
REAL_CONV = st.sidebar.number_input("conversion_after_switch (Real)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
REAL_SWITCH_COMM_PCT = st.sidebar.number_input("switch_commission_percent (Real)", min_value=0.0, max_value=0.2, value=REAL_COMM_PCT, step=0.001, format="%0.3f")
REAL_SWITCH_COMM_FIX = st.sidebar.number_input("switch_commission_fixed (Real)", min_value=0.0, max_value=100.0, value=REAL_COMM_FIX, step=0.1)
REAL_CHURN_LOSS = st.sidebar.number_input("churn_loss_per_order (Real)", min_value=0.0, max_value=1e7, value=0.0, step=100.0)

emp_real = extract_real_psp_empirics(df, real_pay, count_by_cart_id=count_by_cart_id)

# ---------------------------
# Custom PSP controls
# ---------------------------
st.sidebar.header("Custom PSP")

succ_all = df[df["is_success"] == 1]["order_amount"].astype(float)
base_mean = float(succ_all.mean()) if succ_all.size else float(df["order_amount"].mean())
base_std = float(succ_all.std(ddof=1)) if succ_all.size else float(df["order_amount"].std(ddof=1))

# Small helper for scalar or range

def scalar_or_range(label, default_scalar, default_low=None, default_high=None, minv=None, maxv=None, step=None):
    use_range = st.sidebar.checkbox(f"Use range for {label}", value=(default_low is not None and default_high is not None))
    if use_range:
        c1, c2 = st.sidebar.columns(2)
        low = c1.number_input(f"{label} min", value=default_low if default_low is not None else default_scalar, min_value=minv, max_value=maxv, step=step)
        high = c2.number_input(f"{label} max", value=default_high if default_high is not None else default_scalar, min_value=minv, max_value=maxv, step=step)
        return (low, high)
    else:
        val = st.sidebar.number_input(label, value=default_scalar, min_value=minv, max_value=maxv, step=step)
        return val

orders_mu_ctrl = scalar_or_range("orders_mu", default_scalar=float(emp_real["orders_quarter"]), default_low=float(emp_real["orders_quarter"] * 0.95), default_high=float(emp_real["orders_quarter"] * 1.05), minv=0.0, maxv=None, step=1.0)
fail_rate_ctrl = scalar_or_range("fail_rate", default_scalar=1.0 - (emp_real["success_quarter"] / max(emp_real["orders_quarter"], 1)), default_low=0.03, default_high=0.08, minv=0.0, maxv=1.0, step=0.001)
aov_mean_ctrl = scalar_or_range("aov_mean (₽)", default_scalar=base_mean, default_low=base_mean * 0.98, default_high=base_mean * 1.02, minv=0.0, maxv=None, step=100.0)
aov_std_ctrl = scalar_or_range("aov_std (₽)", default_scalar=max(base_std, 1.0), default_low=None, default_high=None, minv=0.0, maxv=None, step=100.0)

commission_percent_ctrl = scalar_or_range("commission_percent", default_scalar=REAL_COMM_PCT, default_low=REAL_COMM_PCT, default_high=min(0.2, REAL_COMM_PCT + 0.01), minv=0.0, maxv=0.2, step=0.001)
commission_fixed_ctrl = scalar_or_range("commission_fixed", default_scalar=REAL_COMM_FIX, default_low=0.0, default_high=5.0, minv=0.0, maxv=100.0, step=0.1)

churn_rate_ctrl = scalar_or_range("churn_rate_after_fail", default_scalar=REAL_CHURN, default_low=0.2, default_high=0.5, minv=0.0, maxv=1.0, step=0.01)
switch_rate_ctrl = scalar_or_range("switch_rate", default_scalar=REAL_SWITCH, default_low=0.1, default_high=0.3, minv=0.0, maxv=1.0, step=0.01)
conv_after_switch_ctrl = scalar_or_range("conversion_after_switch", default_scalar=REAL_CONV, default_low=0.4, default_high=0.6, minv=0.0, maxv=1.0, step=0.01)

switch_commission_percent_ctrl = scalar_or_range("switch_commission_percent", default_scalar=REAL_SWITCH_COMM_PCT, default_low=None, default_high=None, minv=0.0, maxv=0.2, step=0.001)
switch_commission_fixed_ctrl = scalar_or_range("switch_commission_fixed", default_scalar=REAL_SWITCH_COMM_FIX, default_low=None, default_high=None, minv=0.0, maxv=100.0, step=0.1)

churn_loss_ctrl = scalar_or_range("churn_loss_per_order (₽)", default_scalar=0.0, default_low=None, default_high=None, minv=0.0, maxv=None, step=100.0)

params_custom = dict(
    orders_mu=orders_mu_ctrl,
    fail_rate=fail_rate_ctrl,
    aov_mean=aov_mean_ctrl,
    aov_std=aov_std_ctrl,
    commission_percent=commission_percent_ctrl,
    commission_fixed=commission_fixed_ctrl,
    churn_rate_after_fail=churn_rate_ctrl,
    switch_rate=switch_rate_ctrl,
    conversion_after_switch=conv_after_switch_ctrl,
    switch_commission_percent=switch_commission_percent_ctrl,
    switch_commission_fixed=switch_commission_fixed_ctrl,
    churn_loss_per_order=churn_loss_ctrl,
)

# ---------------------------
# Run simulation
# ---------------------------
run = st.button("Run simulations")
if run:
    with st.spinner("Simulating..."):
        net_real = simulate_real_psp(
            emp_real,
            commission_percent=REAL_COMM_PCT,
            commission_fixed=REAL_COMM_FIX,
            churn_rate_after_fail=REAL_CHURN,
            switch_rate=REAL_SWITCH,
            conversion_after_switch=REAL_CONV,
            switch_commission_percent=REAL_SWITCH_COMM_PCT,
            switch_commission_fixed=REAL_SWITCH_COMM_FIX,
            churn_loss_per_order=REAL_CHURN_LOSS,
            n_iter=N_ITER,
        )
        net_custom = simulate_custom_psp(params_custom, n_iter=N_ITER, aov_cap=None, tight=TIGHT)

    def summarize(name, arr):
        return dict(Метод=name, P10=np.percentile(arr, 10), Медиана=np.median(arr), P90=np.percentile(arr, 90), Среднее=np.mean(arr), Std=np.std(arr))

    summary_df = pd.DataFrame([summarize(f"Real: {real_pay}", net_real), summarize("Custom", net_custom)])
    p_custom_beats = float((net_custom > net_real).mean())

    st.subheader("Results")
    st.write(f"**P(Custom > Real)**: {p_custom_beats:.2%}  |  **P(Real > Custom)**: {(1.0 - p_custom_beats):.2%}")
    st.dataframe(summary_df, use_container_width=True)

    # Histograms
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=net_real, name=f"Real: {real_pay}", opacity=0.6))
        x_custom = net_custom
        if TRIM_P995:
            p995 = np.percentile(net_custom, 99.5)
            x_custom = net_custom[net_custom <= p995]
        fig.add_trace(go.Histogram(x=x_custom, name="Custom", opacity=0.6))
        fig.update_layout(barmode="overlay", title="Net Revenue per Quarter: Real vs Custom", xaxis_title="Net (₽)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        delta = net_custom - net_real
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=delta, name="ΔNet (Custom − Real)"))
        fig.update_layout(title="ΔNet distribution", xaxis_title="ΔNet (₽)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Sensitivity (optional)
    st.subheader("Sensitivity heatmaps (optional)")
    do_sens = st.checkbox("Run sensitivity heatmaps", value=False)
    if do_sens:
        iters_grid = st.number_input("Iterations per grid cell", min_value=1000, max_value=20000, value=3000, step=500)
        orders_grid = np.linspace(emp_real["orders_quarter"] * 0.6, emp_real["orders_quarter"] * 1.4, 8).astype(int)
        fail_grid = np.linspace(0.01, 0.20, 10)
        comm_grid = np.linspace(0.0, 0.05, 10)

        def real_sampler(n):
            return simulate_real_psp(
                emp_real,
                commission_percent=REAL_COMM_PCT,
                commission_fixed=REAL_COMM_FIX,
                churn_rate_after_fail=REAL_CHURN,
                switch_rate=REAL_SWITCH,
                conversion_after_switch=REAL_CONV,
                switch_commission_percent=REAL_SWITCH_COMM_PCT,
                switch_commission_fixed=REAL_SWITCH_COMM_FIX,
                churn_loss_per_order=REAL_CHURN_LOSS,
                n_iter=n,
            )

        def grid_sensitivity(custom_template, x_name, x_vals, y_name, y_vals, iters=4000):
            rows = []
            for xv in x_vals:
                for yv in y_vals:
                    params = custom_template.copy()
                    params[x_name] = xv
                    params[y_name] = yv
                    net_c = simulate_custom_psp(params, n_iter=iters, tight=TIGHT)
                    net_r = real_sampler(iters)
                    win = float((net_c > net_r).mean())
                    rows.append({x_name: xv, y_name: yv, "win_prob": win})
            return pd.DataFrame(rows)

        custom_base = params_custom.copy()
        # (a) fail_rate × orders_mu
        df_fail_ord = grid_sensitivity(custom_base, "fail_rate", fail_grid, "orders_mu", orders_grid, iters=iters_grid)
        fig = px.density_heatmap(df_fail_ord, x="fail_rate", y="orders_mu", z="win_prob", title="P(Custom > Real) vs fail_rate × orders", color_continuous_scale="Viridis")
        fig.update_layout(xaxis_title="fail_rate", yaxis_title="orders per quarter")
        st.plotly_chart(fig, use_container_width=True)

        # (b) commission_percent × orders_mu
        df_comm_ord = grid_sensitivity(custom_base, "commission_percent", comm_grid, "orders_mu", orders_grid, iters=iters_grid)
        fig = px.density_heatmap(df_comm_ord, x="commission_percent", y="orders_mu", z="win_prob", title="P(Custom > Real) vs commission_% × orders", color_continuous_scale="Viridis")
        fig.update_layout(xaxis_title="commission_percent", yaxis_title="orders per quarter")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Compare two real PSPs
# ---------------------------
st.subheader("Compare two Real PSPs")
colA, colB = st.columns(2)
with colA:
    pay_a = st.selectbox("PSP A", pay_types, key="payA")
    comm_a_pct = st.number_input("commission_percent A", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%0.3f")
    comm_a_fix = st.number_input("commission_fixed A", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with colB:
    pay_b = st.selectbox("PSP B", pay_types, index=min(1, len(pay_types)-1), key="payB")
    comm_b_pct = st.number_input("commission_percent B", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%0.3f")
    comm_b_fix = st.number_input("commission_fixed B", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

if st.button("Compare A vs B"):
    emp_a = extract_real_psp_empirics(df, pay_a)
    emp_b = extract_real_psp_empirics(df, pay_b)
    na = simulate_real_psp(emp_a, commission_percent=comm_a_pct, commission_fixed=comm_a_fix, n_iter=N_ITER)
    nb = simulate_real_psp(emp_b, commission_percent=comm_b_pct, commission_fixed=comm_b_fix, n_iter=N_ITER)
    pA = float((na > nb).mean())
    out = pd.DataFrame([
        dict(Метод=pay_a, P10=np.percentile(na, 10), Медиана=np.median(na), P90=np.percentile(na, 90), Среднее=np.mean(na), Std=np.std(na)),
        dict(Метод=pay_b, P10=np.percentile(nb, 10), Медиана=np.median(nb), P90=np.percentile(nb, 90), Среднее=np.mean(nb), Std=np.std(nb)),
    ])
    st.write(f"P({pay_a} > {pay_b}): {pA:.2%}  |  P({pay_b} > {pay_a}): {(1-pA):.2%}")
    st.dataframe(out, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=na, name=pay_a, opacity=0.6))
    fig.add_trace(go.Histogram(x=nb, name=pay_b, opacity=0.6))
    fig.update_layout(barmode="overlay", title="Real PSPs: Net per Quarter", xaxis_title="Net (₽)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
