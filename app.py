# -*- coding: utf-8 -*-
"""
MA(q) - Interactive Time Series Laboratory
Application pédagogique Streamlit sur les modèles de moyennes mobiles MA(q).
Lancement : streamlit run app.py
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:  # ACF empirique via statsmodels (repli sur un calcul manuel si absent)
    from statsmodels.tsa.stattools import acf as sm_acf
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------------------------------------------------------------- Palette
NAVY, BLUE, LIGHT, PALE, ORANGE, GREY = "#0B2A5B", "#2563EB", "#93C5FD", "#EFF6FF", "#F59E0B", "#64748B"

# ======================================================================
# 1. FONCTIONS MATHÉMATIQUES
# ======================================================================
def simulate_ma(n, mu, thetas, sigma, seed):
    """Simule X_t = mu + eps_t + theta_1 eps_(t-1) + ... + theta_q eps_(t-q).
    Même logique que le code initial : pour q=1, X = mu + eps[1:] + theta*eps[:-1].
    Retourne (X, eps_complet) avec eps_complet de longueur n+q."""
    thetas = np.asarray(thetas, dtype=float)
    q = len(thetas)
    eps = np.random.RandomState(seed).normal(0, sigma, n + q)   # graine fixée
    X = mu + eps[q:]                                            # terme eps_t
    for i, th in enumerate(thetas, start=1):                    # retard i : eps_(t-i)
        X = X + th * eps[q - i: n + q - i]
    return X, eps


def theoretical_acf(thetas, max_lag):
    """ACF théorique : gamma(k) = sum_{j=0}^{q-k} psi_j psi_(j+k), psi_0 = 1 ; rho = gamma/gamma(0)."""
    psi = np.r_[1.0, np.asarray(thetas, dtype=float)]
    q = len(psi) - 1
    gamma = np.array([np.dot(psi[: q - k + 1], psi[k:]) if k <= q else 0.0 for k in range(max_lag + 1)])
    return gamma / gamma[0]


def empirical_acf(x, nlags):
    """ACF empirique (statsmodels) et demi-largeur de la bande de confiance à 95 % (formule de Bartlett)."""
    x = np.asarray(x, dtype=float)
    if HAS_SM:
        rho = sm_acf(x, nlags=nlags, fft=True)
    else:
        xm = x - x.mean()
        rho = np.array([np.sum(xm[k:] * xm[: len(x) - k]) for k in range(nlags + 1)]) / np.sum(xm ** 2)
    se = np.zeros(nlags + 1)
    for k in range(1, nlags + 1):
        se[k] = np.sqrt((1 + 2 * np.sum(rho[1:k] ** 2)) / len(x))
    return rho, 1.96 * se


def compute_stats(X, mu, thetas, sigma):
    """Statistiques théoriques et empiriques."""
    var_th = sigma ** 2 * (1 + float(np.sum(np.square(thetas))))
    return {"mean_th": mu, "mean_emp": float(np.mean(X)),
            "var_th": var_th, "var_emp": float(np.var(X, ddof=1)),
            "std_th": float(np.sqrt(var_th)), "std_emp": float(np.std(X, ddof=1))}


def formula_latex(mu, thetas):
    """Formule LaTeX correspondant exactement aux paramètres choisis."""
    s = f"X_t = {mu:g} + \\varepsilon_t"
    for i, th in enumerate(thetas, start=1):
        sign = "+" if th >= 0 else "-"
        lag = "t" if i == 0 else f"t-{i}"
        s += f" {sign} {abs(th):g}\\,\\varepsilon_{{{lag}}}"
    return s


def explain(q, thetas, sigma):
    """Interprétations automatiques (courtes, utilisables à l'oral)."""
    th = list(thetas)
    out = []
    if q == 0:
        out.append("**MA(0)** : X_t = μ + ε_t est un simple bruit blanc décalé de μ. Aucune mémoire : l'ACF est nulle pour tout retard k ≥ 1.")
    else:
        out.append(f"Le modèle **MA({q})** dépend de l'innovation actuelle et de **{q} innovation{'s' if q > 1 else ''} passée{'s' if q > 1 else ''}**. "
                   f"L'ACF théorique est nulle pour les retards supérieurs à {q}.")
        facteur = 1 + float(np.sum(np.square(th)))
        out.append(f"**Variance** : Var(X_t) = σ²(1 + Σθ²) = {sigma:g}² × {facteur:.3f} = {sigma ** 2 * facteur:.3f}. "
                   "Plus les |θ| sont grands, plus la variance augmente, car chaque choc passé contribue davantage à X_t.")
        sgn = "positive" if th[0] > 0 else ("négative" if th[0] < 0 else "nulle")
        out.append(f"**Signe de θ₁** : l'autocorrélation au lag 1 est {sgn}. "
                   + ("Un choc positif tend à être suivi d'une valeur élevée." if th[0] > 0 else
                      "Un choc positif tend à être suivi d'une valeur plus faible (alternance)." if th[0] < 0 else ""))
    out.append("**Stationnarité** : espérance constante (μ), variance finie et constante, autocovariance dépendant seulement du retard. "
               "Toute combinaison linéaire finie d'un bruit blanc est donc stationnaire.")
    return out


# ======================================================================
# 2. FIGURES
# ======================================================================
def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_series(x, mu=None, title="", color=BLUE, label="Série"):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, color=color, lw=1.4, label=label)
    if mu is not None:
        ax.axhline(mu, color=ORANGE, ls="--", lw=1.6, label=f"Moyenne théorique μ = {mu:g}")
    _style(ax, title, "Temps", "Valeur"); ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def fig_acf(emp, band, theo, q):
    """Comparaison ACF empirique (avec bandes de confiance) / ACF théorique, retard q mis en évidence."""
    lags = np.arange(len(emp))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, vals, ttl, col in ((axes[0], emp, "ACF empirique (échantillon)", BLUE),
                               (axes[1], theo, "ACF théorique (formule)", NAVY)):
        ax.bar(lags, vals, width=0.12, color=col)
        ax.scatter(lags, vals, color=col, s=22, zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(q + 0.5, color=ORANGE, ls="--", lw=1.6)
        ax.text(q + 0.6, 0.92, f"coupure après q = {q}", color=ORANGE, fontsize=9, va="top")
        _style(ax, ttl, "Retard k")
    axes[0].fill_between(lags[1:], -band[1:], band[1:], color=LIGHT, alpha=0.5, label="Bande de confiance 95 %")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("Autocorrélation"); axes[0].set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def fig_structure(thetas):
    """Schéma : contributions des innovations passées et actuelle à X_t."""
    q = len(thetas)
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off"); ax.set_xlim(0, q + 2.2); ax.set_ylim(0, 3)
    for j in range(q + 1):                          # j = retard, de q (gauche) à 0 (droite)
        lag = q - j
        x = 0.6 + j
        label = "ε_t" if lag == 0 else f"ε_(t-{lag})"
        coef = "1" if lag == 0 else f"θ{lag}={thetas[lag - 1]:g}"
        ax.text(x, 2.3, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc=BLUE if lag else NAVY, ec="none"))
        ax.annotate("", xy=(q + 1.8, 0.8), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
        ax.text((x + q + 1.8) / 2, 1.5 + 0.1 * j, coef, color=ORANGE, fontsize=10, ha="center", fontweight="bold")
    ax.text(q + 1.8, 0.6, "X_t", ha="center", va="center", color="white", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=NAVY, ec="none"))
    return fig


# ======================================================================
# 3. INTERFACE : utilitaires
# ======================================================================
CSS = f"""
<style>
/* Base theme: explicit contrast, independent of browser dark/light preference */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:#F8FAFF !important; color:#172033 !important; }}
html, body, [data-testid="stMarkdownContainer"], p, li, label, span, div {{ color:inherit; }}
h1,h2,h3,h4,h5,h6 {{ color:{NAVY} !important; }}
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p {{ color:#334155 !important; }}
[data-testid="stSidebar"] {{ background:#F1F5F9 !important; }}
[data-testid="stSidebar"] * {{ color:#172033 !important; }}
.card {{ background:white !important; border:1px solid #DBEAFE; border-left:6px solid {BLUE}; border-radius:12px;
        padding:14px 18px; box-shadow:0 2px 8px rgba(11,42,91,.06); color:#172033 !important; }}
.card .t {{ color:{GREY} !important; font-size:.85rem; }} .card .v {{ color:{NAVY} !important; font-size:1.7rem; font-weight:700; }}
.hero {{ background:linear-gradient(120deg,{NAVY},{BLUE}); color:white !important; padding:22px 28px; border-radius:14px; }}
.hero h1 {{ color:white !important; margin:0; }} .hero p {{ color:#DBEAFE !important; margin:4px 0 0 0; }}
.say {{ background:{PALE}; border-radius:10px; padding:12px 16px; color:{NAVY} !important; }}
/* Mobile-friendly spacing and readable typography */
@media (max-width: 768px) {{
  .block-container {{ padding-left:1rem !important; padding-right:1rem !important; padding-top:1rem !important; }}
  h1 {{ font-size:1.65rem !important; line-height:1.25 !important; }}
  h2 {{ font-size:1.3rem !important; line-height:1.3 !important; }}
  h3 {{ font-size:1.12rem !important; }}
  p, li, label {{ font-size:1rem !important; line-height:1.65 !important; }}
  .hero {{ padding:18px 18px !important; border-radius:0 0 18px 18px !important; }}
  .card {{ padding:12px 14px !important; margin-bottom:8px !important; }}
  .card .v {{ font-size:1.35rem !important; }}
  [data-testid="stHorizontalBlock"] {{ gap:0.65rem !important; }}
  [data-testid="stPlotlyChart"], [data-testid="stImage"] {{ max-width:100% !important; }}
  [data-testid="stDataFrame"] {{ overflow-x:auto !important; }}
  button {{ min-height:44px !important; }}
}}
</style>"""


def card(title, value):
    return f'<div class="card"><div class="t">{title}</div><div class="v">{value}</div></div>'


def get_params():
    """Paramètres globaux lus dans la session (valeurs par défaut incluses)."""
    s = st.session_state
    q = int(s.get("q", 1))
    thetas = [float(s.get(f"th{i}", d)) for i, d in zip(range(1, 5), (0.5, 0.3, 0.2, 0.1))][:q]
    return dict(q=q, thetas=thetas, mu=float(s.get("mu", 10.0)), sigma=float(s.get("sigma", 1.0)),
                n=int(s.get("n", 500)), seed=int(s.get("seed", 42)))


def new_seed():
    st.session_state["seed"] = int(np.random.randint(0, 100000))


def sidebar_controls():
    st.sidebar.markdown("### ⚙️ Paramètres du modèle")
    st.sidebar.select_slider("Ordre q", options=[0, 1, 2, 3, 4], value=1, key="q", format_func=lambda v: f"MA({v})")
    q = st.session_state["q"]
    for i, d in zip(range(1, 5), (0.5, 0.3, 0.2, 0.1)):
        st.sidebar.slider(f"θ{i}", -1.5, 1.5, d, 0.05, key=f"th{i}", disabled=(i > q))
    st.sidebar.slider("Moyenne μ", -50.0, 50.0, 10.0, 0.5, key="mu")
    st.sidebar.slider("Écart-type σ du bruit blanc", 0.1, 5.0, 1.0, 0.1, key="sigma")
    st.sidebar.slider("Taille de l'échantillon n", 50, 5000, 500, 50, key="n")
    st.sidebar.number_input("Graine aléatoire", 0, 99999, 42, key="seed")
    st.sidebar.button("🎲 Générer une nouvelle simulation", on_click=new_seed, use_container_width=True)


def run(p):
    X, eps = simulate_ma(p["n"], p["mu"], p["thetas"], p["sigma"], p["seed"])
    return X, eps, compute_stats(X, p["mu"], p["thetas"], p["sigma"])


def understand(p):
    st.markdown("### 💡 Comprendre les résultats")
    for t in explain(p["q"], p["thetas"], p["sigma"]):
        st.markdown(f"- {t}")


# ======================================================================
# 4. PAGES
# ======================================================================
def page_intro():
    st.markdown('<div class="hero"><h1>MA(q) – Interactive Time Series Laboratory</h1>'
                '<p>Modèles de moyennes mobiles · Séries chronologiques · Master MLIA</p></div>', unsafe_allow_html=True)
    st.markdown("### Les séries chronologiques")
    st.write("Une série chronologique est une suite d'observations d'un même phénomène, ordonnées dans le temps. "
             "Le passé d'une série peut contenir de l'information utile pour comprendre et prévoir son évolution.")
    st.markdown("### Le modèle MA(q)")
    st.write("Un modèle de moyenne mobile d'ordre q exprime la valeur actuelle comme une combinaison linéaire de l'innovation actuelle "
             "et de ses q valeurs passées (bruit blanc).")
    st.latex(r"X_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}")
    c = st.columns(5)
    for col, (t, v) in zip(c, [("X_t", "observation actuelle"), ("μ", "moyenne théorique"), ("ε_t", "innovation actuelle"),
                               ("θ_i", "coefficient du modèle"), ("q", "ordre du modèle")]):
        col.markdown(card(t, f"<span style='font-size:1rem'>{v}</span>"), unsafe_allow_html=True)
    st.markdown("### Comment les innovations construisent X_t")
    q = st.radio("Ordre illustré", [1, 2, 3], horizontal=True, format_func=lambda v: f"MA({v})")
    st.pyplot(fig_structure([0.5, 0.3, 0.2][:q]))
    st.markdown("### MA(1) ou MA(2) ?")
    a, b = st.columns(2)
    a.markdown("**MA(1)**"); a.latex(r"X_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}")
    a.write("Une seule innovation passée. ACF non nulle au lag 1 seulement.")
    b.markdown("**MA(2)**"); b.latex(r"X_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\theta_2\varepsilon_{t-2}")
    b.write("Deux innovations passées. ACF non nulle aux lags 1 et 2, puis nulle.")


def page_simulation():
    p = get_params(); X, eps, s = run(p)
    st.title("Simulation interactive")
    st.caption("Modifiez les paramètres dans la barre latérale : tout se met à jour automatiquement.")
    st.latex(formula_latex(p["mu"], p["thetas"]))
    c = st.columns(4)
    for col, (t, v) in zip(c, [("Ordre", f"MA({p['q']})"), ("Taille n", p["n"]), ("σ", f"{p['sigma']:g}"), ("Graine", p["seed"])]):
        col.markdown(card(t, v), unsafe_allow_html=True)
    st.pyplot(fig_series(X, p["mu"], f"Série simulée MA({p['q']})"))
    with st.expander("Voir les 10 premières valeurs"):
        st.dataframe(pd.DataFrame({"t": range(1, 11), "ε_t": eps[p["q"]:][:10], "X_t": X[:10]}).round(4), hide_index=True)


def page_visualisation():
    p = get_params(); X, eps, s = run(p)
    st.title("Visualisation de la série temporelle")
    st.latex(formula_latex(p["mu"], p["thetas"]))
    st.pyplot(fig_series(eps[p["q"]:], 0, "Bruit blanc ε_t", GREY, "Bruit blanc"))
    st.pyplot(fig_series(X, p["mu"], f"Série MA({p['q']})"))
    st.markdown("### Explications")
    st.markdown("- **Pourquoi la série fluctue ?** Parce qu'elle est construite à partir de chocs aléatoires qui se renouvellent à chaque instant.\n"
                "- **Rôle du bruit blanc** : c'est la source d'aléa. Ses valeurs sont centrées, de variance constante et non corrélées.\n"
                "- **Rôle des coefficients** : θ_i règle le poids du choc survenu i périodes avant. Un |θ| plus grand donne plus de mémoire et plus de variance.\n"
                "- **Stationnarité** : la moyenne (μ) et la variance sont constantes et l'autocovariance ne dépend que du retard, sous les hypothèses du bruit blanc.")
    understand(p)


def page_stats():
    p = get_params(); X, eps, s = run(p)
    st.title("Propriétés statistiques")
    st.latex(r"E(X_t)=\mu \qquad Var(X_t)=\sigma^2\left(1+\theta_1^2+\dots+\theta_q^2\right)")
    c = st.columns(5)
    for col, (t, v) in zip(c, [("Moyenne théorique", f"{s['mean_th']:.3f}"), ("Moyenne empirique", f"{s['mean_emp']:.3f}"),
                               ("Variance théorique", f"{s['var_th']:.3f}"), ("Variance empirique", f"{s['var_emp']:.3f}"),
                               ("Écart-type (emp.)", f"{s['std_emp']:.3f}")]):
        col.markdown(card(t, v), unsafe_allow_html=True)
    df = pd.DataFrame({"Indicateur": ["Moyenne", "Variance", "Écart-type"],
                       "Théorique": [s["mean_th"], s["var_th"], s["std_th"]],
                       "Empirique": [s["mean_emp"], s["var_emp"], s["std_emp"]]})
    df["Écart absolu"] = (df["Théorique"] - df["Empirique"]).abs()
    st.markdown("### Théorie vs échantillon"); st.dataframe(df.round(4), hide_index=True, use_container_width=True)
    st.markdown("- **Moyenne** : E(ε)=0, donc E(X_t)=μ, quel que soit q.\n"
                "- **Variance** : les chocs sont non corrélés, donc leurs variances s'additionnent avec le poids θ_i².\n"
                "- **Écart théorique/empirique** : il diminue quand n augmente (loi des grands nombres).")
    understand(p)


def page_acf():
    p = get_params(); X, eps, s = run(p); q = p["q"]
    st.title("Autocorrélation : ACF")
    nl = st.slider("Nombre de retards affichés", 5, 40, 15)
    emp, band = empirical_acf(X, nl); theo = theoretical_acf(p["thetas"], nl)
    st.pyplot(fig_acf(emp, band, theo, q))
    st.info(f"**Propriété fondamentale** : pour un MA({q}), l'ACF **théorique** est nulle après le retard {q}. "
            f"Ici, la coupure théorique se produit après le lag {q}.")
    st.warning("**ACF empirique ≠ ACF théorique** : calculée sur un échantillon fini, l'ACF empirique garde de petites valeurs non nulles "
               "après q. Seules les barres sortant de la bande de confiance sont significatives.")
    df = pd.DataFrame({"Retard": range(min(nl, 8) + 1), "ACF théorique": theo[: min(nl, 8) + 1], "ACF empirique": emp[: min(nl, 8) + 1]})
    st.dataframe(df.round(4), hide_index=True, use_container_width=True)
    if q >= 1:
        st.caption(f"Formule : γ(k) = σ² Σ θ_(j+k) θ_j, avec θ₀ = 1 ; ρ(1) théorique = {theo[1]:.4f}.")
    understand(p)


def page_compare():
    p = get_params()
    st.title("Comparaison MA(1) vs MA(2)")
    st.caption("μ, σ, n et la graine viennent de la barre latérale ; les coefficients se règlent ici.")
    a, b = st.columns(2)
    t1 = a.slider("MA(1) : θ₁", -1.5, 1.5, 0.5, 0.05, key="c1")
    t2a = b.slider("MA(2) : θ₁", -1.5, 1.5, 0.5, 0.05, key="c2a")
    t2b = b.slider("MA(2) : θ₂", -1.5, 1.5, 0.3, 0.05, key="c2b")
    nl = 12
    for col, th, name in ((a, [t1], "MA(1)"), (b, [t2a, t2b], "MA(2)")):
        X, _ = simulate_ma(p["n"], p["mu"], th, p["sigma"], p["seed"])
        s = compute_stats(X, p["mu"], th, p["sigma"])
        with col:
            st.markdown(f"#### {name}"); st.latex(formula_latex(p["mu"], th))
            st.pyplot(fig_series(X, p["mu"], f"Série {name}"))
            st.markdown(card("Variance théorique / empirique", f"{s['var_th']:.3f} / {s['var_emp']:.3f}"), unsafe_allow_html=True)
            emp, band = empirical_acf(X, nl)
            st.pyplot(fig_acf(emp, band, theoretical_acf(th, nl), len(th)))
    st.markdown("### Différence")
    st.write("MA(1) retient un seul choc passé : ACF non nulle au lag 1 puis nulle. MA(2) retient deux chocs passés : ACF non nulle aux lags 1 et 2 puis nulle. "
             "Un ordre plus élevé donne une mémoire plus longue et, en général, une variance σ²(1+Σθ²) plus grande.")


def page_presentation():
    st.markdown('<div class="hero"><h1>Mode Présentation</h1><p>MA(1) · μ = 10 · θ₁ = 0.5 · σ = 1 · n = 500</p></div>', unsafe_allow_html=True)
    th = [0.5]; mu, sigma, n, seed = 10.0, 1.0, 500, 42
    X, eps = simulate_ma(n, mu, th, sigma, seed); s = compute_stats(X, mu, th, sigma)
    steps = ["1. Formule", "2. Série simulée", "3. Statistiques", "4. ACF", "5. Interprétation"]
    st.session_state.setdefault("step", 0)
    c1, c2, c3 = st.columns([1, 4, 1])
    if c1.button("◀ Précédent", disabled=st.session_state.step == 0):
        st.session_state.step -= 1; st.rerun()
    if c3.button("Suivant ▶", disabled=st.session_state.step == 4):
        st.session_state.step += 1; st.rerun()
    k = st.session_state.step
    c2.progress((k + 1) / 5, text=steps[k])
    if k == 0:
        st.latex(formula_latex(mu, th))
        st.markdown('<div class="say">À dire : le présent est la moyenne 10 plus le choc actuel plus la moitié du choc précédent.</div>', unsafe_allow_html=True)
    elif k == 1:
        st.pyplot(fig_series(X, mu, "Simulation d'un modèle MA(1)"))
        st.markdown('<div class="say">À dire : la série oscille autour de μ = 10 ; le bruit blanc ε_t alimente ces fluctuations.</div>', unsafe_allow_html=True)
    elif k == 2:
        cc = st.columns(4)
        for col, (t, v) in zip(cc, [("Moyenne th. / emp.", f"{s['mean_th']:.2f} / {s['mean_emp']:.2f}"),
                                    ("Variance th. / emp.", f"{s['var_th']:.2f} / {s['var_emp']:.2f}"),
                                    ("Écart-type emp.", f"{s['std_emp']:.2f}"), ("ρ(1) théorique", f"{theoretical_acf(th, 1)[1]:.2f}")]):
            col.markdown(card(t, v), unsafe_allow_html=True)
        st.markdown('<div class="say">À dire : Var(X_t) = σ²(1+θ²) = 1,25 ; l\'échantillon s\'en approche sans l\'égaler exactement.</div>', unsafe_allow_html=True)
    elif k == 3:
        emp, band = empirical_acf(X, 15)
        st.pyplot(fig_acf(emp, band, theoretical_acf(th, 15), 1))
        st.markdown('<div class="say">À dire : un pic au lag 1 (ρ(1) = θ/(1+θ²) = 0,4), puis coupure. Les petites barres suivantes sont du bruit d\'échantillonnage.</div>', unsafe_allow_html=True)
    else:
        st.success("Interprétation : la série est stationnaire, de moyenne 10. Sa mémoire est limitée à un seul choc passé, ce que révèle la coupure nette de l'ACF après le lag 1 : signature d'un MA(1).")


QUIZ = [
    ("Que représente q ?", ["Le nombre d'innovations passées utilisées", "La moyenne du processus", "La variance du bruit"], 0,
     "q est l'ordre du modèle : le nombre de chocs passés qui interviennent (q + 1 chocs avec l'actuel)."),
    ("Quelle est la différence entre MA(1) et MA(2) ?", ["MA(2) utilise deux chocs passés, MA(1) un seul", "MA(1) est non stationnaire", "Aucune"], 0,
     "L'ordre détermine la mémoire : l'ACF se coupe après le lag 1 pour MA(1), après le lag 2 pour MA(2)."),
    ("Quelle est la moyenne théorique d'un MA(q) ?", ["μ", "μ(1 + Σθ)", "0"], 0,
     "E(X_t) = μ, car le bruit blanc est centré : E(ε) = 0."),
    ("Comment calculer sa variance ?", ["σ²(1 + θ₁² + … + θq²)", "σ²(1 + θ₁ + … + θq)", "σ²"], 0,
     "Les chocs étant non corrélés, leurs variances s'additionnent pondérées par θ_i² (θ₀ = 1)."),
    ("Pourquoi l'ACF se coupe-t-elle après q ?", ["Au-delà de q, X_t et X_(t-k) n'ont plus aucun choc en commun", "Parce que la série est courte", "Parce que μ = 0"], 0,
     "Si k > q, les deux sommes de chocs ne partagent aucun terme : γ(k) = 0 donc ρ(k) = 0."),
    ("Quelle est la différence entre MA et AR ?", ["MA : mémoire des chocs ; AR : mémoire des valeurs passées", "MA est toujours non stationnaire", "AR n'utilise pas de bruit"], 0,
     "MA : ACF coupée après q. AR : PACF coupée après p, ACF à décroissance progressive."),
]


def page_quiz():
    st.title("Mini-quiz")
    score = 0
    for i, (q, opts0, ok0, expl) in enumerate(QUIZ):
        opts = random.Random(i).sample(opts0, len(opts0))   # ordre mélangé mais stable
        ok = opts.index(opts0[ok0])
        ans = st.radio(f"**{i + 1}. {q}**", opts, index=None, key=f"quiz{i}")
        if ans is not None:
            if opts.index(ans) == ok:
                score += 1; st.success(f"✅ Correct. {expl}")
            else:
                st.error(f"❌ Incorrect. Bonne réponse : **{opts[ok]}**. {expl}")
    st.markdown(card("Score", f"{score} / {len(QUIZ)}"), unsafe_allow_html=True)


# ======================================================================
# 5. POINT D'ENTRÉE
# ======================================================================
PAGES = {"1 · Introduction au MA(q)": page_intro, "2 · Simulation interactive": page_simulation,
         "3 · Visualisation": page_visualisation, "4 · Propriétés statistiques": page_stats,
         "5 · Autocorrélation (ACF)": page_acf, "6 · Comparaison MA(1) vs MA(2)": page_compare,
         "7 · Mode Présentation": page_presentation, "8 · Mini-quiz": page_quiz}
SIDEBAR_PAGES = {"2 · Simulation interactive", "3 · Visualisation", "4 · Propriétés statistiques",
                 "5 · Autocorrélation (ACF)", "6 · Comparaison MA(1) vs MA(2)"}


def main():
    st.set_page_config(page_title="MA(q) Laboratory", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 📈 MA(q) Laboratory")
    page = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    if page in SIDEBAR_PAGES:
        sidebar_controls()
    PAGES[page]()
    plt.close("all")


if __name__ == "__main__":
    main()
