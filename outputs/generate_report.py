"""
generate_report.py — Produce outputs/DCC_GARCH_Report.pdf from interpretation.md + figures.
"""

import os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register Unicode-capable TrueType fonts (Arial) ───────────────────────────
# ReportLab's built-in Helvetica/Courier only cover Latin-1.
# Arial has full Unicode: Greek, math symbols, dashes, primes, etc.
_FONTS = 'C:/Windows/Fonts/'
pdfmetrics.registerFont(TTFont('Arial',        _FONTS + 'arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold',   _FONTS + 'arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', _FONTS + 'ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', _FONTS + 'arialbi.ttf'))
pdfmetrics.registerFont(TTFont('CourierNew',   _FONTS + 'cour.ttf'))
pdfmetrics.registerFont(TTFont('CourierNew-Bold', _FONTS + 'courbd.ttf'))
pdfmetrics.registerFontFamily(
    'Arial',
    normal='Arial', bold='Arial-Bold',
    italic='Arial-Italic', boldItalic='Arial-BoldItalic',
)
pdfmetrics.registerFontFamily(
    'CourierNew',
    normal='CourierNew', bold='CourierNew-Bold',
    italic='CourierNew', boldItalic='CourierNew-Bold',
)

# ── Unicode sanitiser ──────────────────────────────────────────────────────────
# Arial covers Greek, common math, and typographic chars.
# The following characters are NOT in Arial and render as blank squares:
#   U+0304  combining macron  (Q̄, N̄ — combining diacritics never composite in PDF)
#   U+1D7CF mathematical bold 1 (𝟏)
#   U+2111  black-letter I (ℑ)
#   U+2299  circled dot (⊙)
#   U+207B  superscript minus (⁻)
#   U+2074/2076 superscript 4/6 (⁴ ⁶)
#   U+2080  subscript zero (₀)
# Replace them with safe HTML-tag or ASCII equivalents.

def _clean(s: str) -> str:
    return (s
        # Combining macron — replace the composed form Q̄/N̄
        .replace('Q\u0304', 'Q-bar')
        .replace('N\u0304', 'N-bar')
        # In case macron appears standalone after any other letter
        .replace('\u0304', '')
        # Mathematical bold / black-letter / special operators
        .replace('\u2111', '<i>F</i>')        # ℑ  → italic F  (information set)
        .replace('\U0001D7CF', '<b>1</b>')    # 𝟏  → bold 1   (indicator function)
        .replace('\u2299', '&#x2218;')        # ⊙  → ∘        (ring/Hadamard)
        # Superscript / subscript Unicode chars (not in Arial)
        .replace('\u207B', '<sup>-</sup>')    # ⁻
        .replace('\u2074', '<sup>4</sup>')    # ⁴
        .replace('\u2076', '<sup>6</sup>')    # ⁶
        .replace('\u2080', '<sub>0</sub>')    # ₀
    )

# Monkey-patch Paragraph so every call goes through _clean automatically
_Paragraph_real_init = Paragraph.__init__
def _para_init_clean(self, text, style=None, bulletText=None,
                     frags=None, caseSensitive=1, encoding='utf8'):
    if isinstance(text, str):
        text = _clean(text)
    _Paragraph_real_init(self, text, style, bulletText, frags, caseSensitive, encoding)
Paragraph.__init__ = _para_init_clean

def _clean_table_data(data):
    """Recursively clean strings inside table data lists."""
    return [
        [_clean(cell) if isinstance(cell, str) else cell for cell in row]
        for row in data
    ]

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'DCC_GARCH_Report_Final.pdf')

def fig(name):
    return os.path.join(HERE, name)

# ── Styles ────────────────────────────────────────────────────────────────────

W, H   = A4
MARGIN = 2.2 * cm
TW     = W - 2 * MARGIN   # text width

base   = getSampleStyleSheet()

TITLE   = ParagraphStyle('TITLE',   parent=base['Title'],
                         fontName='Arial-Bold',
                         fontSize=22, leading=28, spaceAfter=6,
                         textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER)
SUBTITLE= ParagraphStyle('SUBTITLE',parent=base['Normal'],
                         fontName='Arial',
                         fontSize=11, leading=15, spaceAfter=18,
                         textColor=colors.HexColor('#444444'), alignment=TA_CENTER)
H1      = ParagraphStyle('H1',      parent=base['Heading1'],
                         fontName='Arial-Bold',
                         fontSize=14, leading=18, spaceBefore=18, spaceAfter=6,
                         textColor=colors.HexColor('#1a1a2e'),
                         borderPad=2)
H2      = ParagraphStyle('H2',      parent=base['Heading2'],
                         fontName='Arial-Bold',
                         fontSize=11, leading=15, spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor('#2d4a8a'))
BODY    = ParagraphStyle('BODY',    parent=base['Normal'],
                         fontName='Arial',
                         fontSize=9.5, leading=14, spaceAfter=6,
                         alignment=TA_JUSTIFY)
BULLET  = ParagraphStyle('BULLET',  parent=BODY,
                         fontName='Arial',
                         leftIndent=14, bulletIndent=0, spaceAfter=3)
CAPTION = ParagraphStyle('CAPTION', parent=base['Normal'],
                         fontName='Arial-Italic',
                         fontSize=8, leading=11, spaceAfter=10,
                         textColor=colors.HexColor('#666666'), alignment=TA_CENTER)
CODE    = ParagraphStyle('CODE',    parent=base['Normal'],
                         fontName='CourierNew',
                         fontSize=8.5, leading=13, spaceAfter=6,
                         backColor=colors.HexColor('#f4f4f4'),
                         borderColor=colors.HexColor('#cccccc'), borderPad=5)
VERDICT = ParagraphStyle('VERDICT', parent=BODY,
                         fontName='Arial',
                         backColor=colors.HexColor('#eef2ff'),
                         borderColor=colors.HexColor('#2d4a8a'),
                         borderPad=6, borderWidth=1,
                         leading=14, spaceAfter=8)

def HR():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc'),
                      spaceAfter=8, spaceBefore=4)

def full_fig(filename, caption, width=None, height=None):
    w = width or TW
    items = [
        Image(fig(filename), width=w, height=height or w * 0.55),
        Paragraph(caption, CAPTION),
        Spacer(1, 6),
    ]
    return items

def table(data, col_widths, header_bg='#2d4a8a'):
    data = _clean_table_data(data)
    style = TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0),  colors.HexColor(header_bg)),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',     (0, 0), (-1, 0),  'Arial-Bold'),
        ('FONTNAME',     (0, 1), (-1, -1), 'Arial'),
        ('FONTSIZE',     (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),   [colors.white, colors.HexColor('#f7f7f7')]),
        ('GRID',         (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
        ('ALIGN',        (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN',        (0, 0), (0, -1),  'LEFT'),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
    ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(style)
    return t

# ── Build story ───────────────────────────────────────────────────────────────

story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 3 * cm),
    Paragraph('DCC / ADCC-GARCH Analysis', TITLE),
    Paragraph('Model Theory, Implementation, and Empirical Results', SUBTITLE),
    HR(),
    Spacer(1, 0.4 * cm),
    Paragraph('10 US Equity Sector ETFs &nbsp;|&nbsp; 1998–2026 &nbsp;|&nbsp; T = 7,099 daily observations', SUBTITLE),
    Paragraph('Models: DCC (Engle 2002) &nbsp;&amp;&nbsp; ADCC (Cappiello, Engle &amp; Sheppard 2006)', SUBTITLE),
    Spacer(1, 1.5 * cm),
]

toc_data = [
    ['Section', 'Title'],
    ['—',  'Executive Summary'],
    ['1',  'Theoretical Background'],
    ['2',  'Model Specification and Implementation'],
    ['3',  'Estimation Results and Diagnostics'],
    ['4',  'Data'],
    ['5',  'Figure-by-Figure Interpretation'],
    ['6',  'Key Takeaways'],
    ['7',  'Out-of-Sample Forecast Evaluation'],
    ['8',  'Live Forecasting Architecture'],
]
story += [
    table(toc_data, [TW*0.12, TW*0.88]),
    Spacer(1, 1 * cm),
]

# ════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    Paragraph('Executive Summary', H1), HR(),
    Spacer(1, 0.2*cm),
]

story += [
    Paragraph('Objective', H2),
    Paragraph(
        'This project implements the Dynamic Conditional Correlation GARCH (DCC-GARCH) model '
        'of Engle (2002) and its asymmetric extension (ADCC) of Cappiello, Engle &amp; Sheppard '
        '(2006) from first principles in Python. The implementation was developed by '
        'reverse-engineering the reference R package rmgarch, validating every equation against '
        'the original academic papers, and resolving a set of documented divergences between the '
        'two. The result is a production-grade library deployed as a daily live forecasting system '
        'for US equity sector correlations and covariances.', BODY),

    Paragraph('Data', H2),
    Paragraph(
        'The study universe consists of 10 US equity sector ETFs from the SPDR family, covering '
        'the full S&amp;P 500 sector decomposition: SPY (S&amp;P 500 broad market), XLB (Materials), '
        'XLE (Energy), XLF (Financials), XLI (Industrials), XLK (Technology), XLP (Consumer '
        'Staples), XLU (Utilities), XLV (Health Care), and XLY (Consumer Discretionary). '
        'Daily closing prices span 1998–2026 (T = 7,099 observations), covering three major '
        'stress regimes: the dot-com bust (2000–2002), the Global Financial Crisis (2007–2009), '
        'and the COVID crash (2020). Log returns are computed as r<sub>t</sub> = '
        '100 × log(P<sub>t</sub>/P<sub>t-1</sub>) and expressed in percent-daily units throughout.', BODY),

    Paragraph('Methodology', H2),
    Paragraph(
        'Estimation follows the standard DCC two-stage procedure. In Stage 1, a '
        'GJR-GARCH(1,1,1) model with Student-t errors is fitted independently to each of the '
        '10 return series. GJR-GARCH was selected to capture the well-known leverage effect '
        '(asymmetric volatility response to negative vs. positive shocks). The resulting '
        'standardised residuals z<sub>i,t</sub> = r<sub>i,t</sub>/σ<sub>i,t</sub> are passed '
        'to Stage 2, which estimates the two DCC parameters (a, b) — or three ADCC parameters '
        '(a, b, g) — by maximising the Gaussian quasi-log-likelihood of the correlation '
        'component. The stationarity constraints (a + b &lt; 1 for DCC; a + b + δg &lt; 1 for '
        'ADCC) are enforced as hard inequalities via the SLSQP optimiser.', BODY),

    Paragraph('Key Results', H2),
]

exec_results = [
    ['Metric',                          'DCC',        'ADCC'],
    ['a (ARCH)',                         '0.0209',     '0.0183'],
    ['b (GARCH)',                        '0.9748',     '0.9699'],
    ['g (asymmetry)',                    '—',          '0.0138'],
    ['Log-likelihood',                   '−71,756.8',  '−71,559.6'],
    ['LR test (ADCC vs DCC)',            '—',          '394  (p ≈ 0)'],
    ['OOS mean QLIKE (test 2020–2026)', '−8.074',     '−7.935'],
    ['OOS winner (DM test)',             'DCC',        '—'],
]
story += [
    table(exec_results, [TW*0.52, TW*0.24, TW*0.24]),
    Spacer(1, 0.3*cm),

    Paragraph(
        'In-sample, ADCC is decisively superior: the likelihood ratio statistic of 394 on 1 '
        'degree of freedom (p ≈ 0) confirms that joint negative shocks drive correlations '
        'above the symmetric DCC baseline, consistent with the leverage effect in correlations '
        'documented by Cappiello et al. (2006). Both AIC and BIC favour ADCC.', BODY),
    Paragraph(
        'Out-of-sample (2020–2026 test set), the ranking reverses: DCC produces significantly '
        'lower QLIKE loss (Diebold-Mariano statistic = −13.3, p ≈ 0). The asymmetric '
        'parameter g, calibrated on GFC and COVID crisis data in the training set, '
        'over-estimates correlations during the subsequent calm regime. ADCC only outperforms '
        'during the 68-day COVID crash window; DCC dominates the remaining 1,352 days.', BODY),

    Paragraph('Practical Recommendation', H2),
    Paragraph(
        '<b>Use DCC for unconditional daily covariance forecasting.</b> '
        'Use ADCC conditionally, only when a crisis or elevated-stress regime is identified '
        'in real time (e.g., via a VIX threshold or drawdown indicator). This is a '
        'textbook bias-variance tradeoff: ADCC captures a genuine structural feature of '
        'equity correlations but pays a variance penalty in calm markets that '
        'outweighs its benefit over the full out-of-sample period.', VERDICT),

    Paragraph('Live System', H2),
    Paragraph(
        'A daily live forecasting pipeline was built and validated end-to-end. It downloads '
        'prices from Yahoo Finance after each US market close, runs the GJR-GARCH filter per '
        'asset (fixed parameters, ~milliseconds), executes one DCC recursion step to produce '
        'H<sub>t+1|t</sub> (the next-day covariance forecast), evaluates the prior day\'s '
        'forecast via QLIKE, and writes all outputs to a local SQLite database. A Streamlit '
        'dashboard visualises correlation dynamics, conditional volatilities, and forecast '
        'accuracy history back to 1999. Parameters are re-estimated monthly on a rolling '
        '5-year window.', BODY),
]

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THEORETICAL BACKGROUND
# ════════════════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    Paragraph('1. Theoretical Background', H1), HR(),
]

story += [
    Paragraph('1.1  The Multivariate Covariance Problem', H2),
    Paragraph(
        'Modelling the joint second moments of a vector of financial returns is a fundamental '
        'challenge in empirical finance. Let r<sub>t</sub> = (r<sub>1,t</sub>, …, r<sub>k,t</sub>)′ '
        'denote a k-dimensional return vector with conditional covariance matrix H<sub>t</sub> = '
        'E[r<sub>t</sub>r<sub>t</sub>′ | ℑ<sub>t−1</sub>]. Accurate estimates of H<sub>t</sub> are '
        'required for portfolio optimisation, risk management (VaR), derivative pricing, and '
        'hedging. The central difficulty is that H<sub>t</sub> has k(k+1)/2 unique elements — '
        'for k = 10 assets that is 55 time-varying quantities to track simultaneously.', BODY),
    Paragraph(
        'Classical approaches suffer from well-known limitations. Rolling historical correlations '
        'assign equal weight to all observations in a fixed window and zero weight outside it. '
        'Exponential smoothers (e.g., RiskMetrics™) fix the decay parameter externally and use '
        'the same λ for all asset pairs. Orthogonal GARCH projects returns onto principal '
        'components and assumes conditional correlations among the components are zero — a '
        'strong restriction. Full multivariate GARCH specifications (BEKK, vec) can guarantee '
        'positive definiteness but require O(k²) or O(k⁴) parameters, making them '
        'computationally infeasible for k > 5.', BODY),
    Paragraph(
        'Engle (2002) proposed the Dynamic Conditional Correlation (DCC) model to fill this gap: '
        'a specification with the flexibility of univariate GARCH, the parsimony of only 2 '
        'additional scalar parameters regardless of k, and a guaranteed positive-definite '
        'covariance matrix at every point in time.', BODY),
]

story += [
    Paragraph('1.2  The DCC Model — Engle (2002)', H2),
    Paragraph(
        'The DCC model decomposes the conditional covariance matrix as:', BODY),
    Paragraph('H<sub>t</sub> = D<sub>t</sub> R<sub>t</sub> D<sub>t</sub>', CODE),
    Paragraph(
        'where D<sub>t</sub> = diag(σ<sub>1,t</sub>, …, σ<sub>k,t</sub>) is the diagonal matrix '
        'of time-varying conditional standard deviations from univariate GARCH models, and '
        'R<sub>t</sub> is the time-varying conditional correlation matrix. The key insight is '
        'that D<sub>t</sub> and R<sub>t</sub> are estimated in two separate stages: the first '
        'is a series of univariate GARCH fits (one per asset), and the second is the correlation '
        'dynamics — requiring only 2 parameters regardless of dimension.', BODY),
    Paragraph('<b>Stage 1 — Univariate GARCH.</b> Each asset i is modelled independently:', BODY),
    Paragraph(
        'r<sub>i,t</sub> = σ<sub>i,t</sub> ε<sub>i,t</sub>,   '
        'σ²<sub>i,t</sub> = ω<sub>i</sub> + α<sub>i</sub> r²<sub>i,t−1</sub> + '
        'β<sub>i</sub> σ²<sub>i,t−1</sub>', CODE),
    Paragraph(
        'This produces standardised residuals z<sub>i,t</sub> = r<sub>i,t</sub> / σ<sub>i,t</sub>, '
        'which by construction have mean zero and variance one. These are the inputs to Stage 2.', BODY),
    Paragraph('<b>Stage 2 — Correlation dynamics.</b> Define the pseudo-correlation matrix '
              'Q<sub>t</sub> via the recursion:', BODY),
    Paragraph(
        'Q<sub>t</sub> = (1 − a − b) Q̄  +  a z<sub>t−1</sub>z′<sub>t−1</sub>  +  b Q<sub>t−1</sub>', CODE),
    Paragraph(
        'where Q̄ = T⁻¹ Σ z<sub>t</sub>z′<sub>t</sub> is the unconditional covariance of the '
        'standardised residuals (estimated from data), and a, b ≥ 0 with a + b < 1 are the '
        'two scalar DCC parameters. The first term is the mean-reversion target, the second '
        'is the ARCH term (updating Q toward the outer product of the most recent shock), and '
        'the third is the GARCH persistence term.', BODY),
    Paragraph(
        'The correlation matrix R<sub>t</sub> is recovered by normalising Q<sub>t</sub>:', BODY),
    Paragraph(
        'R<sub>t</sub>[i,j] = Q<sub>t</sub>[i,j] / √(Q<sub>t</sub>[i,i] · Q<sub>t</sub>[j,j])', CODE),
    Paragraph(
        'This normalisation guarantees R<sub>t</sub> is a valid correlation matrix (ones on '
        'the diagonal, all off-diagonal elements in [−1, 1]) as long as Q<sub>t</sub> is '
        'positive definite — which is ensured by the stationarity constraint a + b < 1 '
        '(the recursion is a convex combination of a positive definite matrix Q̄ and positive '
        'semidefinite outer products).', BODY),
    Paragraph('<b>Two-stage log-likelihood.</b> The full log-likelihood decomposes as:', BODY),
    Paragraph(
        'L(θ, φ) = L<sub>V</sub>(θ) + L<sub>C</sub>(φ | θ)', CODE),
    Paragraph(
        'where L<sub>V</sub>(θ) = −½ Σ<sub>t</sub> Σ<sub>i</sub> [log σ²<sub>i,t</sub> + '
        'r²<sub>i,t</sub>/σ²<sub>i,t</sub>] is the sum of univariate GARCH log-likelihoods, '
        'and the correlation component is:', BODY),
    Paragraph(
        'L<sub>C</sub>(φ | θ) = −½ Σ<sub>t</sub> [log|R<sub>t</sub>| + z′<sub>t</sub> '
        'R<sup>−1</sup><sub>t</sub> z<sub>t</sub> − z′<sub>t</sub> z<sub>t</sub>]', CODE),
    Paragraph(
        'The difference z′<sub>t</sub>R<sup>−1</sup><sub>t</sub>z<sub>t</sub> − '
        'z′<sub>t</sub>z<sub>t</sub> captures the gain (or loss) from modelling correlation '
        'relative to the identity baseline. Engle (2002) shows that maximising these two '
        'components separately yields consistent but not fully efficient estimates — a '
        'standard two-step quasi-MLE result.', BODY),
]

story += [
    Paragraph('1.3  The ADCC Extension — Cappiello, Engle &amp; Sheppard (2006)', H2),
    Paragraph(
        'A well-documented stylised fact in equity markets is that correlations increase more '
        'after simultaneous negative shocks than after positive shocks of equal magnitude. '
        'Cappiello et al. (2006) formalise this in the Asymmetric DCC (ADCC) model, which '
        'extends the DCC correlation recursion to include an asymmetric innovation term.', BODY),
    Paragraph(
        'Define the negative-shock indicator n<sub>t</sub> = z<sub>t</sub> ⊙ 𝟏[z<sub>t</sub> < 0], '
        'where ⊙ denotes element-wise multiplication. The ADCC recursion is:', BODY),
    Paragraph(
        'Q<sub>t</sub> = [(1−a−b) Q̄ − g N̄]  +  a z<sub>t−1</sub>z′<sub>t−1</sub>  '
        '+  g n<sub>t−1</sub>n′<sub>t−1</sub>  +  b Q<sub>t−1</sub>', CODE),
    Paragraph(
        'where N̄ = E[n<sub>t</sub>n′<sub>t</sub>] is the unconditional second moment of '
        'the negative innovations (estimated as T⁻¹ Σ n<sub>t</sub>n′<sub>t</sub>, '
        'uncentred). The intercept term [(1−a−b) Q̄ − g N̄] is adjusted to ensure '
        'E[Q<sub>t</sub>] = Q̄ unconditionally — a correction the paper makes explicit.', BODY),
    Paragraph(
        '<b>Economic interpretation:</b> when both assets simultaneously receive negative '
        'shocks (z<sub>i,t</sub> < 0 and z<sub>j,t</sub> < 0), the asymmetric term '
        'g n<sub>i,t</sub>n<sub>j,t</sub> adds a positive increment to Q<sub>t</sub>[i,j]. '
        'This produces a larger correlation spike during joint sell-offs — precisely the '
        'regime risk that diversified portfolios are most exposed to. '
        'Cappiello et al. document this phenomenon extensively in international equity and '
        'bond returns, providing the motivating empirical evidence for the g parameter.', BODY),
    Paragraph(
        '<b>Stationarity constraint.</b> The ADCC model requires a stronger stationarity '
        'condition than the DCC. Define δ as the largest eigenvalue of '
        'Q̄<sup>−½</sup> N̄ Q̄<sup>−½</sup>. The constraint is:', BODY),
    Paragraph('a + b + δ · g < 1', CODE),
    Paragraph(
        'This reduces to the DCC constraint a + b < 1 when g = 0. The scalar δ scales '
        'the asymmetric contribution relative to the long-run correlation structure.', BODY),
    Paragraph(
        '<b>The scalar ADCC.</b> The full model of Cappiello et al. uses asset-specific '
        'parameter matrices A, B, G (diagonal). The scalar ADCC implemented here sets '
        'A = aI, B = bI, G = gI — identical news impact across all pairs. This is the '
        'model of choice for moderate-dimensional systems (k = 10) as it preserves '
        'parsimony and allows a clean comparison with the symmetric DCC.', BODY),
]

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL SPECIFICATION AND IMPLEMENTATION
# ════════════════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    Paragraph('2. Model Specification and Implementation', H1), HR(),
]

story += [
    Paragraph('2.1  Univariate GARCH Specification', H2),
    Paragraph(
        'For the univariate Stage 1, GJR-GARCH(1,1,1) with Student-t errors was selected for '
        'each of the 10 assets. The GJR model (Glosten, Jagannathan &amp; Runkle, 1993) extends '
        'standard GARCH by adding a threshold term that allows negative innovations to have a '
        'larger impact on variance than positive ones:', BODY),
    Paragraph(
        'σ²<sub>t</sub> = ω + α ε²<sub>t−1</sub> + γ ε²<sub>t−1</sub> 𝟏[ε<sub>t−1</sub> < 0] + β σ²<sub>t−1</sub>', CODE),
    Paragraph(
        'The γ parameter captures the <i>leverage effect</i>: after a negative return shock, '
        'variance increases by α + γ rather than just α. The Student-t distribution handles '
        'the excess kurtosis present in all 10 return series. The full specification as '
        'implemented uses the arch library:', BODY),
    Paragraph(
        "arch_model(returns × 100, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')", CODE),
    Paragraph(
        'Returns are scaled by 100 internally so that all quantities (sigmas, residuals) are '
        'in percent-daily units throughout the pipeline. This avoids numerical scaling issues '
        'in the optimiser and produces σ values in the interpretable range 0.5–3% per day.', BODY),

    Paragraph('2.2  DCC Layer: Order and Parameterisation', H2),
    Paragraph(
        'DCC(1,1) was chosen — one lag of the shock outer product and one lag of Q<sub>t</sub>. '
        'This is the standard specification in the literature and was shown by Engle (2002) '
        'to capture the bulk of correlation dynamics in daily financial returns. '
        'Higher-order DCC(p,q) models would add parameters without meaningful gain given the '
        'high persistence already implied by b ≈ 0.97.', BODY),
    Paragraph(
        'Estimation uses Gaussian QML (Normal quasi-maximum likelihood). Although the '
        'standardised residuals exhibit excess kurtosis of 4–6 (confirmed empirically), '
        'Gaussian QML remains consistent under mild regularity conditions regardless of the '
        'true error distribution. This is the standard approach in academic DCC estimation '
        'and is the method used in the reference R package rmgarch.', BODY),

    Paragraph('2.3  Library Architecture and Design Choices', H2),
    Paragraph(
        'The DCC layer was implemented as a standalone Python library (python/dcc/) '
        'reverse-engineered from the rmgarch R package and validated against the original '
        'papers. Key design invariants:', BODY),
    Paragraph('• <b>Single recursion function.</b> <i>_update_Q(Q_prev, z_prev, n_prev, AUQ, a, b, g, model)</i> '
              'is the sole implementation of the Q<sub>t</sub> recursion. It is called identically '
              'by the optimiser objective and the post-convergence path reconstruction. '
              'No duplication of the core equation.', BULLET),
    Paragraph('• <b>Pre-computed intercept.</b> The intercept AUQ = (1−a−b)·Q̄ [− g·N̄] is '
              'computed once per optimiser call, outside the time loop. This avoids T × k² '
              'redundant matrix additions and makes the per-step cost a single scalar broadcast.', BULLET),
    Paragraph('• <b>Cholesky everywhere.</b> Both log|R<sub>t</sub>| and R<sub>t</sub>⁻¹z<sub>t</sub> '
              'are computed via Cholesky decomposition rather than direct LU inversion. '
              'This is numerically more stable and exploits the positive definiteness of R<sub>t</sub>. '
              'If Cholesky fails (non-PD Q<sub>t</sub>), the objective returns a stateless '
              'penalty constant (1×10⁶) rather than NaN-propagating through the optimiser.', BULLET),
    Paragraph('• <b>Hard constraints.</b> The stationarity constraints are enforced as hard '
              'inequality constraints via SciPy SLSQP, not as soft penalties. This guarantees '
              'feasibility at the solution and avoids the gradient-discontinuity issues that '
              'arise with penalty-based approaches.', BULLET),
    Paragraph('• <b>Symmetry enforcement.</b> Q<sub>t</sub> is symmetrised at each step as '
              '(Q + Q′)/2 to suppress floating-point drift that would otherwise accumulate '
              'over T = 7,099 iterations.', BULLET),

    Paragraph('2.4  Key Deviations from rmgarch', H2),
    Paragraph(
        'The reverse-engineering process identified several divergences between the R package '
        'and the academic papers. The following corrections were made intentionally:', BODY),
]

dev_data = [
    ['ID', 'Item', 'rmgarch behaviour', 'This library'],
    ['DEV-02/03', 'Q_t initialisation',      'Inconsistent (zero-pad / ones-pad)', 'Q[0] = Q̄  (clean)'],
    ['DEV-04',   'Q_t symmetry',             'Not enforced',                        '(Q + Q′)/2 at each step'],
    ['DEV-05',   'R_t⁻¹ z_t',               'LU inversion',                        'Cholesky forward-substitution'],
    ['DEV-06',   'log|R_t|',                 'log(det(R))',                          '2·Σ log(diag(L)) via Cholesky'],
    ['DEV-07',   'Cholesky failure',          'No guard (NaN propagation)',           'Stateless PENALTY = 1e6'],
    ['DEV-08',   'Constraint enforcement',   'Soft penalty (nlminb path)',           'Hard inequality (SLSQP)'],
    ['DEV-01b',  'N̄ estimator',              'cov() — centred',                     'T⁻¹ N′N — uncentred (theory-faithful)'],
]
story += [
    table(dev_data, [TW*0.12, TW*0.22, TW*0.33, TW*0.33]),
    Spacer(1, 0.3*cm),
    Paragraph(
        'The most important theoretical correction is DEV-01b: rmgarch computes N̄ using a '
        'centred covariance estimator, which breaks the intercept correction that ensures '
        'E[Q<sub>t</sub>] = Q̄. The uncentred estimator T⁻¹ Σ n<sub>t</sub>n′<sub>t</sub> '
        'is what Cappiello et al. (2006) specify, and is what preserves the unconditional '
        'mean reversion property of the model.', BODY),
    Paragraph('2.5  Module Structure', H2),
]
module_data = [
    ['File',                   'Role'],
    ['python/dcc/utils.py',    'Pre-computation: estimate_Qbar, make_N_matrix, estimate_Nbar, compute_delta'],
    ['python/dcc/model.py',    '_update_Q (recursion), compute_Q, compute_R, dcc_objective, loglikelihood'],
    ['python/dcc/optimizer.py','dcc_constraint, adcc_constraint, fit() entry point (SLSQP)'],
    ['python/dcc/validate.py', 'Five-level validation suite: shapes, PD, diag=1, R symmetry, LLH match'],
    ['python/garch/gjr_garch.py','fit_gjr_garch, filter_gjr_garch, fit_multivariate_gjr'],
    ['project/pipeline.py',    'run_pipeline(returns, model) — end-to-end GJR + DCC in one call'],
    ['live/daily_run.py',      'Bootstrap / daily update / monthly refit orchestrator'],
    ['live/dashboard.py',      'Streamlit dashboard (reads DB only, no model code)'],
]
story += [
    table(module_data, [TW*0.32, TW*0.68]),
    Spacer(1, 0.3*cm),
]

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ESTIMATION RESULTS AND DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    Paragraph('3. Estimation Results and Diagnostics', H1), HR(),
    Paragraph('3.1  Estimated Parameters', H2),
]

est_data = [
    ['Parameter',          'DCC',       'ADCC',      'Interpretation'],
    ['a (ARCH)',           '0.020856',  '0.018328',  'Speed of response to shocks'],
    ['b (GARCH)',          '0.974774',  '0.969925',  'Persistence of correlations'],
    ['g (asymmetry)',      '—',         '0.013819',  'Extra response to joint neg. shocks'],
    ['a + b',             '0.995630',  '0.988253',  'Total persistence'],
    ['a + b + δg',        '0.995630',  '0.999999',  'Stationarity constraint value'],
    ['Log-likelihood',    '−71,756.8', '−71,559.6', 'DCC: 2 params; ADCC: 3 params'],
    ['AIC',               '143,517.6', '143,125.2', 'Lower is better'],
    ['BIC',               '143,531.3', '143,145.8', 'Lower is better'],
]
story += [
    table(est_data, [TW*0.17, TW*0.14, TW*0.14, TW*0.55]),
    Spacer(1, 0.3*cm),
    Paragraph('All parameters satisfy their theoretical constraints. Both models converged '
              'without issues from the default starting values (a₀ = 0.05, b₀ = 0.90).', BODY),

    Paragraph('3.2  Parameter Interpretation', H2),
    Paragraph(
        '<b>a (ARCH term):</b> very small (≈ 0.02). A single day\'s shock contributes only '
        '2% weight to the next period\'s correlation matrix. This reflects the slow-moving '
        'nature of cross-asset correlations compared to individual volatilities.', BODY),
    Paragraph(
        '<b>b (GARCH term):</b> very large (≈ 0.97). Yesterday\'s correlation matrix is the '
        'dominant predictor of today\'s. The model is essentially a first-order autoregressive '
        'filter on correlations with a near-unit root.', BODY),
    Paragraph(
        '<b>Half-life of a correlation shock:</b> once correlations are elevated (e.g., during '
        'a market crisis), they decay slowly. The half-life is computed as:', BODY),
    Paragraph('HL = log(0.5) / log(a + b) = log(0.5) / log(0.9956) ≈ 157 trading days  (≈ 7.5 months)', CODE),
    Paragraph(
        '<b>g (asymmetry term, ADCC only):</b> small in absolute terms (0.014) but highly '
        'significant. The LR test statistic of 394 on 1 degree of freedom — the 0.001% '
        'critical value is approximately 10.8 — leaves no doubt that joint negative shocks '
        'drive correlations above what the symmetric model predicts. The economic magnitude '
        'is captured in Figures 2 and 4.', BODY),
    Paragraph(
        '<b>ADCC stationarity constraint is binding:</b> a + b + δg = 0.9999. The optimiser '
        'reached the boundary of the feasible set. This signals that the data wants even '
        'more asymmetry than the constraint allows — a finding confirmed by Cappiello et al. '
        'in their original international equity sample. It is also a warning sign for '
        'out-of-sample performance (see Section 7).', BODY),

    Paragraph('3.3  Log-Likelihood and Model Fit', H2),
    Paragraph(
        'The DCC log-likelihood of −71,756.8 represents the sum of the Stage 2 correlation '
        'component L<sub>C</sub> only (Stage 1 GARCH contributions are constant at the DCC '
        'optimisation stage). The average per-observation contribution is:', BODY),
    Paragraph('−71,756.8 / 7,099 ≈ −10.11 per observation (10-dimensional, T = 7,099)', CODE),
    Paragraph(
        'ADCC improves this by +197.2 units using one additional parameter. The likelihood '
        'ratio statistic LR = 2 × 197.2 = 394.4 vastly exceeds any standard critical value '
        '(χ²(1) at 0.1% = 10.8). Both AIC and BIC favour ADCC despite the complexity '
        'penalty — ADCC saves 392.4 AIC points (= 2×197.2 − 2×1) and 386.7 BIC points.', BODY),

    Paragraph('3.4  Stationarity and Stability', H2),
    Paragraph(
        '<b>Covariance stationarity.</b> Both models satisfy their respective stationarity '
        'constraints strictly (DCC: 0.9956 < 1; ADCC: 0.9999 < 1). The Q<sub>t</sub> process '
        'is covariance stationary, implying correlations mean-revert to Q̄ in expectation.', BODY),
    Paragraph(
        '<b>Positive definiteness.</b> The five-level validation suite confirms that R<sub>t</sub> '
        'is positive definite at every t (all eigenvalues > 0), has unit diagonal (confirmed to '
        'machine precision), and is symmetric. No numerical failure (Cholesky failure) was '
        'encountered during the full-sample filter pass.', BODY),
    Paragraph(
        '<b>Convergence.</b> The SLSQP optimiser converged for both DCC and ADCC from default '
        'starting values (a₀ = 0.05, b₀ = 0.90, g₀ = 0.02). The constraint was active at '
        'convergence for ADCC (binding), inactive for DCC (a + b = 0.9956, interior solution).', BODY),

    Paragraph('3.5  Initial Correlation Dynamics', H2),
    Paragraph(
        'The unconditional correlation matrix Q̄ (time-average of z<sub>t</sub>z′<sub>t</sub>) '
        'serves as the long-run anchor for the DCC recursion. Key features of the estimated '
        'correlation dynamics:', BODY),
    Paragraph('• <b>High baseline correlations.</b> SPY–XLK averages 0.87, SPY–XLI 0.84 over '
              'the full sample. All sector pairs are positively correlated — there are no '
              'natural hedges within the 10-ETF universe.', BULLET),
    Paragraph('• <b>Utility sector (XLU) as structural outlier.</b> Every pair involving XLU '
              'has the lowest time-averaged correlation. XLU–XLK = 0.34, XLU–XLE = 0.36. '
              'Utilities exhibit the defensive, low-beta characteristics expected from theory.', BULLET),
    Paragraph('• <b>Energy sector (XLE) as the most dynamic.</b> SPY–XLE correlations swung '
              'from 0.3 to 0.9 and back over the sample, driven by the commodity price cycle '
              'and its decoupling from broad market earnings.', BULLET),
    Paragraph('• <b>Crisis correlation convergence.</b> Average pairwise correlation peaked '
              'above 0.80 during both the GFC (2008) and COVID (2020) crash — a 60–70% '
              'increase from the pre-crisis baseline of ~0.45–0.50. This is the diversification '
              'failure event that motivates the entire DCC literature.', BULLET),
    Paragraph('• <b>Post-2014 structural decline.</b> Average correlations fell from ~0.65 '
              'to ~0.30 by 2021, reflecting the divergence between technology-sector growth '
              'and energy/financial underperformance. This secular trend was not anticipated '
              'by models estimated on pre-2014 data.', BULLET),
]

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA SUFFICIENCY (renumbered from 1)
# ════════════════════════════════════════════════════════════════════════════
# ── 4. Data ───────────────────────────────────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('4. Data', H1), HR(),
    Paragraph('4.1  Universe and Time Series', H2),
]

etf_data = [
    ['Ticker', 'Name',              'Sector'],
    ['SPY',    'SPDR S&P 500 ETF',              'Broad US equity market'],
    ['XLB',    'Materials Select Sector SPDR',   'Materials'],
    ['XLE',    'Energy Select Sector SPDR',      'Energy'],
    ['XLF',    'Financial Select Sector SPDR',   'Financials'],
    ['XLI',    'Industrial Select Sector SPDR',  'Industrials'],
    ['XLK',    'Technology Select Sector SPDR',  'Information Technology'],
    ['XLP',    'Consumer Staples Select Sector SPDR', 'Consumer Staples'],
    ['XLU',    'Utilities Select Sector SPDR',   'Utilities'],
    ['XLV',    'Health Care Select Sector SPDR', 'Health Care'],
    ['XLY',    'Consumer Discret. Select Sector SPDR', 'Consumer Discretionary'],
]
story += [
    table(etf_data, [TW*0.10, TW*0.48, TW*0.42]),
    Spacer(1, 0.3*cm),
    Paragraph(
        'These 10 ETFs together track the full GICS sector decomposition of the S&amp;P 500 '
        '(with SPY as the aggregate). They were selected because their long shared history '
        'since December 1998 provides a rich cross-section of sector dynamics while keeping '
        'the model dimension (N = 10) tractable for DCC estimation. Daily adjusted closing '
        'prices span <b>22 December 1998 to 20 March 2026</b> (T = 6,851 trading days for '
        'the live system; T = 7,099 for the research sample using Bloomberg total return '
        'indices). Log returns are computed as r<sub>t</sub> = 100 × log(P<sub>t</sub>/P<sub>t-1</sub>), '
        'expressing all quantities in percent-daily units throughout the pipeline.', BODY),

    Paragraph('4.2  Data Sufficiency', H2),
    Paragraph(
        'The dataset is well-suited for DCC estimation.', BODY),
    Paragraph('• <b>T = 7,099, N = 10</b> — far above the minimum needed. With 45 unique pairs '
              'and only 2 (DCC) or 3 (ADCC) parameters to estimate, degrees of freedom are abundant.', BULLET),
    Paragraph('• <b>27.5 years of daily data</b> — covers three distinct stress regimes '
              '(dot-com bust, GFC, COVID), critical for identifying ARCH and GARCH parameters reliably.', BULLET),
    Paragraph('• <b>Z is clean</b> — standardised residuals have std ∈ [0.996, 1.001] across all assets, '
              'zero NaN/Inf. The upstream GARCH step is correct.', BULLET),
    Paragraph('• <b>Distribution note:</b> Z has excess kurtosis of 4–6 (fat tails). Normal QML is still '
              'consistent under these conditions — the estimates are valid — but slightly less efficient '
              'than a Student-t specification. This is standard practice in academic DCC analysis.', BULLET),
]

# ── 4b. Model Parameters ──────────────────────────────────────────────────────
story += [
    Spacer(1, 0.5 * cm),
    Paragraph('4.3  Model Parameters (Summary Table)', H2),
]
param_data = [
    ['Parameter',          'DCC',       'ADCC'],
    ['a (ARCH)',           '0.020856',  '0.018328'],
    ['b (GARCH)',          '0.974774',  '0.969925'],
    ['g (asymmetry)',      '—',         '0.013819'],
    ['a + b',             '0.995630',  '0.988253'],
    ['Constraint value',  '0.995630',  '0.999999'],
    ['Log-likelihood',    '−71,756.8', '−71,559.6'],
    ['AIC',               '143,517.6', '143,125.2'],
    ['BIC',               '143,531.3', '143,145.8'],
    ['LR statistic',      '—',         '394.37'],
    ['LR p-value',        '—',         '~0.000'],
]
story += [
    table(param_data, [TW*0.45, TW*0.275, TW*0.275]),
    Spacer(1, 0.3 * cm),
]

# ── 3. Figure-by-Figure Interpretation ───────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('5. Figure-by-Figure Interpretation', H1), HR(),
]

figs = [
    ('fig1_correlations_dcc_vs_adcc.png',
     'Fig 1 — Conditional Correlations Over Time (Selected Pairs)',
     'DCC and ADCC track almost identically at the pair level across the full 27-year sample. '
     'Visible differences appear at crisis peaks. In SPY–XLF and XLF–XLI, ADCC (dashed red) '
     'spikes slightly above DCC during the GFC and COVID — the asymmetry mechanism at work: '
     'the g term amplifies correlation specifically when both assets experience simultaneous '
     'negative shocks. SPY–XLE is the most volatile pair (correlations swung from ~0.3 to ~0.9). '
     'XLP–XLU (defensive sectors) is the most stable, never exceeding 0.7.'),

    ('fig2_crisis_correlations.png',
     'Fig 2 — Crisis Zoom: Average Pairwise Correlation',
     '<b>Dot-com bust (2000–2002):</b> Correlations climbed steadily as the sell-off broadened. '
     'The ADCC–DCC gap is modest because the bust was gradual, not a sudden shock. '
     '<b>GFC (2007–2009):</b> Correlations hit 0.80+ at the peak in late 2008. The sharp pre-crisis '
     'dip (~0.50 in mid-2007) is the calm before the storm. '
     '<b>COVID crash (2020):</b> The fastest and largest correlation spike in the sample — near 0.80 '
     'within weeks. ADCC sits above DCC for the full recovery period.'),

    ('fig3_avg_corr_and_vol.png',
     'Fig 3 — Average Correlation and Market Volatility (SPY)',
     'The classic volatility–correlation feedback loop is clearly visible. Every spike in SPY '
     'conditional vol coincides with a correlation surge. GFC peak (~100% annualised) dwarfs '
     'COVID (~80%) in vol terms, but COVID was faster. Post-2014 structural decline: average '
     'correlations fell from ~0.65 to ~0.30 by 2021 — reflecting sector divergence (tech '
     'outperformance, energy underperformance). The most recent observations (2024–2026) show '
     'a renewed vol pickup consistent with macro uncertainty.'),

    ('fig4_adcc_minus_dcc.png',
     'Fig 4 — ADCC minus DCC: The Asymmetric Effect',
     'The difference (ADCC avg correlation minus DCC avg correlation) is mostly positive, '
     'concentrated in crisis episodes. This confirms g is capturing something real and episodic: '
     'joint negative shocks drive correlations above what symmetric DCC predicts. '
     'The lower panel (fraction of negative shocks, 63-day MA) shows no systematic trend — '
     'the asymmetry is driven by the intensity of co-negative shocks during specific episodes, '
     'not a drift in the shock distribution.'),

    ('fig5_correlation_heatmaps.png',
     'Fig 5 — Correlation Heatmaps: Unconditional vs DCC vs ADCC (time-averaged)',
     'The three matrices are nearly identical in structure. Time-averaged DCC and ADCC '
     'correlations converge closely to the unconditional Q̄ baseline. '
     'Key hierarchy: highest pairs are SPY–XLK (0.87), SPY–XLI (0.84), SPY–XLY (0.84) — '
     'cyclical sectors move with the broad market. Lowest: XLU–XLK (0.34), XLU–XLE (0.36) — '
     'utilities are the natural diversifier in this universe.'),

    ('fig6_correlation_distribution.png',
     'Fig 6 — Distribution of Pairwise Conditional Correlations',
     'DCC and ADCC have nearly identical full-sample distributions. ADCC is very slightly '
     'shifted: heavier right tail (more observations near 1.0 during crises) and heavier left '
     'tail (slightly lower correlations in calm periods, because b is smaller in ADCC, meaning '
     'faster mean reversion when shocks are absent). XLU is the structural outlier — every pair '
     'involving utilities has the lowest time-average.'),

    ('fig7_conditional_volatilities.png',
     'Fig 7 — GARCH Conditional Volatilities',
     'XLE and XLF are the most volatile sectors (peaked ~150% annualised vol during GFC). '
     'SPY and XLP are the least. The GFC vol spike is the dominant event in the sample, '
     'with COVID as the second. The most recent right edge (2025–2026) shows a broad-based '
     'vol pickup consistent with macroeconomic uncertainty.'),
]

for fname, title, caption in figs:
    story += [
        Paragraph(title, H2),
        Paragraph(caption, BODY),
    ] + full_fig(fname, f'<i>{title}</i>') + [Spacer(1, 0.4 * cm)]

# ── 4. Key Takeaways ──────────────────────────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('6. Key Takeaways', H1), HR(),

    Paragraph('<b>1. ADCC is statistically and economically superior to DCC (in-sample).</b>', H2),
    Paragraph('LR stat = 394 on 1 degree of freedom is decisive — not a borderline result. '
              'Both AIC and BIC favour ADCC. The asymmetric coefficient g = 0.0138 is small '
              'in absolute terms but its effect accumulates over T = 7,099 observations, '
              'producing a +197 log-likelihood unit improvement.', BODY),

    Paragraph('<b>2. Correlations are highly persistent.</b>', H2),
    Paragraph('a + b = 0.9956 (DCC). The half-life of a correlation shock is:', BODY),
    Paragraph('log(0.5) / log(0.9956)  ≈  157 trading days  (~7.5 months)', CODE),
    Paragraph('Once elevated, sector correlations stay elevated for a long time.', BODY),

    Paragraph('<b>3. The ADCC stationarity constraint is binding.</b>', H2),
    Paragraph('a + b + δ·g = 0.9999. The optimiser pushed g to the feasibility frontier. '
              'The data wants even more asymmetry than the constraint allows. '
              'This is an economic signal: the leverage effect in sector correlations is a '
              'structural feature of this dataset, not a sample artefact.', BODY),

    Paragraph('<b>4. Practical implication for portfolio construction.</b>', H2),
    Paragraph('During crises, DCC will underestimate true correlations relative to ADCC. '
              'A minimum-variance portfolio using DCC-implied covariances will appear more '
              'diversified than it actually is during drawdowns — precisely when diversification '
              'matters most. ADCC provides more conservative and accurate correlation estimates '
              'in stress scenarios.', BODY),
]

# ── 5. OOS Evaluation ─────────────────────────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('7. Out-of-Sample Forecast Evaluation', H1), HR(),
    Paragraph('<b>Split:</b> Train = first 80% (1999-12-23 – 2020-09-28, T=5,679) | '
              'Test = last 20% (2020-09-29 – 2026-03-09, T=1,420). '
              'Parameters estimated on training data only (no look-ahead). '
              'Realised covariance proxy: r<sub>t</sub>r<sub>t</sub>′ (daily outer product — '
              'noisy but unbiased).', BODY),

    Paragraph('OOS Parameters (training set only)', H2),
]
oos_params = [
    ['',      'Full sample', 'Train only'],
    ['DCC a', '0.020856',   '0.021539'],
    ['DCC b', '0.974774',   '0.973561'],
    ['ADCC a','0.018328',   '0.017991'],
    ['ADCC b','0.969925',   '0.968185'],
    ['ADCC g','0.013819',   '0.016715'],
]
story += [
    table(oos_params, [TW*0.35, TW*0.325, TW*0.325]),
    Spacer(1, 0.3 * cm),
    Paragraph('Parameters are stable between estimation windows. g is slightly larger on training '
              'data (which contained the full GFC and COVID onset), consistent with it capturing '
              'crisis-era asymmetry.', BODY),

    Paragraph('OOS Loss Summary', H2),
]
loss_data = [
    ['Metric',           'DCC',     'ADCC',    'Winner'],
    ['Mean QLIKE',       '−8.0744', '−7.9347', 'DCC'],
    ['Mean MSE',         '180.04',  '186.00',  'DCC'],
    ['QLIKE improvement','—',       '−1.73%',  ''],
]
story += [
    table(loss_data, [TW*0.38, TW*0.2, TW*0.2, TW*0.22]),
    Spacer(1, 0.3 * cm),

    Paragraph('Diebold-Mariano Test (H₀: equal predictive accuracy)', H2),
]
dm_data = [
    ['',              'QLIKE',     'MSE'],
    ['mean(d_t)',      '−0.1396',  '−5.96'],
    ['DM statistic',  '−13.271',  '−7.167'],
    ['p-value',       '~0.000',   '~0.000'],
    ['Verdict',       'DCC wins', 'DCC wins'],
]
story += [
    table(dm_data, [TW*0.40, TW*0.30, TW*0.30]),
    Spacer(1, 0.3 * cm),

    Paragraph('Sub-Period Breakdown', H2),
]
sub_data = [
    ['Period',              'N',     'DCC QLIKE', 'ADCC QLIKE', 'ADCC better?'],
    ['COVID crash',         '68',    '−7.0625',   '−7.0653',    'Yes (barely)'],
    ['Calm (non-COVID)',    '1,352', '−8.1252',   '−7.9785',    'No'],
]
story += [
    table(sub_data, [TW*0.32, TW*0.08, TW*0.18, TW*0.18, TW*0.24]),
    Spacer(1, 0.4 * cm),
]

# OOS figures
oos_figs = [
    ('fig8_oos_qlike.png',
     'Fig 8 — OOS QLIKE: DCC vs ADCC (test period 2020–2026)',
     'Daily QLIKE loss for each model across the test period.'),
    ('fig9_cumulative_qlike.png',
     'Fig 9 — Cumulative QLIKE Difference (DCC − ADCC)',
     'Positive = DCC winning. ADCC leads briefly during the COVID crash (68 days); '
     'DCC accumulates a ~200 unit advantage over the subsequent 1,352 calm days.'),
    ('fig10_forecast_vs_realized.png',
     'Fig 10 — Forecast vs Realised Volatility',
     'Model forecast σ vs realised |return| across the test period.'),
]
for fname, title, caption in oos_figs:
    story += [Paragraph(title, H2)] + full_fig(fname, caption) + [Spacer(1, 0.3 * cm)]

# ── 6. Interpretation of OOS Results ─────────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('7b. Interpretation of OOS Results', H1), HR(),
    Paragraph('<b>DCC forecasts better out-of-sample, decisively — a reversal of the in-sample finding.</b>', BODY),

    Paragraph('1. The asymmetry parameter g was over-fitted to in-sample crises.', H2),
    Paragraph('ADCC estimated g using GFC (2008) and COVID onset (2020) in the training set. '
              'These concentrated the signal for asymmetric correlations in a handful of episodes. '
              'Out of sample (2020–2026), the market entered a post-COVID regime where asymmetric '
              'effects were absent, and the inflated g systematically over-estimated correlations.', BODY),

    Paragraph('2. ADCC wins in stress, loses in calm — and calm dominates the test period.', H2),
    Paragraph('The sub-period table confirms this exactly. For portfolio risk management, if you can '
              'identify stress regimes in real time, ADCC is the right model during them. '
              'Without that conditioning information, DCC is safer.', BODY),

    Paragraph('3. The binding ADCC constraint was a warning sign.', H2),
    Paragraph('In-sample, a + b + δ·g = 0.9999. When the optimiser pushes a parameter to the '
              'boundary, it is extracting maximum in-sample signal. This typically signals '
              'over-fitting — confirmed out-of-sample.', BODY),

    Spacer(1, 0.4 * cm),
    Paragraph('Final Verdict', H2),
]
verdict_data = [
    ['Criterion',                              'Winner'],
    ['In-sample fit (LLH, AIC, BIC, LR test)', 'ADCC'],
    ['OOS QLIKE (Diebold-Mariano)',             'DCC'],
    ['OOS MSE (Diebold-Mariano)',               'DCC'],
    ['Stress periods only',                    'ADCC (marginally)'],
]
story += [
    table(verdict_data, [TW*0.65, TW*0.35]),
    Spacer(1, 0.4 * cm),
    Paragraph('<b>Use DCC for unconditional covariance forecasting.</b><br/>'
              '<b>Use ADCC conditionally on a stress/crisis indicator</b> — it is the right '
              'model during episodes of synchronous negative shocks, but over-shoots in calm '
              'regimes. This is a textbook bias-variance tradeoff: ADCC has lower bias '
              '(captures a real phenomenon) but higher variance (one extra parameter, binding '
              'constraint). Over a test period dominated by calm, the variance penalty dominates.', VERDICT),
]

# ── 7. Live Forecasting Architecture ─────────────────────────────────────────
story += [
    PageBreak(),
    Paragraph('8. Live Forecasting: Filter State vs Parameters', H1), HR(),

    Paragraph('The DCC model has two conceptually separate objects that update on different timescales.', BODY),

    Paragraph('Filter State — Q_t (updates every day, no optimisation)', H2),
    Paragraph('Q_{t+1} = (1−a−b)·Q̄ + a·z_t z_t′ + b·Q_t', CODE),
    Paragraph('One matrix operation, microseconds. This runs every day by construction — it is not '
              'optional. Every day you receive new data, you run this line and you have tomorrow\'s '
              'covariance forecast.', BODY),

    Paragraph('Parameters — (a, b, g) and Q̄ (estimated by optimisation over history)', H2),
    Paragraph('These represent structural features of the process: how fast correlations react, '
              'how persistent they are, how asymmetric they are. They are identified from thousands '
              'of observations and do not change meaningfully day to day. Adding one new observation '
              'to a sample of T=7,000 changes the MLE estimate by order 1/T ≈ 0.014%. Running a '
              '30-second optimisation to move a from 0.020856 to 0.020857 is redundant, not wrong.', BODY),

    Paragraph('Recommended Live Architecture', H2),
]
arch_data = [
    ['Frequency',  'Action'],
    ['Every day',  'Download prices → GARCH filter → DCC one-step update → save H_{t+1|t}'],
    ['Monthly',    'Re-estimate (a, b, g) and Q̄ on rolling 5-year window → replace stored params'],
]
story += [
    table(arch_data, [TW*0.20, TW*0.80]),
    Spacer(1, 0.3 * cm),
    Paragraph('The real lever is not <i>how often</i> you re-estimate but <i>what window</i> you use. '
              'A rolling window that drops old crisis episodes eventually allows g to shrink toward zero '
              'as a calm regime accumulates, correcting the over-fitting documented in Section 6.', BODY),
]

# ── Build PDF ─────────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN,  bottomMargin=MARGIN,
    title='DCC/ADCC-GARCH Analysis',
    author='MFE UCLA — DCC-GARCH Project',
)
doc.build(story)
sys.stdout.buffer.write(f'PDF saved: {os.path.basename(OUT_PDF)} ({os.path.getsize(OUT_PDF):,} bytes)\n'.encode('utf-8'))
