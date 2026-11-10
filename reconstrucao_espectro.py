import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# ---- escolha aqui ----
boia_id, ano = "41004", 2025
instante = pd.Timestamp("2025-01-06 20:00")      # hora do ERA5 (UTC)
fmin, fmax = 0.0345, 0.485                       # faixa de frequências comum
npz = os.path.expanduser("~/Desktop/ic/era5/era5_41004_2025_01_d01-07.npz")
pkl = os.path.expanduser(
    f"~/Desktop/ic/espectral-wave-data/spectral-wave-data-analysis-main/"
    f"dados/{boia_id}_{ano}/dados_{boia_id}_{ano}.pkl")
# ----------------------

# ---- ERA5 ----
e = np.load(npz)
S_all, freq_e, dir_e = e["S"], e["freq"], e["dir_de"]
tempos = pd.to_datetime(e["tempos"])
i_t = int(np.argmin(np.abs((tempos - instante).values)))
hora_e = tempos[i_t]
S_e = S_all[i_t].T                                # (frequência, direção), m²/Hz/rad

# ---- boia: dados do instante ----
with open(pkl, "rb") as f:
    d = pickle.load(f)

def linha(df, alvo):
    t = pd.to_datetime(dict(year=df["#YY"], month=df["MM"], day=df["DD"],
                            hour=df["hh"], minute=df["mm"]))
    dif = (t - alvo).abs()
    i = int(dif.values.argmin())
    return i, t.iloc[i], dif.iloc[i]

def pega(nome):
    df = d[nome]
    i, t, dif = linha(df, hora_e)
    if dif > pd.Timedelta("20min"):
        raise SystemExit(f"{nome}: sem registro a menos de 20 min de {hora_e}")
    return df.iloc[i, 5:].to_numpy(dtype=float), t

fb = np.array([float(c) for c in d["w"].columns[5:]])
E_b, t_reg = pega("w")
a1, _ = pega("alpha1")
a2, _ = pega("alpha2")
r1, _ = pega("r1")
r2, _ = pega("r2")
a1 = np.where(a1 > 360, np.nan, a1)
a2 = np.where(a2 > 360, np.nan, a2)
r1 = np.where(r1 > 1, np.nan, r1)
r2 = np.where(r2 > 1, np.nan, r2)

titulo = (f"Boia {boia_id} | {hora_e:%d/%m/%Y} | "
          f"ERA5 às {hora_e:%H:%M} UTC, boia às {t_reg:%H:%M} UTC")
print(titulo)

# ---- MEM de Hashimoto e Kobune (mesma função do valida_reconstrucao.py) ----
ang_g = np.arange(0, 361, 1)
ang_r = np.radians(ang_g)
dth = ang_r[1] - ang_r[0]
phi = np.array([np.cos(ang_r), np.sin(ang_r), np.cos(2 * ang_r), np.sin(2 * ang_r)])

def _dist(lam):
    logw = -(lam @ phi)
    mx = logw.max()
    ew = np.exp(logw - mx)
    Z = ew.sum() * dth
    return ew / Z, np.log(Z) + mx

def mem_shannon(m, tol=1e-10, maxiter=200):
    lam = np.zeros(4)
    D, logZ = _dist(lam)
    F = logZ + lam @ m
    for _ in range(maxiter):
        E = phi @ D * dth
        grad = m - E
        if np.max(np.abs(grad)) < tol:
            return D, True
        cov = (phi * D) @ phi.T * dth - np.outer(E, E)
        try:
            passo = np.linalg.solve(cov + 1e-14 * np.eye(4), grad)
        except np.linalg.LinAlgError:
            return D, False
        t = 1.0
        while t > 1e-12:
            lam_novo = lam - t * passo
            D_novo, logZ_novo = _dist(lam_novo)
            F_novo = logZ_novo + lam_novo @ m
            if np.isfinite(F_novo) and F_novo <= F - 1e-4 * t * (grad @ passo):
                break
            t *= 0.5
        else:
            return D, False
        lam, D, F = lam_novo, D_novo, F_novo
    return D, bool(np.max(np.abs(m - phi @ D * dth)) < 1e-6)

S_hk = np.full((len(fb), len(ang_g)), np.nan)     # m²/Hz/rad
for k in range(len(fb)):
    if np.isnan([E_b[k], a1[k], a2[k], r1[k], r2[k]]).any():
        continue
    c1 = r1[k] * np.exp(1j * np.radians(a1[k]))
    c2 = r2[k] * np.exp(2j * np.radians(a2[k]))
    if not (abs(c1) < 1 and abs(c2 - c1 ** 2) < 1 - abs(c1) ** 2):
        continue
    D, ok = mem_shannon(np.array([c1.real, c1.imag, c2.real, c2.imag]))
    if ok:
        S_hk[k] = E_b[k] * D

# ---- faixa comum ----
m_e = (freq_e >= fmin) & (freq_e <= fmax)
m_b = (fb >= fmin) & (fb <= fmax)
sem_mem = int(np.isnan(S_hk[m_b, 0]).sum())
print(f"frequências da boia na faixa comum: {int(m_b.sum())} | sem MEM: {sem_mem}")

# ---- energia por direção (integrada em frequência na faixa comum) ----
G_e = np.trapezoid(S_e[m_e], x=freq_e[m_e], axis=0)                      # (24,)
G_b1 = np.trapezoid(np.nan_to_num(S_hk[m_b]), x=fb[m_b], axis=0)         # (361,)
setor = ang_g[:-1] // 15
G_b = np.array([G_b1[:-1][setor == k].mean() for k in range(24)])        # setores de 15°
p_e = G_e / G_e.sum()
p_b = G_b / G_b.sum()

def dir_media(p, centros):
    a = np.radians(centros)
    return np.degrees(np.arctan2((p * np.sin(a)).sum(), (p * np.cos(a)).sum())) % 360

dm_e, dm_b = dir_media(p_e, dir_e), dir_media(p_b, dir_e)
print(f"direção média (de onde vem): ERA5 {dm_e:.1f}° | boia (HK) {dm_b:.1f}°")
print(f"setor de pico: ERA5 {dir_e[np.argmax(p_e)]:.1f}° | boia (HK) {dir_e[np.argmax(p_b)]:.1f}°")

nome = f"{boia_id}_{hora_e:%Y%m%d_%H%M}"

# ---- gráfico 1: mapas polares (ERA5 interpolado para visualização) ----
conv = np.pi / 180                                  # m²/Hz/rad -> m²/Hz/grau
S_e_g, S_hk_g = S_e * conv, S_hk * conv
vmax = max(np.nanmax(S_e_g[m_e]), np.nanmax(S_hk_g[m_b]))
niveis = np.linspace(0, vmax, 30)

lf = np.log(freq_e)
th = np.radians(dir_e)                              # 7,5°, 22,5°, ..., 352,5°
pad = 3                                             # repete colunas para fechar o círculo
th_p = np.concatenate([th[-pad:] - 2 * np.pi, th, th[:pad] + 2 * np.pi])
Z = np.concatenate([S_e_g[:, -pad:], S_e_g, S_e_g[:, :pad]], axis=1)
spl = RectBivariateSpline(lf, th_p, Z, kx=3, ky=3)
lf_fino = np.linspace(lf[0], lf[-1], 200)
th_fino = np.radians(np.arange(0, 361, 1))
Z_fino = np.clip(spl(lf_fino, th_fino), 0.0, vmax)

fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), subplot_kw={"projection": "polar"},
                        layout="constrained")
A2, F2 = np.meshgrid(th_fino, np.exp(lf_fino))
cs = axs[0].contourf(A2, F2, Z_fino, cmap="jet", levels=niveis)
axs[0].set_title("ERA5\n", fontsize=12, fontweight="bold")
Ang, Fr = np.meshgrid(ang_r, fb)
axs[1].contourf(Ang, Fr, S_hk_g, cmap="jet", levels=niveis)
axs[1].set_title("Boia, MEM\n(Hashimoto e Kobune)", fontsize=12, fontweight="bold")
for ax in axs:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(fmin, fmax)
fig.suptitle(titulo, fontsize=13, fontweight="bold")
fig.colorbar(cs, ax=axs, shrink=0.75, pad=0.04,
             label="Densidade de Energia ($m^2/Hz/grau$)")
fig.savefig(f"polar_era5_vs_boia_{nome}.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- gráfico 2: energia por direção ----
plt.figure(figsize=(9, 4.8))
plt.bar(dir_e, 100 * p_e, width=15, alpha=0.5, label=f"ERA5 (direção média {dm_e:.0f}°)")
plt.plot(dir_e, 100 * p_b, "o-", color="purple",
         label=f"Boia, MEM H-K (direção média {dm_b:.0f}°)")
plt.xlim(0, 360)
plt.xlabel("Direção de onde vem (graus)")
plt.ylabel("Energia na faixa comum, por setor de 15° (%)")
plt.title(titulo, fontsize=10)
plt.legend(); plt.grid(alpha=0.4)
plt.savefig(f"direcao_era5_vs_boia_{nome}.png", dpi=300, bbox_inches="tight")
plt.show()