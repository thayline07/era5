import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


boia_id, ano = "41004", 2025
fmin, fmax = 0.0345, 0.485
npz = os.path.expanduser("~/Desktop/ic/era5/era5_41004_2025_01_d01-07.npz")
pkl = os.path.expanduser(
    f"~/Desktop/ic/espectral-wave-data/spectral-wave-data-analysis-main/"
    f"dados/{boia_id}_{ano}/dados_{boia_id}_{ano}.pkl")


e = np.load(npz)
S_all, freq_e, dir_e = e["S"], e["freq"], e["dir_de"]
tempos = pd.to_datetime(e["tempos"])

with open(pkl, "rb") as f:
    d = pickle.load(f)
nomes = ["w", "alpha1", "alpha2", "r1", "r2"]
t_df = {n: pd.to_datetime(dict(year=d[n]["#YY"], month=d[n]["MM"], day=d[n]["DD"],
                               hour=d[n]["hh"], minute=d[n]["mm"])).values for n in nomes}
V = {n: d[n].iloc[:, 5:].to_numpy(dtype=float) for n in nomes}
fb = np.array([float(c) for c in d["w"].columns[5:]])

# MEM de Hashimoto e Kobune (mesma função do valida_reconstrucao.py)
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

def reconstroi(E_b, a1, a2, r1, r2):
    S_hk = np.full((len(fb), len(ang_g)), np.nan)          # m²/Hz/rad
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
    return S_hk

def dir_media(p, centros):
    a = np.radians(centros)
    return np.degrees(np.arctan2((p * np.sin(a)).sum(), (p * np.cos(a)).sum())) % 360

def dif_ang(a, b):
    return (a - b + 180) % 360 - 180

m_e = (freq_e >= fmin) & (freq_e <= fmax)
m_b = (fb >= fmin) & (fb <= fmax)
setor = ang_g[:-1] // 15
dtheta = np.radians(15.0)

linhas = []
for i, tempo in enumerate(tempos):
    idx, ok_t = {}, True
    for n in nomes:
        dif = np.abs(t_df[n] - tempo.to_datetime64())
        j = int(dif.argmin())
        if dif[j] > np.timedelta64(20, "m"):
            ok_t = False
            break
        idx[n] = j
    if not ok_t:
        continue

    E_b = V["w"][idx["w"]]
    a1 = np.where(V["alpha1"][idx["alpha1"]] > 360, np.nan, V["alpha1"][idx["alpha1"]])
    a2 = np.where(V["alpha2"][idx["alpha2"]] > 360, np.nan, V["alpha2"][idx["alpha2"]])
    r1 = np.where(V["r1"][idx["r1"]] > 1, np.nan, V["r1"][idx["r1"]])
    r2 = np.where(V["r2"][idx["r2"]] > 1, np.nan, V["r2"][idx["r2"]])
    S_hk = reconstroi(E_b, a1, a2, r1, r2)
    S_e = S_all[i].T

    G_e = np.trapezoid(S_e[m_e], x=freq_e[m_e], axis=0)
    G_b1 = np.trapezoid(np.nan_to_num(S_hk[m_b]), x=fb[m_b], axis=0)
    G_b = np.array([G_b1[:-1][setor == k].mean() for k in range(24)])
    p_e, p_b = G_e / G_e.sum(), G_b / G_b.sum()

    linhas.append(dict(
        t=tempo,
        dm_e=dir_media(p_e, dir_e), dm_b=dir_media(p_b, dir_e),
        pk_e=dir_e[np.argmax(p_e)], pk_b=dir_e[np.argmax(p_b)],
        Hs_e=4 * np.sqrt(np.trapezoid(S_e[m_e].sum(axis=1) * dtheta, x=freq_e[m_e])),
        Hs_b=4 * np.sqrt(np.trapezoid(E_b[m_b], x=fb[m_b])),
        sem_mem=int(np.isnan(S_hk[m_b, 0]).sum())))

r = pd.DataFrame(linhas)
r["dif_media"] = dif_ang(r["dm_e"], r["dm_b"])
r["dif_pico"] = dif_ang(r["pk_e"], r["pk_b"])
r.to_csv(f"resumo_direcional_{boia_id}.csv", index=False)

print(f"horas comparadas: {len(r)} de {len(tempos)}")
print(f"frequências sem MEM por hora: média {r['sem_mem'].mean():.1f}, máximo {r['sem_mem'].max()}")
for col, nome in [("dif_media", "direção média"), ("dif_pico", "setor de pico")]:
    x = r[col]
    print(f"{nome}: diferença média {x.mean():+.1f}° | mediana de |dif| {x.abs().median():.1f}° | "
          f"|dif| ≤ 15°: {100 * (x.abs() <= 15).mean():.0f}% | |dif| ≤ 30°: {100 * (x.abs() <= 30).mean():.0f}%")

# gráfico: diferença de direção ao longo do tempo
x = r["dif_media"]
fig, ax = plt.subplots(figsize=(10, 4.5))
sc = ax.scatter(r["t"], x, c=r["Hs_b"], cmap="viridis", s=22)
ax.axhline(0, color="k", linewidth=1)
ax.axhspan(-15, 15, color="gray", alpha=0.15, label="±15° (largura de um setor do ERA5)")
ax.set_ylabel("Direção média ERA5 − boia (graus)")
ax.set_title(f"Boia {boia_id} | N = {len(r)} horas | diferença média {x.mean():+.1f}° | "
             f"|dif| ≤ 15°: {100 * (x.abs() <= 15).mean():.0f}%", fontsize=10)
fig.colorbar(sc, label="$H_s$ da boia na faixa comum (m)")
ax.legend(loc="upper right"); ax.grid(alpha=0.4)
fig.autofmt_xdate()
fig.savefig(f"diferenca_direcao_era5_boia_{boia_id}.png", dpi=300, bbox_inches="tight")
plt.show()