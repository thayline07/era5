import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculo do Hs e comparação com a boia (pelo código, só está funcionando no computador de origem, pois os caminhos seguem as pastas existentes lá)


pkl = os.path.expanduser(
    "~/Desktop/ic/espectral-wave-data/spectral-wave-data-analysis-main/"
    "dados/41004_2025/dados_41004_2025.pkl")
npz = os.path.expanduser("~/Desktop/ic/era5/era5_41004_2025_01_d01-07.npz")

# Hs do ERA5 
e = np.load(npz)
S, freq_e, tempos = e["S"], e["freq"], e["tempos"]
dtheta = np.radians(15.0)
E_f = S.sum(axis=1) * dtheta                                   
Hs_era5 = 4.0 * np.sqrt(np.trapezoid(E_f, x=freq_e, axis=1))

# Hs da boia, a partir do espectro
with open(pkl, "rb") as f:
    w = pickle.load(f)["w"]
freq_b = np.array([float(c) for c in w.columns[5:]])
df = np.diff(freq_b)
df = np.append(df, df[-1])                                    
m0_b = (w.iloc[:, 5:].to_numpy(dtype=float) * df).sum(axis=1)
Hs_boia = 4.0 * np.sqrt(m0_b)
t_boia = pd.to_datetime(dict(year=w["#YY"], month=w["MM"], day=w["DD"],
                             hour=w["hh"], minute=w["mm"]))

# tempo (janela de 20 min)
era = pd.DataFrame({"t": pd.to_datetime(tempos).astype("datetime64[ns]"), "era5": Hs_era5})
boia = pd.DataFrame({"t": t_boia.astype("datetime64[ns]"), "boia": Hs_boia}).sort_values("t")
par = pd.merge_asof(era.sort_values("t"), boia, on="t",
                    direction="nearest", tolerance=pd.Timedelta("20min")).dropna()

x = par["boia"].to_numpy()
y = par["era5"].to_numpy()
erro_medio = np.mean(y - x)
rmse = np.sqrt(np.mean((y - x) ** 2))
r = np.corrcoef(x, y)[0, 1]
print(f"N = {len(par)} (de {len(era)} horas do ERA5)")
print(f"erro médio (ERA5 - boia) = {erro_medio:+.3f} m")
print(f"RMSE = {rmse:.3f} m")
print(f"r = {r:.3f}")
print(f"Hs máximo: boia {x.max():.2f} m | ERA5 {y.max():.2f} m")

# gráfico 1: dispersão
lim = 1.05 * max(x.max(), y.max())
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=18, alpha=0.7)
plt.plot([0, lim], [0, lim], "k--", linewidth=1, label="$y = x$")
plt.xlim(0, lim); plt.ylim(0, lim)
plt.xlabel("$H_s$ da boia 41004 (m)")
plt.ylabel("$H_s$ do ERA5 (m)")
plt.title(f"N = {len(par)} | erro médio = {erro_medio:+.2f} m | "
          f"RMSE = {rmse:.2f} m | r = {r:.3f}", fontsize=10)
plt.legend(); plt.grid(alpha=0.4)
plt.savefig("dispersao_hs_era5_41004.png", dpi=300, bbox_inches="tight")
plt.show()

# gráfico 2: série temporal 
plt.figure(figsize=(9, 4))
plt.plot(par["t"], par["boia"], "o-", markersize=3, label="Boia 41004")
plt.plot(par["t"], par["era5"], "s-", markersize=3, label="ERA5")
plt.ylabel("$H_s$ (m)"); plt.legend(); plt.grid(alpha=0.4)
plt.gcf().autofmt_xdate()
plt.savefig("serie_hs_era5_41004.png", dpi=300, bbox_inches="tight")
plt.show()

fmin, fmax = 0.0345, 0.485
m_e = (freq_e >= fmin) & (freq_e <= fmax)
m_b = (freq_b >= fmin) & (freq_b <= fmax)

Hs_era5_c = 4.0 * np.sqrt(np.trapezoid(E_f[:, m_e], x=freq_e[m_e], axis=1))
m0_bc = (w.iloc[:, 5:].to_numpy(dtype=float)[:, m_b] * df[m_b]).sum(axis=1)
Hs_boia_c = 4.0 * np.sqrt(m0_bc)

era_c = pd.DataFrame({"t": pd.to_datetime(tempos).astype("datetime64[ns]"), "era5": Hs_era5_c})
boia_c = pd.DataFrame({"t": t_boia.astype("datetime64[ns]"), "boia": Hs_boia_c}).sort_values("t")
par_c = pd.merge_asof(era_c.sort_values("t"), boia_c, on="t",
                      direction="nearest", tolerance=pd.Timedelta("20min")).dropna()
xc, yc = par_c["boia"].to_numpy(), par_c["era5"].to_numpy()
print("faixa comum:", fmin, "a", fmax, "Hz")
print(f"N = {len(par_c)} | erro médio = {np.mean(yc - xc):+.3f} m | "
      f"RMSE = {np.sqrt(np.mean((yc - xc) ** 2)):.3f} m | r = {np.corrcoef(xc, yc)[0, 1]:.3f}")