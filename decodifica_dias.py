import glob
import numpy as np
import xarray as xr

def decodifica(arquivo):
    ds = xr.open_dataset(arquivo, engine="cfgrib")
    a = ds["d2fd"].squeeze(["latitude", "longitude"])
    a = a.transpose("time", "directionNumber", "frequencyNumber")
    S = np.nan_to_num(10.0 ** a.values, nan=0.0)      # m²/Hz/rad
    return S, a["time"].values

arquivos = sorted(glob.glob("era5_41004_2025_01_d0?.grib"))
partes, tempos = [], []
for arq in arquivos:
    S, t = decodifica(arq)
    partes.append(S)
    tempos.append(t)
    print(arq, S.shape)

S = np.concatenate(partes, axis=0)
tempos = np.concatenate(tempos)

n_dir, n_freq = S.shape[1], S.shape[2]
freq = 0.03453 * 1.1 ** np.arange(n_freq)
dir_de = (7.5 + 15.0 * np.arange(n_dir) + 180.0) % 360.0
ordem = np.argsort(dir_de)
dir_de = dir_de[ordem]
S = S[:, ordem, :]

dtheta = np.radians(15.0)
Hs = 4.0 * np.sqrt(np.trapezoid(S.sum(axis=1) * dtheta, x=freq, axis=1))
print("horas:", S.shape[0], "| Hs de", round(Hs.min(), 2), "a", round(Hs.max(), 2), "m")

np.savez("era5_41004_2025_01_d01-07.npz", S=S, freq=freq, dir_de=dir_de, tempos=tempos)
print("salvo: era5_41004_2025_01_d01-07.npz")