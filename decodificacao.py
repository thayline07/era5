import numpy as np
import xarray as xr

# Arquivo geral para decodificar os dados baixados

arquivo = "era5_41004_2025_01_d01.grib"
ds = xr.open_dataset(arquivo, engine="cfgrib")
print(ds)
a = ds["d2fd"].squeeze(["latitude", "longitude"])
a = a.transpose("time", "directionNumber", "frequencyNumber")

# log10, vazio vira zero
S = np.nan_to_num(10.0 ** a.values, nan=0.0)      # m²/Hz/rad, (hora, direção, frequência)
tempos = a["time"].values

n_hora, n_dir, n_freq = S.shape

# 2) eixos
freq = 0.03453 * 1.1 ** np.arange(n_freq)         # Hz, 30 valores
dir_para = 7.5 + 15.0 * np.arange(n_dir)          # graus, "para onde" (ERA5)

# 3) direção: "para onde" -> "de onde vem" (convenção NDBC), em ordem crescente
dir_de = (dir_para + 180.0) % 360.0

ordem = np.argsort(dir_de)
dir_de = dir_de[ordem]
S = S[:, ordem, :]

# 4) conferências
dtheta = np.radians(15.0)
E_f = S.sum(axis=1) * dtheta                      # m²/Hz, formato (hora, frequência)
m0 = np.trapezoid(E_f, x=freq, axis=1)
Hs = 4.0 * np.sqrt(m0)
print("Hs aproximado, hora a hora (m):")
print(np.round(Hs, 2))

t = 9
i_dir, i_f = np.unravel_index(np.argmax(S[t]), S[t].shape)
print(f"\nhora {t:02d}: pico em f = {freq[i_f]:.3f} Hz (T = {1 / freq[i_f]:.1f} s), "
      f"direção de onde vem = {dir_de[i_dir]:.1f}°")
print("faixa de frequências do ERA5:", round(freq[0], 4), "a", round(freq[-1], 4), "Hz")

np.savez("era5_41004_2025_01_d01.npz", S=S, freq=freq, dir_de=dir_de, tempos=tempos)
print("salvo: era5_41004_2025_01_d01.npz")