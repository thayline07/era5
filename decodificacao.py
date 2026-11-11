import glob
import os
import numpy as np
import xarray as xr

# Decodifica os blocos de 3 em 3 horas de janeiro das três boias
# e salva um arquivo npz por boia, com o mesmo formato do script dos 7 dias.

base = os.path.expanduser("~/Desktop/ic/era5")

# ano, latitude e longitude
boias = {
    "41004": (2025, 32.502, -79.099),
    "41040": (2019, 14.568, -53.037),
    "46080": (2025, 57.910, -150.129),
}


def abre(arquivo):
    ds = xr.open_dataset(arquivo, engine="cfgrib", backend_kwargs={"indexpath": ""})
    a = ds["d2fd"]
    for dim in ("time", "latitude", "longitude"):
        if dim not in a.dims:
            a = a.expand_dims(dim)
    a = a.transpose("time", "directionNumber", "frequencyNumber", "latitude", "longitude")
    return a.values, a["time"].values, a["latitude"].values, a["longitude"].values


for boia, (ano, lat_b, lon_b) in boias.items():
    pasta = os.path.join(base, f"janeiro-{boia}-3em3h")
    arquivos = sorted(glob.glob(os.path.join(pasta, f"era5_{boia}_{ano}_01_d*_3h.grib")))
    print(f"\n=== Boia {boia} ({ano}) | {len(arquivos)} arquivos em {pasta}")
    if not arquivos:
        print("nenhum arquivo encontrado, pulando")
        continue

    partes, tempos = [], []
    for arq in arquivos:
        x, t, lat, lon = abre(arq)      
        partes.append(x)
        tempos.append(t)
        print(" ", os.path.basename(arq), x.shape)

    x = np.concatenate(partes, axis=0)
    tempos = np.concatenate(tempos)

    tempos, idx = np.unique(tempos, return_index=True)
    x = x[idx]


    print("  pontos de grade (lat, lon, distância à boia, fração de vazios):")
    melhor = None
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            vazio = float(np.isnan(x[..., i, j]).mean())
            dist = np.hypot((la - lat_b) * 111.0, (lo - lon_b) * 111.0 * np.cos(np.radians(lat_b)))
            print(f"    {la:6.1f} {lo:7.1f}  {dist:6.1f} km  vazios {100 * vazio:5.1f}%")
            if vazio < 0.999 and (melhor is None or dist < melhor[0]):
                melhor = (dist, i, j)
    if melhor is None:
        print("  todos os pontos vazios, pulando")
        continue
    dist, i, j = melhor
    print(f"  ponto usado: lat {lat[i]}, lon {lon[j]} ({dist:.1f} km da boia)")

    S = np.nan_to_num(10.0 ** x[..., i, j], nan=0.0)

    n_dir, n_freq = S.shape[1], S.shape[2]
    freq = 0.03453 * 1.1 ** np.arange(n_freq)
    dir_de = (7.5 + 15.0 * np.arange(n_dir) + 180.0) % 360.0
    ordem = np.argsort(dir_de)
    dir_de = dir_de[ordem]
    S = S[:, ordem, :]

    dtheta = np.radians(15.0)
    Hs = 4.0 * np.sqrt(np.trapezoid(S.sum(axis=1) * dtheta, x=freq, axis=1))

    passos = np.unique(np.diff(tempos))
    print(f"  horários: {S.shape[0]} (esperado 248 para 31 dias) | passos: {passos}")
    print(f"  Hs de {Hs.min():.2f} a {Hs.max():.2f} m")

    saida = os.path.join(base, f"era5_{boia}_{ano}_01_3h.npz")
    np.savez(saida, S=S, freq=freq, dir_de=dir_de, tempos=tempos,
             lat_no=lat[i], lon_no=lon[j])
    print("  salvo:", saida)
