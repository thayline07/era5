import xarray as xr
import numpy as np

ds = xr.open_dataset("era5_41004_2025_01_d01.grib", engine="cfgrib")
print(ds)

v = list(ds.data_vars)[0]
a = ds[v]
print()
print("variável:", v)
print("dimensões:", a.dims)
print("formato:", a.shape)
print("vazios (NaN):", int(np.isnan(a.values).sum()), "de", a.size)
print("mínimo e máximo:", float(np.nanmin(a.values)), float(np.nanmax(a.values)))
print("horas:", a["time"].values[:3] if "time" in a.coords else "sem coordenada time")