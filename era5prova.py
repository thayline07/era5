import cdsapi

# Script criado para fazer um pedido teste

boia, lat, lon = "41004", 32.5, -79.0
ano, mes = 2025, 1
ini, fim = 1, 10

horas = '/'.join(f'{h:02d}:00:00' for h in range(0, 24, 3))
arq = f'era5_{boia}_{ano}_{mes:02d}_d{ini:02d}-{fim:02d}_3h.grib'

c = cdsapi.Client()
c.retrieve('reanalysis-era5-complete', {
    'class': 'ea', 'expver': '1', 'stream': 'wave', 'type': 'an',
    'param': '251.140', 'direction': '1/to/24', 'frequency': '1/to/30',
    'date': f'{ano}-{mes:02d}-{ini:02d}/to/{ano}-{mes:02d}-{fim:02d}',
    'time': horas,
    'area': [lat, lon, lat, lon], 'grid': '0.5/0.5', 'format': 'grib',
}, arq)
print("OK", arq)