import cdsapi
import calendar
import math
import os
import time

# boia: (lat, lon, margem). margem 0 = ponto único; 0.5 = quadrado de ±0,5°
boias = {
    "41004": (32.5, -79.0, 0.0),
    "41040": (14.5, -53.0, 0.0),
    "46080": (58.0, -150.0, 0.5),
}

# em ordem: (boia, ano, mês)
tarefas = [
    ("41004", 2025, 1),
    ("41040", 2019, 1),
    ("46080", 2025, 1),
]

horas = '/'.join(f'{h:02d}:00:00' for h in range(0, 24, 3))

def blocos_do_mes(ano, mes):
    ultimo = calendar.monthrange(ano, mes)[1]
    n = math.ceil(ultimo / 10)
    tam = math.ceil(ultimo / n)
    return [(i, min(i + tam - 1, ultimo)) for i in range(1, ultimo + 1, tam)]

def pedir(boia, ano, mes, ini, fim):
    lat, lon, m = boias[boia]
    area = [lat + m, lon - m, lat - m, lon + m]
    arq = f'era5_{boia}_{ano}_{mes:02d}_d{ini:02d}-{fim:02d}_3h.grib'
    if os.path.exists(arq):
        print("já existia", arq, flush=True)
        return
    c = cdsapi.Client()
    while True:
        try:
            c.retrieve('reanalysis-era5-complete', {
                'class': 'ea', 'expver': '1', 'stream': 'wave', 'type': 'an',
                'param': '251.140', 'direction': '1/to/24', 'frequency': '1/to/30',
                'date': f'{ano}-{mes:02d}-{ini:02d}/to/{ano}-{mes:02d}-{fim:02d}',
                'time': horas,
                'area': area, 'grid': '0.5/0.5', 'format': 'grib',
            }, arq + '.tmp')
            os.rename(arq + '.tmp', arq)
            print("OK", arq, flush=True)
            return
        except Exception as e:
            if "queued requests" in str(e):
                print("fila cheia, espero 10 min:", arq, flush=True)
                time.sleep(600)
                continue
            print("ERRO", arq, e, flush=True)
            return

for boia, ano, mes in tarefas:
    blocos = blocos_do_mes(ano, mes)
    # janeiro/2025 da 41004: o bloco 1-10 já está na fila, é baixado à parte
    if (boia, ano, mes) == ("41004", 2025, 1):
        blocos = [(11, 20), (21, 25), (26, 31)]
    for ini, fim in blocos:
        pedir(boia, ano, mes, ini, fim)