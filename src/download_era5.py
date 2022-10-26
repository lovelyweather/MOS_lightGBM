import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'temperature',
        'pressure_level': '1000',
        'year': '2022',
        'month': '09',
        'day': '01',
        'time': '00:00',
        'format': 'grib',
    },
    'download.grib')