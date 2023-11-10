import csv
import datetime

PV_FILE = 'data/fotowoltaika.csv'
METEO_FILE = 'data/pogoda.csv'
OUTPUT_FILE = 'data/merged.csv'

# Column names in the output file
output_fieldnames = ['time','energy','temperature_2m','weathercode',
    'cloudcover', 'cloudcover_low','cloudcover_mid','cloudcover_high',
    'shortwave_radiation', 'direct_radiation','diffuse_radiation',
    'direct_normal_irradiance','terrestrial_radiation']


# Synchronizes timestamps in both readers
def sync():
    global meteo_reader
    global pv_reader

    meteo = next(meteo_reader)
    pv = next(pv_reader)

    while True:
        meteo_time = int(meteo['time'])
        pv_time = int(pv['timestamp'])

        if meteo_time < pv_time:
            meteo = next(meteo_reader)
        elif pv_time < meteo_time:
            pv = next(pv_reader)
        else:   # so they are equal
            assert(pv_time == meteo_time)
            return pv_time


with open(PV_FILE, 'r') as pv_file, open(METEO_FILE, 'r') as meteo_file, open(OUTPUT_FILE, 'w') as output_file:
    pv_reader = csv.DictReader(pv_file)
    meteo_reader = csv.DictReader(meteo_file)
    output = csv.DictWriter(output_file, fieldnames=output_fieldnames)

    output.writeheader()

    pv_time = meteo_time = sync()

    # merging loop (main)
    while True:
        try:
            meteo_time_start = meteo_time
            meteo = next(meteo_reader)
            meteo_time = int(meteo['time'])

            if not meteo['temperature_2m']: # end of useful data
                break

            # integrate power data to calculate the energy
            energy_cnt = 0

            while pv_time < meteo_time:
                pv = next(pv_reader)

                pv_time = int(pv['timestamp'])
                pv_power = int(pv['power'])
                energy_cnt += pv_power

            # check if the period is one hour, energy_cnt must
            # be scaled to convert watts (power) to watt-hours (energy)
            assert(meteo_time - meteo_time_start == 3600)

            new_row = meteo
            new_row['energy'] = energy_cnt
            #new_row['time'] = datetime.datetime.utcfromtimestamp(int(new_row['time'])).isoformat()
            output.writerow(new_row)

        except StopIteration:
            break
