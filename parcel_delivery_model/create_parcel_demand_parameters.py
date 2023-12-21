# Factor to go from year to average delivery day
year_factor = (6 / 7) * 365

# Factor to increase or decrease parcel demand
# TODO: Apply growth factor for base year 2023 and forecast years
growth_factor = 1.0

# Shares of B2B and B2C (+C2C) parcel deliveries
# Source (DE, 2021): https://www.biek.de/files/biek/downloads/papiere/BIEK_KEP-Studie_2022.pdf (page 20)
perc_b2b = 0.23
perc_b2c = 0.71 + 0.06

# Total number of parcel deliveries
# (Swiss Post: 201.1 mln., minus 2-3 mln. export)
# (Private: 50.0 mln.)
n_parcels_total = (201.1 - 2.5 + 50.0) * 1000000

# Total employment and population
# Source (CH, 2013): FTE_pro_ZoneNPVM.csv
# TODO: Use 2021 zone statistics instead of 2013
sum_jobs = 3880099
sum_persons = 8139631

# Calculations
n_parcels_b2b = n_parcels_total * perc_b2b
n_parcels_b2c = n_parcels_total * perc_b2c

n_parcels_per_job = n_parcels_b2b / sum_jobs
n_parcels_per_person = n_parcels_b2c / sum_persons

n_parcels_per_job /= year_factor
n_parcels_per_person /= year_factor

n_parcels_per_job *= growth_factor
n_parcels_per_person *= growth_factor

print('n_parcels_per_job:', round(n_parcels_per_job, 4))
print('n_parcels_per_person:', round(n_parcels_per_person, 4))

p_attempt_1 = 0.908
p_attempt_2 = 0.038

average_num_attempts = (
    1 * p_attempt_1 +
    2 * (1.0 - p_attempt_1) * p_attempt_2 +
    3 * (1.0 - p_attempt_1) * (1.0 - p_attempt_2))

print('average_num_attemps:', round(average_num_attempts, 4))
