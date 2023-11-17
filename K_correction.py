from scipy import interpolate
import numpy as np
import csv

def creating_lists(data_file):
    # output is two lists (for each column). works for SED files or space delimiter csv files.
    with open(data_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        wavelength = []
        value = []
        for row in csv_reader:
            count = 0
            for i in row:
                if i != "":
                    if count == 0:
                        wavelength.append(float(i))
                        count += 1
                    else:
                        value.append(float(i))
    return wavelength, value

def k_correction(z):
    "Takes a 814AB observed magnitude at redshift Z and find the absolute Bband Vega magnitude (redshift 0) using K correction"
    # loading relevnt files to lists - 1st arg wavelength, 2nd arg flux
    SED = list(creating_lists(r"k_correction_files-20230627T111219Z-001/k_correction_files/El_B2004a.sed"))
    QB_band = list(creating_lists(r"k_correction_files-20230627T111219Z-001/k_correction_files/B_Johnson.res"))
    R814 = list(creating_lists(r"k_correction_files-20230627T111219Z-001/k_correction_files/HST_ACS_WFC_F814W.res"))
    vega_spec = list(creating_lists(r"k_correction_files-20230627T111219Z-001/k_correction_files/vega_spec.csv"))
    I_band = list(creating_lists(r'k_correction_files-20230627T111219Z-001/k_correction_files/i_SDSS.res'))

    band = QB_band

    # creating a grid for QB band and interpolate the missing wavelengths
    tck = interpolate.splrep(band[0], band[1], s=0)
    band[0] = np.arange(np.ceil(min(band[0])), np.ceil(max(band[0])) + 1, 1)
    band[1] = interpolate.splev(band[0], tck, der=0)

    # matching SED and Vega to band grid and interpolate the missing wavelengths
    # notice we need the SED for redshifted values - lambda_observed*(1+z)
    tck = interpolate.splrep(SED[0], SED[1], s=0)
    SED_QB_shifted = []
    SED_QB_shifted.append([x * (1 + z) for x in band[0]])
    SED_QB_shifted.append(interpolate.splev(SED_QB_shifted[0], tck, der=0))

    tck = interpolate.splrep(vega_spec[0], vega_spec[1], s=0)
    vega_QB = []
    vega_QB.append(band[0])
    vega_QB.append(interpolate.splev(vega_QB[0], tck, der=0))

    # creating a grid for R814 and interpolate the missing wavelengths
    tck = interpolate.splrep(R814[0], R814[1], s=0)
    R814[0] = np.arange(np.ceil(min(R814[0])), np.ceil(max(R814[0])) + 1, 1)
    R814[1] = interpolate.splev(R814[0], tck, der=0)

    # matching SED to R814 grid
    SED[0] = [x * (1 + z) for x in SED[0]]  # ADDED THIS - IS THERE ANY AFFECT ON THE VALUES??
    tck = interpolate.splrep(SED[0], SED[1], s=0)
    SED_R814 = []
    SED_R814.append(R814[0])
    SED_R814.append(interpolate.splev(SED_R814[0], tck, der=0))

    # first integral (for lambda observed at 814 band)- dlambda*lamabda*flux(lambda)*R814(lambda), for us dlambda=1
    int1 = 0
    for i in range(len(R814[0])):
        int1 += R814[0][i] * SED_R814[1][i] * R814[1][i]

    # second integral (for lambda emitted at b band)- dlambda*lamabda*vega_spec(lambda)*band(lambda),
    # for us dlambda=1
    int2 = 0
    for i in range(len(band[0])):
        int2 += band[0][i] * vega_QB[1][i] * band[1][i]

    # third integral (for lambda observed at 814 band) - dlambda*lambda*gR(lambda)*R814(lambda)
    # where gR(lambda_obs) is the calibrator spectrum, for AB mag this goes as c * (lambda_emit ^ -2) * gR(ni_obs)
    # gR(ni_obs) = 3631Jy
    int3 = 0
    for i in range(len(R814[0])):
        int3 += R814[0][i] * (3631 / (3.335 * (10 ** 4) * pow(R814[0][i], 2))) * R814[1][i]

    # forth integral - dlambda*lamabda*flux(lambda*[1+z])*band(lambda), for us dlambda=1
    int4 = 0
    for i in range(len(band[0])):
        int4 += band[0][i] * SED_QB_shifted[1][i] * band[1][i]

    # Calculate K-QR
    K_QR = -2.5 * np.log10((int1 * int2) / (int3 * int4 * (1 + z)))

    return (K_QR)

print(k_correction(0.5))