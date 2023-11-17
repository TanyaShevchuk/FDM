from astropy import cosmology
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy import integrate
import emcee
import corner
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAnnulus, SkyCircularAperture
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import aperture_photometry, ApertureStats
import scipy
from multiprocessing import Pool

cosmo = cosmology.LambdaCDM(67.4, 0.315, 0.685)


JJ0008_0004 = ['J0008-0004', 0.440, 1.192, 6.59, 2.0123155*u.degree, -0.0689329*u.degree]  # [name,z_len,z_source,R_e, ra, dec, gal_radius]
J0728_3835 = ['J0728+3835', 0.206, 0.688, 4.21, 112.0208299* u.degree, 38.5906288* u.degree, ] #not relvant
J0841_3824 = ['J0841+3824', 0.116, 0.657, 2.96, 130.3700380* u.degree, 38.4037831* u.degree, 14.3]
J9012_0029 = ['J0912+0022', 0.164, 0.324, 4.58 ,138.0217447* u.degree, 0.4837464* u.degree, 3.4]
J0935_0003 = ['J0935-0003', 0.347, 0.467, 4.26, 143.9331588* u.degree, -0.0599282* u.degree, 2]
J0946_1006 = ['J0946+1006', 0.222, 0.609, 4.95, 146.7360642* u.degree, 10.1146756* u.degree, 1.6]
J1100_5329 = ['J1100+5329', 0.317, 0.858, 7.02, 165.1013885* u.degree, 53.4871539* u.degree, 1.8]
J1112_0826 = ['J1112+0826', 0.273, 0.629, 6.19, 168.2106716* u.degree, 8.4366443* u.degree, 2.4]
J1251_0208 = ['J1251-0208', 0.224, 0.784, 3.03, 192.8988227* u.degree, -2.1346186* u.degree, 2.4]
J1318_0313 = ['J1318-0313', 0.240, 1.300, 6.01, 199.6638624* u.degree, -3.2261945* u.degree, 2.2]
J1430_4105 = ['J1430+4105', 0.285, 0.575, 6.53, 217.5169042* u.degree, 41.0992508* u.degree, 2.6]
J1621_3931 = ['J1621+3931', 0.245, 0.602, 4.97, 245.3872647* u.degree, 39.5291026* u.degree, 4.7]
J2300_0022 = ['J2300+0022', 0.228, 0.463, 4.51, 345.2214991* u.degree, 0.3772111* u.degree, 2.1]
J2303_1422 = ['J2303+1422', 0.155, 0.517, 4.35, 345.8405304* u.degree, 14.3716864* u.degree, 4]





###      Mass functions
def sigmax(x, r_s, rho):
    # CDM(NFW) mass surface density
    # Note!! returns sigma*x, wrote that way because we need to integrate to get the mass.
    # This is an analytical result
    if x < 1:
        s = 1 / (x ** 2 - 1) * (1 - 2 / np.sqrt(1 - x ** 2) * np.arctanh(np.sqrt((1 - x) / (1 + x))))
    elif x == 1:
        s = 1 / 3
    elif x > 1:
        s = 1 / (x ** 2 - 1) * (1 - 2 / np.sqrt(x ** 2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x))))
    sigma = 2 * r_s * rho * s
    return float(sigma * x)


def m_CDM(R_e, r_s, rho):
    # Total CDM(NFW) mass inside radius R_e
    return 2 * np.pi * r_s ** 2 * (integrate.quad(sigmax, 0, R_e / r_s, args=(r_s, rho))[0])


def sigma_FDM(theta, r_NFW, r_sol, r_epsilon, ma):
    # FDM mass surface density
    # This is an analytical result as calculated by Antonio Herrera-Martín
    alpha = r_sol / r_NFW
    rho_sol = 2.4 * 10 ** 9 * r_sol ** -4 * ma ** -2  # ma=m/10**-22eV, r_sol given in kpc! rho_sol given in M_sun/kpc**3
    rho_NFW = rho_sol * r_epsilon / r_NFW * (1 + r_epsilon / r_NFW) ** 2 / (1 + (r_epsilon / r_sol) ** 2) ** 8
    if theta < r_epsilon / r_sol:
        x = np.arctan(np.sqrt(((r_epsilon / r_sol) ** 2 - theta ** 2) / (1 + theta ** 2)))
        int1 = 429 / 2048 * x + 1001 / 16384 * (
                    3 * np.sin(2 * x) + np.sin(4 * x) + 1 / 3 * np.sin(6 * x) + 1 / 11 * np.sin(
                8 * x) + 1 / 55 * np.sin(10 * x) + 1 / 429 * np.sin(12 * x) + 1 / 7007 * np.sin(14 * x))
        xx = alpha * theta
        y = alpha * r_epsilon / r_sol
        if xx < 1:
            int2 = (xx ** 2 - 1) ** -1 * (
                        1 - np.sqrt(abs(y ** 2 - xx ** 2)) / (1 + y) - 2 / np.sqrt(1 - xx ** 2) * np.arctanh(
                    np.sqrt(1 - xx ** 2) / (1 + y + np.sqrt(abs(y ** 2 - xx ** 2)))))
        elif xx == 1:
            int2 = 1 / 3 * (1 - (y + 2) / (y + 1) * np.sqrt((y - 1) / (y + 1)))
        elif xx > 1:
            int2 = (xx ** 2 - 1) ** -1 * (
                        1 - np.sqrt(abs(y ** 2 - xx ** 2)) / (1 + y) - 2 / np.sqrt(xx ** 2 - 1) * np.arctan(
                    np.sqrt(xx ** 2 - 1) / (1 + y + np.sqrt(abs(y ** 2 - xx ** 2)))))
        sigma = float(r_sol * rho_sol * (1 + theta ** 2) ** (-15 / 2) * int1 + r_NFW * rho_NFW * int2)
    else:
        x = alpha * theta
        if x < 1:
            int3 = 1 / (x ** 2 - 1) * (1 - 2 / np.sqrt(1 - x ** 2) * np.arctanh(np.sqrt((1 - x) / (1 + x))))
        elif x == 1:
            int3 = 1 / 3
        elif x > 1:
            int3 = 1 / (x ** 2 - 1) * (1 - 2 / np.sqrt(x ** 2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x))))
        sigma = rho_NFW * r_NFW * int3
    return float(2 * sigma)


def m_FDM(R_e, r_NFW, r_sol, r_epsilon, ma):
    # Total FDM mass inside radius R_e
    # Integral calculated as a sum
    # ma=m/10**-22eV
    num = 100  # number of iterations
    dr = (R_e - 0.001) / num
    r = np.linspace(0.001, R_e, num)
    M = 0
    for i in range(len(r)):
        M += r[i] * dr * sigma_FDM(r[i] / r_sol, r_NFW, r_sol, r_epsilon, ma)
    return 2 * np.pi * M


def elc_per_s_to_mass(x,FWHM,z_len,photo_key):
    #converge e/s to solar mass per pix
    d_l = cosmo.luminosity_distance(z_len)
    f = x * photo_key * FWHM #flux, erg/s/cm2/pix
    L = (f*4*np.pi*d_l**2*(1+z_len)).to(u.L_sun) #L_sun/pix
    M = 2*L/u.L_sun*u.M_sun #convert solar luminosity to solar mass/pix
    return M #sol mass/pix

def elc_per_s_to_flux(x,FWHM,photo_key):
    #converge e/s to solar mass per pix
    f = x * photo_key * FWHM #flux, erg/s/cm2/pix
    return f #flux/pix

def pix_to_kpc2(x,z_len, CD2_2): #the CD2_2 convert 1pix to deg
    pix_to_rad = CD2_2.to(u.rad)/u.rad#convert 1deg to 1rad
    d_a = cosmo.angular_diameter_distance(z_len)
    return (x*(d_a*pix_to_rad)**2).to(u.kpc**2)


def int_sigma_bar(ra,dec,image,R):
    #calculating the interpulation for the image
    hdul = fits.open(image)
    #hdul.info()
    image = hdul[1].data #exsport the image
    header = hdul[1].header #exsport table with all the data
    hdul.close()
    deg_to_aecsec = (header['CD2_2']*u.degree).to(u.arcsec)
    position = SkyCoord(ra=ra, dec=dec, frame='fk5') #dec, ra - the galaxy cooradinate
    photo_key = header['PHOTFLAM']*u.erg/u.s/u.cm**2/u.Angstrom # per pix, factor to  convert to luminosity
                                                                 #NOTE THAT THE UNITS CAN BE DIFFERENT FOR DIFFERENT GALAXIS!!
    r = np.arange(0.01,R,deg_to_aecsec/u.arcsec) #vector to extrpolate
    #aper = SkyCircularAperture(position, 1.8 * u.arcsec)
    #aper_pix = aper.to_pixel(WCS(header))
    #I_tot = ApertureStats(image, aper_pix)
    I = np.zeros(len(r))
    #I_tot = 0
    for i in range(len(r)):
        aperture = SkyCircularAnnulus(position, r_in=r[i]*u.arcsec, r_out=(r[i]*u.arcsec+deg_to_aecsec)) #the value of brightnrss on the ring
        pix_aperture = aperture.to_pixel(WCS(header))
        I[i] = ApertureStats(image, pix_aperture).mean #the mean value
        #I_tot += ApertureStats(image, pix_aperture).sum #compute the total flux
    R=r/deg_to_aecsec/u.arcsec
    intI = scipy.interpolate.UnivariateSpline(R, I)
    return intI , photo_key, header['CD2_2']*u.degree, I  #returns function and photo key

def M_bar(R_e, z_len,FWHM, interp, photo_key, CD2_2):
    # Total stellar mass inside radius R_e
    # Integral calculated as a sum
    num = 1000 #number of iterations
    dr = (R_e- 0.001) / num
    r = np.linspace(0.001, R_e, num)
    M = 0
    for i in range(len(r)):
        Sigma = elc_per_s_to_mass(interp(r[i]), FWHM, z_len, photo_key) / pix_to_kpc2(r[i], z_len, CD2_2) #units convertion to Msun/kpc2
        M += r[i] * dr * Sigma * u.kpc**2
    return 2*np.pi*M / u.M_sun

def Sigma_bar(r, FWHM, z_len, photo_key,CD2_2, interp):
    Sigma = elc_per_s_to_mass(interp(r), FWHM, z_len, photo_key) / pix_to_kpc2(r, z_len,
                                                                                  CD2_2)  # units convertion to Msun/kpc2
    return Sigma


def tot_mass(R, r_NFW_CDM, rho, r_NFW_DFM, r_sol, r_epsilon, f, ma, galaxy):
    # Total DM mass
    gal_name, z_len, z_source, R_e, ra, dec, FWHM, M_stellar = galaxy
    return f * m_FDM(R, r_NFW_DFM, r_sol, r_epsilon, ma) + (1 - f) * m_CDM(R, r_NFW_CDM,
                                                                           rho) + M_stellar(R) #+ M_bar(R, z_len,FWHM, galaxy[-1], photo_key, CD2_2)


def Sigma_cr(galaxy):
    # Calculating the critical density for a given redshift
    gal_name, z_len, z_source, R_e, ra, dec, FWHM, M_stellar = galaxy
    d_l = cosmo.angular_diameter_distance(z_len)  # given in Mpc
    d_s = cosmo.angular_diameter_distance(z_source)  # given in Mpc
    d_ls = cosmo.angular_diameter_distance_z1z2(z_len, z_source)  # given in Mpc
    G = const.G.to(u.kpc ** 3 / u.M_sun / u.s ** 2)
    c = const.c.to(u.kpc / u.s)
    Sigma_cr = c ** 2 * d_s / (4 * np.pi * G * d_ls * d_l.to(u.kpc))
    return Sigma_cr


def sigma_av(R, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy):
    # Given the average surface density
    return tot_mass(R, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy) / (np.pi * R ** 2)


def kappa(R, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy):
    # sigma_av/sigma_cr
    # kappa(R_e)=1
    return sigma_av(R, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy) / Sigma_cr(
        galaxy) / u.kpc ** 2 * u.M_sun


###       Calculating R_e using bisection
def validate_interval(function, x0, x1, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy):
    # return True/False    # need this function for the bisection
    return (function(x0, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy) - 1) * (
                function(x1, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy) - 1) < 0


def ein_radius(interval, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy, tol=10 ** -4):
    x0, x1 = interval[0], interval[1]
    # check interval can be used to solve for root
    if not validate_interval(kappa, x0, x1, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy):
        # returns a very large number if R_e is not in the interval
        return 1000
    counter = 1
    while True:
        root_approx = x0 + ((x1 - x0) / 2)  # calculate root approximation
        y = kappa(root_approx, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma,
                  galaxy)  # evaluate y at current estimate
        if -tol < y - 1 < tol:  # check tolerance condition
            break
        if validate_interval(kappa, x0, root_approx, r_NFW_CDM, rho, r_NFW_FDM, r_sol, r_epsilon, f, ma, galaxy):
            x1 = root_approx
        else:
            x0 = root_approx
        counter += 1
    return root_approx


###         MCMC functions
def log_likelihood(p, y, yerr, f, interval, ma, galaxy):
    r_sol, r_NFW_CDM, log_rho, r_NFW_FDM, r_epsilon = p  # the model parmeters we are looking for
    model = ein_radius(interval, r_NFW_CDM, 10 ** log_rho, r_NFW_FDM, r_sol, r_epsilon, f, ma,
                       galaxy)  # the model will be compere to the data
    return -0.5 * (np.sum((y - model) ** 2 / yerr ** 2))


def log_prior(p):
    r_sol, r_NFW_CDM, log_rho, r_NFW_FDM, r_epsilon = p  # the model parmeters we are looking for
    alpha = r_sol / r_NFW_FDM
    r_min = max(np.roots(np.array([13 * alpha, 15, -3 * alpha,
                                   -1]))) * r_sol  # the minimum value for r_epsilon, taken from Antonio Herrera-Martín paper
    if 0.1 < r_sol < 3 and 0.1 < r_NFW_CDM < 3 and 9 < log_rho < 13 and 0.1 < r_NFW_FDM < 3 and r_min < r_epsilon < 4:
        return 0.0
    return -np.inf


def lnprob(p, y, yerr, f, interval, ma, galaxy):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p, y, yerr, f, interval, ma, galaxy)


def walker_plot(ndim, sampler, labels, file_path, name):
    # shows the positions of each walker as a function of the number of steps
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples_ = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot((samples_[:, :, i]), "k", alpha=0.3)
        ax.set_xlim(0, len(samples_))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig(file_path + name + '_walkers_plot.png')  # saving the fig
    # plt.show()


def GRforParameter(sampMatrix):
    # Gelman-Rubin certiria
    s = np.array(sampMatrix)
    meanArr = []
    varArr = []
    n = s.shape[0]
    for samp in s:
        meanArr += [np.mean(samp)]
        varArr += [np.std(samp) ** 2]
    a = np.std(meanArr) ** 2
    b = np.mean(varArr)
    return np.sqrt((1 - 1 / n) + a / (b * n))


def MyFunction(f, galaxy, interval=[0.1, 15], initial=np.array([1, 1, 12, 1, 1]), nwalkers=10, nsteps=10000, ndim=5,
               ma=1, show_walkers=True):
    '''initial = [r_sol, r_NFW_CDM, log(rho), r_NFW_FDM, r_epsilon] array
    R_e given in kpc
    r_sol, r_NFW, r_epsilon given in Kpc
    r_s given in Kpc
    rho_CDM given in M_sum/Kpc**3
    rho_sol/NFW given in M_sol/Kpc**3
    galaxy = [name,z_len,z_source,R_e]'''

    gal_name, z_len, z_source, R_e, ra, dec, FWHM, M_stellar = galaxy
    Re_err = 0.05 * R_e

    file_path = ''#'/gpfs0/elyk/users/shevchut/run_mcmc_notebook/'+galaxy[0]+'/ma_'+str(ma)+'/'
    name = gal_name + '_ma_' + str(ma) + 'f_' + str(f)  # name the files will save at

    pos = [np.array(initial) + 10 ** -2 * np.random.randn(ndim) for i in
           range(nwalkers)]  # the initial geuss for the parameters we looking for

    GR_vec = np.empty(0)
    loopcriteria = True
    t = 0
    while loopcriteria:
        with Pool() as pool:
            # backend = emcee.backends.HDFBackend(file_path+name+'h5', read_only=False)
            # backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,pool=pool, args=(R_e, Re_err, f, interval, ma, galaxy))
            pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)
            samples = sampler.get_chain()
            GR = np.empty(ndim)
            for j in range(ndim):
                GR[j] = (GRforParameter(samples[:, :, j]))
            GR_vec = np.append(GR_vec, GR, axis=0)
            t += 1
            nsteps += nsteps
            flat_samples = sampler.get_chain(discard=0, flat=True)
            loopcriteria = not all(np.absolute(1 - GR) < 10 ** -4)
            if nsteps == 16000:
                break
    np.save(file_path + name + '_samples_chain', flat_samples)
    np.save(file_path + name + '_Gelman_Rubin', np.resize(GR_vec, (t, ndim)))
    np.save(file_path + name + '_nsteps', nsteps)
    np.save(file_path + name + '_log_prob', sampler.get_log_prob(flat=True))

    # corner plot
    labels = [r'$(r_{sol})[kpc]$', r'$(r_{NFW,CDM})[kpc]$', r'$log(\rho)[\frac{M_\odot}{kpc^3}]$',
              r'$(r_{NFW,FDM})[kpc]$', r'$(r_\epsilon)[kpc]$']
    fig = corner.corner(flat_samples, labels=labels, show_titles=True, plot_datapoints=True,
                        quantiles=[0.16, 0.5, 0.84])  # ploting corner plot of the parameters
    plt.savefig(file_path + name + '_corner_plot.png')  # saving the fig
    fig.show()

    # shows the positions of each walker as a function of the number of steps
    if show_walkers == True:
        walker_plot(ndim, sampler, labels, file_path, name)


def tot_mass_len_eq(galaxy):
    # the mass as function of R_e, the rhs of the len eq.
    z_len = galaxy[1]
    z_source = galaxy[2]
    d_l = cosmo.angular_diameter_distance(z_len)  # given in Mpc
    d_s = cosmo.angular_diameter_distance(z_source)  # given in Mpc
    d_ls = cosmo.angular_diameter_distance_z1z2(z_len, z_source)  # given in Mpc
    R_e = galaxy[3] * u.kpc
    D = d_l * d_s / d_ls.to(u.kpc)
    G = const.G.to(u.kpc ** 3 / u.M_sun / u.s ** 2)
    c = const.c.to(u.kpc / u.s)
    theta_e = R_e / d_l
    return c ** 2 / (4 * G) * D * theta_e ** 2  # units of M_sun

def mass_plot(galaxy, f, ma):
    # ploting the total galaxy mass as function of r for different values of f
    gal_name, z_len, z_source, R_e, ra, dec, FWHM, M_stellar = galaxy
    M = tot_mass_len_eq(galaxy)
    file_path = '/mnt/DATA/Results/' + gal_name + '_new4/ma_' + str(ma) + '/'
    color = ['dodgerblue','forestgreen','darkorange']
    for fi,c in zip(f,color):
        name = galaxy[0] + '_ma_' + str(ma) + 'f_' + str(fi)
        probs = np.exp(np.load(file_path + name + '_log_prob.npy'))  # the log_probability of each values set
        samples = np.load(file_path + name + '_samples_chain.npy')  # Getting the chain
        result = np.where(probs == np.amax(probs))  # Taking the most probable values
        r_sol, r_CDM, log_rho, r_FDM, r_epsilon = samples[result][0]  # The model parameters best fit
        r = np.linspace(0.1, 1.2 * R_e, 100)
        m = np.empty(len(r))
        for i in range(len(r)):
            m[i] = tot_mass(r[i], r_CDM, 10 ** log_rho, r_FDM, r_sol, r_epsilon,fi, ma, galaxy)
        p = np.loadtxt(file_path+name+'_percentile')
        plt.loglog(r, m, label=f'f={fi}',color=c, )
        #plt.fill_between(r,p[0],p[-1],alpha=0.3,color=c,label=r'$2\sigma$')
        #plt.fill_between(r, p[1], p[3], alpha=0.5,color=c)
    plt.axhline(M / u.M_sun, linewidth=1, color='black', linestyle='--', label=r'$M_e$')
    plt.axvline(R_e, lw=1, color='black', linestyle='-.', label=r'$R_e$')
    plt.xlabel(r'$r[kpc]$',fontsize = 12)
    plt.ylabel(r'$M[M_\odot]$',fontsize = 12)
    #plt.title(r'Total mass as function of the radius $m_a=$' + str(ma) + r'$\cdot 10^{-22}$[eV]')
    #plt.title(r'Total mass as function of the radius $m_a=1\cdot 10^{-24}$[eV]',fontsize = 15)
    plt.legend()
    #plt.xlim([4,5])
    #plt.ylim([3.e11,5.e11])
    plt.savefig(file_path + '/Total_mass_as_function_of_the_radius' + str(ma) + '.png')
    plt.show()

def mcmc_percentile(ma, fi, galaxy, one_point=True, percentiles=[2.5, 16, 50, 84, 97.5], n=100):
    gal_name, z_len, z_source, R_e, ra, dec,_, M_stellar = galaxy

    file_path = '/mnt/DATA/Results/'+gal_name+'/ma_'+str(ma)+'/'
    name = gal_name + '_ma_' + str(ma) + 'f_' + str(fi)  # name the files will save at

    # Extract chain
    samples = np.load(file_path + name + '_samples_chain.npy')
    err = np.empty([len(samples)])
    r = np.linspace(0.1, 1.2 * R_e, n)
    dist = np.empty([len(samples), n])
    for j in range(0, len(samples), 1):
        r_sol, r_CDM, log_rho, r_FDM, r_epsilon = samples[j]
        if one_point == True:
            # Compute function for each MCMC sample for one point
            err[j] = tot_mass(R_e, r_CDM, 10 ** log_rho, r_FDM, r_sol, r_epsilon, fi,ma, galaxy)
        else:
            # Compute function for each MCMC sample for array
            for i in range(n):
                dist[j, i] = tot_mass(r[i], r_CDM, 10 ** log_rho, r_FDM, r_sol, r_epsilon,fi, ma, galaxy)
    # Compute percentiles
    p = np.percentile(err, percentiles, axis=0) if one_point == True else np.percentile(dist, percentiles, axis=0)
    # Save percentiles to file
    np.savetxt(file_path + name + '_percentile', p, delimiter=' ')
    return p


#gal_name, z_len, z_source, R_e, ra, dec, FWHM, gal_image = galaxy
#interp, photo_key, CD2_2 = int_sigma_bar(ra, dec, gal_image)  # caling the interpulation for the stellar mass
#galaxy.append(photo_key)
#galaxy.append(CD2_2)


