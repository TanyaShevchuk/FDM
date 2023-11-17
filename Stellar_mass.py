import Main as T
from astropy import units as u
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy import cosmology
import K_correction as K_corr

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"})

cosmo = cosmology.LambdaCDM(67.4, 0.315, 0.685)



J0008_0004 = ['J0008-0004', 0.440, 1.192, 6.59, 2.0123155*u.degree, -0.0689329*u.degree]  # [name,z_len,z_source,R_e, ra, dec, gal_radius]
J0728_3835 = ['J0728+3835', 0.206, 0.688, 4.21, 112.0208299* u.degree, 38.5906288* u.degree, ] #not relvant
J0841_3824 = ['J0841+3824', 0.116, 0.657, 2.96, 130.3700380* u.degree, 38.4037831* u.degree, 14.3]
J9012_0029 = ['J0912+0022', 0.164, 0.324, 4.58, 138.0217447* u.degree, 0.4837464* u.degree, 3.4]
J0935_0003 = ['J0935-0003', 0.347, 0.467, 4.26, 143.9331588* u.degree, -0.0599282* u.degree, 2]
J0946_1006 = ['J0946+1006', 0.222, 0.609, 4.95, 146.7360642* u.degree, 10.1146756* u.degree, 1.6]
J0912_0029 = ['J0912+0022', 0.164, 0.324, 4.58, 138.0217447* u.degree, 0.4837464* u.degree, 3.4]
J1100_5329 = ['J1100+5329', 0.317, 0.858, 7.02, 165.1013885* u.degree, 53.4871539* u.degree, 1.8]
J1112_0826 = ['J1112+0826', 0.273, 0.629, 6.19, 168.2106716* u.degree, 8.4366443* u.degree, 2.4]
J1251_0208 = ['J1251-0208', 0.224, 0.784, 3.03, 192.8988227* u.degree, -2.1346186* u.degree, 4.3]
J1318_0313 = ['J1318-0313', 0.240, 1.300, 6.01, 199.6638624* u.degree, -3.2261945* u.degree, 2.2]
J1430_4105 = ['J1430+4105', 0.285, 0.575, 6.53, 217.5169042* u.degree, 41.0992508* u.degree, 2.6]
J1621_3931 = ['J1621+3931', 0.245, 0.602, 4.97, 245.3872647* u.degree, 39.5291026* u.degree, 4.7]
J2300_0022 = ['J2300+0022', 0.228, 0.463, 4.51, 345.2214991* u.degree, 0.3772111* u.degree, 2.1]
J2303_1422 = ['J2303+1422', 0.155, 0.517, 4.35, 345.8405304* u.degree, 14.3716864* u.degree, 4]


galaxies = [J0841_3824,J0935_0003,J0946_1006,J1100_5329,J1112_0826,J1251_0208,J1318_0313,J1430_4105,J1621_3931,J2300_0022,J2303_1422]
#galaxies=[J1251_0208]#[J1318_0313]


def c_to_s_to_F_nu(x,photo_key,lamda=8060):
    #converge e/s to flux
    f_lam = x * photo_key  #flux, erg/s/cm2/pix/A
    f_nu = 3.34*10**(4+9)*lamda**2*f_lam/(u.erg/u.Angstrom/u.s/u.cm**2)
    return f_nu #returns F_nu in nJy

def m_AB(F_nu):
    #returns the mAB magnitude for given flux/herz in nano J
    return 31.4-2.5*np.log10(F_nu)

def DM(z):
    d_l = cosmo.luminosity_distance(z)
    return 5*np.log10(d_l/(10*u.pc))

def M_abs(mAB,DM,K):
    return mAB-K-DM

def L(M):
    return 147.91*10**(-0.4*M)


def fraction_plot(galaxies):
    ff=[]
    mm=[]
    fig = plt.figure()
    ax = fig.add_subplot()
    for galaxy in galaxies:
        f_of_m = galaxy[-1]
        ma = np.array([0.0001])
        ma = np.append(ma,f_of_m.T[:][0])
        ma = np.append(ma,1)
        f = np.array([0.1])
        f = np.append(f,f_of_m.T[:][1])
        f = np.append(f,1)
        ff = np.append(ff, f_of_m.T[:][1])
        mm = np.append(mm,f_of_m.T[:][0])
        plt.loglog(ma*10**-22,f,label=galaxy[0])
    f=np.array([0.1,0.1,	0.1090909091,0.1090909091,	0.1181818182,	0.2727272727,	0.5,	0.7090909091,	0.8363636364,	0.8636363636,	0.9,	0.9272727273,	0.9454545455,	0.9818181818,1,	1])
    m = np.array([0.0001,0.0085,0.009,0.0095,	0.01,	0.015,	0.02,	0.025,	0.03,	0.035,	0.04,	0.045,	0.05,	0.055,0.06,1])
    plt.loglog(m*10**-22,f,color='black',label='Mean fraction')
    plt.xlabel(r'$m_a$[eV]')
    plt.ylabel('f')
    plt.title('Fraction as function of bosonic mass')
    plt.legend()
    plt.ylim([0.05,1.1])
    plt.savefig('/mnt/DATA/Results/f as function of m.png')
    plt.show()

def max_rho_plot():
    def rho(x,e):
        return e*x*(1+e*x)**2*(1+x**2)**-8

    e =[ 1.5, 1, 0.5]
    x = np.linspace(0,1.75,100)
    for ei in e:
        plt.plot(x, rho(x,ei), label=r'$\frac{r_{sol}}{r_{CDM}}=$'+str(ei))
    plt.legend(fontsize = 12)
    plt.title(r'$\frac{\rho_{FDM}}{\rho_{sol}}$ as function of $\frac{r_\epsilon}{r_{sol}}$',fontsize = 15)
    plt.xlabel(r'$\frac{r_\epsilon}{r_{sol}}$',fontsize = 14)
    plt.ylabel(r'$\frac{\rho_{FDM}}{\rho_{sol}}$',fontsize = 14)
    plt.savefig('/mnt/DATA/Results/rho FDM as function of r epsilon.png')
    plt.show()

r = np.linspace(0.001,12,100)
def stellar_plot(galaxies,save=True):
    for galaxy in galaxies:
        gal_name, z_len, z_source, R_e, ra, dec, R = galaxy
        print(gal_name)
        gal_image = '/mnt/DATA/ds/'+gal_name+'.fits'
        interp, photo_key, CD2_2, I0 = T.int_sigma_bar(ra, dec, gal_image,R)
        Dm = DM(z_len)
        K = K_corr.k_correction(z_len)
        F_nu = c_to_s_to_F_nu(interp(r), photo_key)
        mAB = m_AB(F_nu)
        Mabs = M_abs(mAB, Dm, K)
        Lum = L(Mabs)
        M = 2 * Lum
        sigma = M / T.pix_to_kpc2(r, z_len, CD2_2)
        plt.loglog(r,sigma, label=gal_name)
        intS = scipy.interpolate.UnivariateSpline(r, sigma)
        print(format(intS(R_e),'.2E'))

        # Save the interpolated function
        #with open('/mnt/DATA/Results/' + gal_name + '_stellar_density.pickle', 'wb') as f:
         #   pickle.dump(intS, f)

    plt.ylabel(r'$\Sigma_{bar}(r) [\frac{M_\odot}{kpc^2}]$',fontsize = 13)
    plt.xlabel(r'$r[kpc]$',fontsize = 13)
    plt.title('Stellar surface mass density as a function of r',fontsize = 15)
    plt.legend()
    if save==True:
        plt.savefig('/mnt/DATA/Results/Sigma_bar_new.png')
    plt.show()

#stellar_plot(galaxies)

print(format(T.tot_mass_len_eq(J1318_0313),'.2E'))



print('end')
