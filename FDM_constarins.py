import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"})

# Prepare figure
plt.figure(figsize=(18,8))
# Define density parameters (to convert the other constraints from FDM density parameter to FDM fraction)
Omega_m = 0.3153
Omega_b = 0.0493
# Define the dictionary for the other constraints
constraints = {}
#constraints['Eri_II'] = {'label':r'\textbf{Eri II}','x':0.75,'y':0.3,
 #                       'label color':'#B58E74','color':'#B58E74','fontsize':30}
constraints['SPARC'] = {'label':r'\textbf{SPARC}','x':0.7,'y':0.62,
                        'label color':'#C300FF','color':'#C300FF','fontsize':30}
constraints['Lyaf1'] = {'label':r'\textbf{Ly$\alpha$f}','x':0.835,'y':0.68,
                        'label color':'#FF621E','color':'orange','fontsize':35}
constraints['Rei'] = {'label':r'\textbf{Rei}','x':0.84,'y':0.835,
                      'label color':'#00F900','color':'#36753B','fontsize':28}
constraints['BOSS'] = {'label':r'\textbf{+BOSS}','x':0.3,'y':0.57,
                       'label color':'#1300F5','color':'#1300F5','fontsize':30}
constraints['CMB'] = {'label':r'\textbf{CMB}','x':0.17,'y':0.77,
                      'label color':'#56C1FF','color':'#0076BA','fontsize':33}
constraints['DES'] = {'label':r'\textbf{+DES}','x':0.43,'y':0.82,
                      'label color':'#F53DE7','color':'#F53DE7','fontsize':30}
#constraints['CMBS4'] = {'label':r'\textbf{CMB-S4}','x':0.32,'y':0.4,
 #                       'label color':'#56C1FF','color':'#56C1FF','fontsize':30}
#constraints['M87'] = {'label':r'\textbf{M87}','x':0.83,'y':0.35,
 #                       'label color':'black','color':'grey','fontsize':30}
'''
constraints['OVS4'] = {'label':r'\textbf{OV-S4}','x':0.2,'y':0.25,
                       'label color':'purple','color':'purple','fontsize':30}
constraints['kSZS4'] = {'label':r'\textbf{kSZ-S4}','x':0.2,'y':0.48,
                        'label color':'pink','color':'pink','fontsize':25}
'''

f = np.array([0.1,0.1,0.1272727273,0.2454545455,0.3727272727,0.5454545455,0.6909090909,0.7454545455,0.7909090909,
0.8272727273,0.8545454545,0.8818181818,0.9,0.9181818182,0.9454545455,0.9454545455,0.9545454545,
0.9545454545,0.9636363636,0.9727272727,1,1])
m = np.array([0.0001,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.15,1])
#plt.fill_between(m*10**-22,f,np.ones(len(m)),color='gold')



# For each constraint...
for constraint in constraints.keys():
    # Extract the constraint from the file
    filename = ""+constraint+"_data.dat"
    m_FDM_other, Omega_FDM_thresh_other = np.loadtxt(filename,unpack=True,usecols=[0,1])
    # Convert from FDM density parameter to FDM fraction
    if not constraint in ['M87', 'SPARC']:
        f_FDM_thresh_other = Omega_FDM_thresh_other/(Omega_m-Omega_b)
    else:
        f_FDM_thresh_other = Omega_FDM_thresh_other
    # Fill area for the following constraints
    if constraint in ['BOSS','CMB','DES','Lyaf1','Rei','M87','SPARC','Eri_II']:
        plt.fill_between(m_FDM_other, f_FDM_thresh_other,np.ones(len(m_FDM_other)),
                         color=constraints[constraint]['color'])
    # Plot the curves of the other constraints
    else:
        plt.plot(m_FDM_other, f_FDM_thresh_other,linewidth=5.,
                 color=constraints[constraint]['color'],zorder=1)
    # Add the label of the constraint
    plt.figtext(x=constraints[constraint]['x'],
                y=constraints[constraint]['y'],
                s=constraints[constraint]['label'],
                color=constraints[constraint]['label color'],
                fontsize=constraints[constraint]['fontsize'])

#plt.figtext(x=0.45,y=0.6,s=r'\textbf{SL}',
 #           color='darkgoldenrod',fontsize=35)
# Prettify figure
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticks=np.logspace(-28.,-18.,11),fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([1.5e-26,2.e-23])
plt.ylim([1.5e-2,1.])
plt.xlabel('$m_a\,[\mathrm{eV}]$',fontsize=25)
plt.ylabel('$f_\mathrm{FDM}$',fontsize=25)
plt.savefig('/mnt/DATA/Results/Results window')
plt.show()
