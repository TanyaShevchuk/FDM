import os

#m = [0.025,0.03,0.04,0.045,0.05,0.055,0.06,0.065]
#m = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075]
m = [0.08,0.085,0.09,0.095,0.1,0.15,0.2]#,0.25,0.3,0.35,0.4,0.45]
#m = [0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08]

gal_name = 'J1251-0208'

#os.system('rm -r '+gal_name)
#os.system('mkdir '+gal_name)

command = 'mkdir '+gal_name+'/ma_'
for mi in m:
    os.system(command+str(mi))

command = 'qsub -pe shared 1 FDM.sh'
for mi in m:
    # Read the bash file
    bash_file = open("FDM.sh", "r")
    bash_file_lines = bash_file.readlines()
    bash_file.close()
    # Change the command in the bash file
    bash_file_lines[14] = 'python J1251_0208.py ' + str(mi)
    # Write the new command to the bash file
    bash_file = open("FDM.sh","w")
    bash_file.writelines(bash_file_lines)
    bash_file.close()
    # Submit the job to the queue
    os.system(command)
