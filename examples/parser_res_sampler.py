
import glob
import numpy as np
from astropy.table import Table
files=glob.glob('slurm-*.out')
type_likelihoods=[]
type_sampler=[]
dimensions=[]
populations=[]
logZs=[]
logZ_errs=[]
nb_call_likelihood=[]
ESS=[]
time=[]
U_test_convergence=[]
U_test_nb_it_corr=[]
for file in files:
    type_opti=False
    logZ_founded=False
    time_founded=False
    ESS_founded=False
    U_test_founded=False
    nb_call_likelihood_founded=False
    try:
        lines=open(file).readlines()
        for i in range(min(len(lines),300)):
            if len(lines[-i-1])<2: continue
            line=lines[-i-1][:-1]
            #print(len(line))
            if len(line)>10:
                if line[:9]=="Namespace":
                    name_dir=line.split("log_dir='")[1].split("',")[0]
                    name_dir=name_dir.split("_")
                    dimensions.append(int(name_dir[2]))
                    type_likelihoods.append(name_dir[0])
                    type_sampler.append(name_dir[1])
                    populations.append(int(name_dir[4]))
                    type_opti=True
            if len(line)>4:
                 if line[:4]=="real":
                    time.append(float(line.split("real")[1]))
                    time_founded=True
            if len(line)>4:
                 if line[:4]=="logZ":
                    logz_all=line.split("logZ =")[1].split("+-")
                    logZs.append(float(logz_all[0]))
                    logZ_errs.append(float(logz_all[1]))
                    logZ_founded=True
            line_ESS="[ultranest] Effective samples strategy satisfied (ESS ="
            if len(line)>len(line_ESS):
                if line[:len(line_ESS)]==line_ESS:
                    ESS.append(float(line.split(line_ESS)[1].split(",")[0]))
                    ESS_founded=True
            line_nb_call="[ultranest] Likelihood function evaluations:"
            if len(line)>len(line_nb_call):
                if line[:len(line_nb_call)]==line_nb_call and not nb_call_likelihood_founded:
                    nb_call_likelihood.append(int(line.split(line_nb_call)[1]))
                    nb_call_likelihood_founded=True
            line_U_test="insert order U test :"
            if len(line)>len(line_U_test):
                if line[:len(line_U_test)]==line_U_test:
                    U_test_conv=bool(line.split(line_U_test+" converged: ")[1].split("correlation")[0])
                    nb_it_corr=line.split(line_U_test+" converged: ")[1].split("correlation:")[1].split("iterations")[0]
                    nb_it_corr= np.inf if nb_it_corr==' inf ' else int(nb_it_corr)
                    U_test_convergence.append(U_test_conv)
                    U_test_nb_it_corr.append(nb_it_corr)
                    U_test_founded=True
            if type_opti and logZ_founded and time_founded and ESS_founded and nb_call_likelihood_founded and U_test_founded:
                break
        if not ESS_founded and type_opti and logZ_founded and time_founded and nb_call_likelihood_founded and U_test_founded:
            ESS_founded=True
            ESS.append(0)
        if not type_opti or not logZ_founded or not time_founded or not ESS_founded or not nb_call_likelihood_founded or not U_test_founded:
            if type_opti:
                type_likelihoods.pop()
                type_sampler.pop()
                dimensions.pop()
                populations.pop()
            if logZ_founded:
                logZs.pop()
                logZ_errs.pop()
            if time_founded:
                time.pop()
            if ESS_founded:
                ESS.pop()
            if nb_call_likelihood_founded:
                nb_call_likelihood.pop()
            if U_test_founded:
                U_test_convergence.pop()
                U_test_nb_it_corr.pop()
        
    except:
        continue

dict_Table={"type_likelihood":type_likelihoods,"type_sampler":type_sampler,"dimensions":dimensions,"populations":populations,"logZ":logZs,"logZ_err":logZ_errs,"nb_call_likelihood":nb_call_likelihood,"ESS":ESS,"time":time,"U_test_convergence":U_test_convergence,"U_test_nb_it_corr":U_test_nb_it_corr}
print(len(type_likelihoods),len(type_sampler),len(dimensions),len(populations),len(logZs),len(logZ_errs),len(nb_call_likelihood),len(ESS),len(time),len(U_test_convergence),len(U_test_nb_it_corr))
Table(dict_Table).write("results_sampler.fits",format="fits",overwrite=True)
