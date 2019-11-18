import numpy as np
import matplotlib.pyplot as plt
import os
from pylab import *
import pprint

##### Parameters ######
filename = sys.argv[-1] # log filename
cl_to_plot_id = 1 # Target class, typically the low frequent one
if 'reddit' in filename or ('bitcoin' in filename and 'edge' in filename):
    cl_to_plot_id = 0 # 0 for reddit dataset or bitcoin edge cls

simulate_early_stop = 0 # Early stop patience
eval_k = 1000 # to compute metrics @K (for instance precision@1000)
print_params = True # Print the parameters of each simulation
##### End parameters ######

if 'elliptic' in filename or 'reddit' in filename or ('bitcoin' in filename and 'edge' in filename):
	target_measure='f1' # map mrr f1 p r loss avg_p avg_r avg_f1
else:
    target_measure='map' # map mrr f1 p r loss avg_p avg_r avg_f1


# Hyper parameters to analyze
params = []
params.append('learning_rate')
params.append('num_hist_steps')
params.append('layer_1_feats')
params.append('lstm_l1_feats')
params.append('class_weights')
params.append('adj_mat_time_window')
params.append('cls_feats')
params.append('model')


res_map={}
errors = {}
losses = {}
MRRs = {}
MAPs = {}
prec = {}
rec = {}
f1 = {}
prec_at_k = {}
rec_at_k = {}
f1_at_k = {}
prec_cl = {}
rec_cl = {}
f1_cl = {}
prec_at_k_cl = {}
rec_at_k_cl = {}
f1_at_k_cl = {}
best_measure = {}
best_epoch = {}

last_test_ep={}
last_test_ep['precision'] =  '-'
last_test_ep['recall'] = '-'
last_test_ep['F1'] = '-'
last_test_ep['AVG-precision'] = '-'
last_test_ep['AVG-recall'] = '-'
last_test_ep['AVG-F1'] = '-'
last_test_ep['precision@'+str(eval_k)] =  '-'
last_test_ep['recall@'+str(eval_k)] = '-'
last_test_ep['F1@'+str(eval_k)] = '-'
last_test_ep['AVG-precision@'+str(eval_k)] =  '-'
last_test_ep['AVG-recall@'+str(eval_k)] = '-'
last_test_ep['AVG-F1@'+str(eval_k)] = '-'
last_test_ep['MRR'] = '-'
last_test_ep['MAP'] = '-'
last_test_ep['best_epoch'] = -1

sets = ['TRAIN', 'VALID', 'TEST']

for s in sets:
    errors[s] = {}
    losses[s] = {}
    MRRs[s] = {}
    MAPs[s] = {}
    prec[s] = {}
    rec[s] = {}
    f1[s] = {}
    prec_at_k[s] = {}
    rec_at_k[s] = {}
    f1_at_k[s] = {}
    prec_cl[s] = {}
    rec_cl[s] = {}
    f1_cl[s] = {}
    prec_at_k_cl[s] = {}
    rec_at_k_cl[s] = {}
    f1_at_k_cl[s] = {}

    best_measure[s] = 0
    best_epoch[s] = -1

str_comments=''
str_comments1=''

exp_params={}

print ("Start parsing: ",filename)
with open(filename) as f:
    params_line=True
    readlr=False
    for line in f:
        line=line.replace('INFO:root:','').replace('\n','')
        if params_line: #print parameters
            if "'learning_rate':" in line:
                   readlr=True
            if not readlr:
                str_comments+=line+'\n'
            else:
                str_comments1+=line+'\n'
            if params_line: #print parameters
                for p in params:
                    str_p='\''+p+'\': '
                    if str_p in line:
                        exp_params[p]=line.split(str_p)[1].split(',')[0]
            if line=='':
                params_line=False

        if 'TRAIN epoch' in line or 'VALID epoch' in line or 'TEST epoch' in line:
            set = line.split(' ')[1]
            epoch = int(line.split(' ')[3])+1
            if set=='TEST':
                last_test_ep['best_epoch'] = epoch
            if epoch==50000:
                break
        elif 'mean errors' in line:
            v=float(line.split('mean errors ')[1])#float(line.split('(')[1].split(')')[0])
            errors[set][epoch]=v
            if target_measure=='errors':
                if v<best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
        elif 'mean losses' in line:
            v = float(line.split('(')[1].split(')')[0].split(',')[0])
            losses[set][epoch]=v
            if target_measure=='loss':
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
        elif 'mean MRR' in line:
            v = float(line.split('mean MRR ')[1].split(' ')[0])
            MRRs[set][epoch]=v
            if set=='TEST':
                last_test_ep['MRR'] = v
            if target_measure=='mrr':
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
            if 'mean MAP' in line:
                v=float(line.split('mean MAP ')[1].split(' ')[0])
                MAPs[set][epoch]=v
                if target_measure=='map':
                    if v>best_measure[set]:
                        best_measure[set]=v
                        best_epoch[set]=epoch
                if set=='TEST':
                    last_test_ep['MAP'] = v
        elif 'measures microavg' in line:
            prec[set][epoch]=float(line.split('precision ')[1].split(' ')[0])
            rec[set][epoch]=float(line.split('recall ')[1].split(' ')[0])
            f1[set][epoch]=float(line.split('f1 ')[1].split(' ')[0])
            if (target_measure=='avg_p' or target_measure=='avg_r' or target_measure=='avg_f1'):
                if target_measure=='avg_p':
                    v=prec[set][epoch]
                elif target_measure=='avg_r':
                    v=rec[set][epoch]
                else: #F1
                    v=f1[set][epoch]
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
            if set=='TEST':
                last_test_ep['AVG-precision'] = prec[set][epoch]
                last_test_ep['AVG-recall'] = rec[set][epoch]
                last_test_ep['AVG-F1'] = f1[set][epoch]

        elif 'measures@'+str(eval_k)+' microavg' in line:
            prec_at_k[set][epoch]=float(line.split('precision ')[1].split(' ')[0])
            rec_at_k[set][epoch]=float(line.split('recall ')[1].split(' ')[0])
            f1_at_k[set][epoch]=float(line.split('f1 ')[1].split(' ')[0])
            if set=='TEST':
                last_test_ep['AVG-precision@'+str(eval_k)] =  prec_at_k[set][epoch]
                last_test_ep['AVG-recall@'+str(eval_k)] = rec_at_k[set][epoch]
                last_test_ep['AVG-F1@'+str(eval_k)] = f1_at_k[set][epoch]
        elif 'measures for class ' in line:
            cl=int(line.split('class ')[1].split(' ')[0])
            if cl not in prec_cl[set]:
                prec_cl[set][cl] = {}
                rec_cl[set][cl] = {}
                f1_cl[set][cl] = {}
            prec_cl[set][cl][epoch]=float(line.split('precision ')[1].split(' ')[0])
            rec_cl[set][cl][epoch]=float(line.split('recall ')[1].split(' ')[0])
            f1_cl[set][cl][epoch]=float(line.split('f1 ')[1].split(' ')[0])
            if (target_measure=='p' or target_measure=='r' or target_measure=='f1') and cl==cl_to_plot_id:
                if target_measure=='p':
                    v=prec_cl[set][cl][epoch]
                elif target_measure=='r':
                    v=rec_cl[set][cl][epoch]
                else: #F1
                    v=f1_cl[set][cl][epoch]
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
            if set=='TEST':
                last_test_ep['precision'] = prec_cl[set][cl][epoch]
                last_test_ep['recall'] = rec_cl[set][cl][epoch]
                last_test_ep['F1'] = f1_cl[set][cl][epoch]
        elif 'measures@'+str(eval_k)+' for class ' in line:
            cl=int(line.split('class ')[1].split(' ')[0])
            if cl not in prec_at_k_cl[set]:
                prec_at_k_cl[set][cl] = {}
                rec_at_k_cl[set][cl] = {}
                f1_at_k_cl[set][cl] = {}
            prec_at_k_cl[set][cl][epoch]=float(line.split('precision ')[1].split(' ')[0])
            rec_at_k_cl[set][cl][epoch]=float(line.split('recall ')[1].split(' ')[0])
            f1_at_k_cl[set][cl][epoch]=float(line.split('f1 ')[1].split(' ')[0])
            if (target_measure=='p@k' or target_measure=='r@k' or target_measure=='f1@k') and cl==cl_to_plot_id:
                if target_measure=='p@k':
                    v=prec_at_k_cl[set][cl][epoch]
                elif target_measure=='r@k':
                    v=rec_at_k_cl[set][cl][epoch]
                else:
                    v=f1_at_k_cl[set][cl][epoch]
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
            if set=='TEST':
                last_test_ep['precision@'+str(eval_k)] = prec_at_k_cl[set][cl][epoch]
                last_test_ep['recall@'+str(eval_k)] = rec_at_k_cl[set][cl][epoch]
                last_test_ep['F1@'+str(eval_k)] = f1_at_k_cl[set][cl][epoch]



if  best_epoch['TEST']<0 and  best_epoch['VALID']<0 or last_test_ep['best_epoch']<1:
    print ('best_epoch<0: -> skip')
    exit(0)

try:
    res_map['model'] = exp_params['model'].replace("'","")
    str_params=(pprint.pformat(exp_params))
    if print_params:
        print ('str_params:\n', str_params)
    if best_epoch['VALID']>=0:
        best_ep = best_epoch['VALID']
        print ('Highest %s values among all epochs: TRAIN  %0.4f\tVALID  %0.4f\tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['VALID'], best_measure['TEST']))
    else:
        best_ep = best_epoch['TEST']
        print ('Highest %s values among all epochs:\tTRAIN F1 %0.4f\tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['TEST']))

    use_latest_ep = True
    try:
        print ('Values at best Valid Epoch (%d) for target class: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, prec_cl['TEST'][cl_to_plot_id][best_ep],rec_cl['TEST'][cl_to_plot_id][best_ep],f1_cl['TEST'][cl_to_plot_id][best_ep]))
        print ('Values at best Valid Epoch (%d) micro-AVG: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, prec['TEST'][best_ep],rec['TEST'][best_ep],f1['TEST'][best_ep]))
        res_map['precision'] =  prec_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['recall'] = rec_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['F1'] = f1_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['AVG-precision'] = prec['TEST'][best_ep]
        res_map['AVG-recall'] = rec['TEST'][best_ep]
        res_map['AVG-F1'] = f1['TEST'][best_ep]
    except:
        res_map['precision'] =  last_test_ep['precision']
        res_map['recall'] = last_test_ep['recall']
        res_map['F1'] = last_test_ep['F1']
        res_map['AVG-precision'] = last_test_ep['AVG-precision']
        res_map['AVG-recall'] = last_test_ep['AVG-F1']
        res_map['AVG-F1'] = last_test_ep['AVG-F1']
        use_latest_ep = False
        print ('WARNING: last epoch not finished, use the previous one.')

    try:
        print ('Values at best Valid Epoch (%d) for target class@%d: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, eval_k, prec_at_k_cl['TEST'][cl_to_plot_id][best_ep],rec_at_k_cl['TEST'][cl_to_plot_id][best_ep],f1_at_k_cl['TEST'][cl_to_plot_id][best_ep]))
        res_map['precision@'+str(eval_k)] =  prec_at_k_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['recall@'+str(eval_k)] = rec_at_k_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['F1@'+str(eval_k)] = f1_at_k_cl['TEST'][cl_to_plot_id][best_ep]

        print ('Values at best Valid Epoch (%d) micro-AVG@%d: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, eval_k, prec_at_k['TEST'][best_ep],rec_at_k['TEST'][best_ep],f1_at_k['TEST'][best_ep]))
        res_map['AVG-precision@'+str(eval_k)] =  prec_at_k['TEST'][best_ep]
        res_map['AVG-recall@'+str(eval_k)] = rec_at_k['TEST'][best_ep]
        res_map['AVG-F1@'+str(eval_k)] = f1_at_k['TEST'][best_ep]

    except:
        res_map['precision@'+str(eval_k)] =  last_test_ep['precision@'+str(eval_k)]
        res_map['recall@'+str(eval_k)] = last_test_ep['recall@'+str(eval_k)]
        res_map['F1@'+str(eval_k)] = last_test_ep['F1@'+str(eval_k)]
        res_map['AVG-precision@'+str(eval_k)] = last_test_ep['AVG-precision@'+str(eval_k)]
        res_map['AVG-recall@'+str(eval_k)] = last_test_ep['AVG-recall@'+str(eval_k)]
        res_map['AVG-F1@'+str(eval_k)] = last_test_ep['AVG-F1@'+str(eval_k)]

    try:
        print ('Values at best Valid Epoch (%d) MAP: TRAIN  %0.8f - VALID %0.8f - TEST %0.8f' % (best_ep,  MAPs['TRAIN'][best_ep],  MAPs['VALID'][best_ep],  MAPs['TEST'][best_ep]))
        res_map['MAP'] = MAPs['TEST'][best_ep]
    except:
        res_map['MAP'] = last_test_ep['MAP']
    try:
        print ('Values at best Valid Epoch (%d) MRR: TRAIN  %0.8f - VALID %0.8f - TEST %0.8f' % (best_ep,  MRRs['TRAIN'][best_ep],  MRRs['VALID'][best_ep],  MRRs['TEST'][best_ep]))
        res_map['MRR'] = MRRs['TEST'][best_ep]
    except:
        res_map['MRR'] = last_test_ep['MRR']

    if use_latest_ep:
        res_map['best_epoch'] = best_ep
    else:
        res_map['best_epoch'] = last_test_ep['best_epoch']

except:
    print('Some error occurred in', filename,' - Epochs read: ',epoch)
    exit(0)

str_results = ''
str_legend = ''
for k, v in res_map.items():
    str_results+=str(v)+','
    str_legend+=str(k)+','
for k, v in exp_params.items():
    str_results+=str(v)+','
    str_legend+=str(k)+','
str_results+=filename.split('/')[1].split('.log')[0]
str_legend+='log_file'
print ('\n\nCSV-like output:')
print (str_legend)
print (str_results)
