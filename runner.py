import subprocess
import matplotlib.pyplot as plt

default_learning_rate = .1
default_dropout = .4
default_momentum = .4
default_weight_decay = .01

def run(prog,args,result_vector,label):
    print(label)
    cmd = ['python3', prog]
    if(len(args)>0):
        cmd.extend([str(a) for a in args])
    run = subprocess.run(cmd, stdout=subprocess.PIPE)
    results = str(run.stdout)
    acc = float(results[results.find('epochs:')+8:-3])
    result_vector.append(acc)
    print('Final testing accuracy: {:.2f}'.format(acc))

fullres = {}

progs = ['q1.py','q2.py','q3.py','q4.py']
fig_num = 0
for prog in progs:
    if(prog == 'q1.py' or prog == 'q2.py'):
        if(prog=='q1.py'):
            name = 'q1'
            iter_title = 'Running Q1: Sigmoid activation'
        else:
            name = 'q2'
            iter_title = 'Running Q2: Relu activation'


        lr_vals = [.0001,.001,.01,.1,1.,10.]
        accs = []
        fullres['{}_lr'.format(name)] = []
        for lr in lr_vals:
            run(prog,[lr],accs,'{}: LR={}'.format(iter_title,lr))
            fullres['{}_lr'.format(name)].append((lr,accs[-1]))
        
        plt.figure(fig_num)
        fig_num+=1
        plt.plot(lr_vals,accs)
        plt.xscale('log')
        plt.title('Testing Accuracy vs Learning Rate')
        plt.savefig('images/{}_acc_over_lr.png'.format(name))

    elif(prog == 'q3.py'):
        d_vals = [.1,.2,.3,.4,.5,.7,.9]
        m_vals = [.1,.2,.3,.4,.5,.7,.9]
        wd_vals = [.01,.02,.03,.04,.05,.07,.09]

        d_accs,m_accs,wd_accs = [],[],[]

        #Dropout wth fixed momentum and WD
        fullres['q3_d'] = []
        for d in d_vals:
            run(prog,[d,default_momentum,default_weight_decay],d_accs,'Running Q3: Dropout={}, fix Momentum, WD'.format(d))
            fullres['q3_d'].append((d,d_accs[-1]))
        fullres['q3_m'] = []
        for m in m_vals:
            run(prog,[default_dropout,m,default_weight_decay],m_accs,'Running Q3: Momentum={}, fix Dropout, WD'.format(m))
            fullres['q3_m'].append((m,m_accs[-1]))
        fullres['q3_wd'] = []
        for wd in wd_vals:
            run(prog,[default_dropout,default_momentum,wd],wd_accs,'Running Q3: Weight Decay={}, fix Dropout, Momentum'.format(wd))
            fullres['q3_wd'].append((wd,wd_accs[-1]))
        
        plt.figure(fig_num)
        fig_num+=1
        plt.plot(d_vals,d_accs)
        plt.title('Testing Accuracy vs Dropout, fix momentum, weight decay')
        plt.savefig('images/q3_acc_over_d.png')
        plt.figure(fig_num)
        fig_num+=1
        plt.plot(m_vals,m_accs)
        plt.title('Testing Accuracy vs Momentum, fix dropout, weight decay')
        plt.savefig('images/q3_acc_over_m.png')
        plt.figure(fig_num)
        fig_num+=1
        plt.plot(wd_vals,wd_accs)
        plt.title('Testing Accuracy vs Weight decay, fix dropout, momentum')
        plt.savefig('images/q3_acc_over_wd.png')
    
    elif(prog == 'q4.py'):
        acc = []
        run(prog,[],acc,'Running Q4')
        fullres['q4'] = [(2,acc[-1])]

import os
os.makedirs('results/',exist_ok=True)
for key in fullres:
    print('{}: {}'.format(key,fullres[key]))
    with open('results/'+key+'.csv','w') as f:
        for (param,acc) in fullres[key]:
            f.write('{},{}\n'.format(param,acc))