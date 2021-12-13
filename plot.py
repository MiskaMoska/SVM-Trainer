from matplotlib import pyplot as plt

log_num = 5
x,y = [],[]
for i in range(log_num):
    x.append([])
    y.append([])

for i in range(log_num):
    log_file = './log/log_'+str(i+1)+'.txt'
    with open(log_file,'r') as lf:
        for j,line in enumerate(lf,1):
            x[i].append(j)
            y[i].append(float(line))

ovl = 0
for i in range(log_num):
    plt.plot(x[i],y[i],'-',label='class overlap '+str(ovl)+'%',
                linewidth = 3)
    ovl += 15

plt.xlabel('number of iterations',fontsize='14')
plt.ylabel('LP optimal value',fontsize='14')
plt.xticks(fontsize='12')
plt.yticks(fontsize='12')
plt.legend(fontsize=12)
plt.show()