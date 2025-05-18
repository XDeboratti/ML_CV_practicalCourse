import csv
import json
import numpy as np
import matplotlib.pyplot as plt 

ssd_loss_noaug = []
ssd_loss = []


with open('') as file:  
    lines = csv.reader(file, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        ssd_loss.append(float(row[2]))
ssd_loss_every3rd = ssd_loss[:185:3]
ssd_loss_final = ssd_loss_every3rd[:50]

with open('') as file:  
    lines = csv.reader(file, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        ssd_loss_noaug.append(float(row[2]))
ssd_loss_noaug_every3rd = ssd_loss_noaug[:185:3]
ssd_loss_noauug_final = ssd_loss_noaug_every3rd[:50]


plt.figure(figsize=(6.4, 4.8)) 
plt.plot(np.arange(50), ssd_loss_final[:50], linestyle = 'solid', label = "ssd with augmentation") 
plt.plot(np.arange(50), ssd_loss_noauug_final[:50], linestyle = 'solid', label = "ssd without augmentation") 

  
plt.minorticks_off()
plt.xlim(0)
plt.xscale('linear')
plt.ylim(0)
plt.yscale('linear')
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.legend() 


plt.savefig('')