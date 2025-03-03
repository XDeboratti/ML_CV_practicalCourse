import csv
import json
import numpy as np
import matplotlib.pyplot as plt 

ssd_map = []


with open('/home/kornwolfd/ML_CV_practicalCourse/plots/plotData/map_lightning_logs_mapillary_ssd_blur_noise_scale_tans.csv') as f:  
    lines = csv.reader(f, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        ssd_map.append(float(row[2]))

plt.figure(figsize=(6.4, 4.8)) 
plt.plot(np.arange(50), ssd_map[:50], linestyle = 'solid', label = "ssd with augmentation, map@50 = " + str(round(max(ssd_map[:50]),2)) +"%") 
#plt.plot(np.arange(50), ssd_noaug_map[:50], linestyle = 'solid', label = "ssd without augmentation, map@50 = " + str(round(max(ssd_noaug_map[:50]),2)) +"%") 

  
plt.minorticks_off()
plt.xlim(0)
plt.xscale('linear')
plt.ylim(0)
plt.yscale('linear')
plt.xlabel('Epoch') 
plt.ylabel('mAP@50') 
#plt.legend() 


plt.savefig('/home/kornwolfd/ML_CV_practicalCourse/plots/overfitting_Aug_map.png')