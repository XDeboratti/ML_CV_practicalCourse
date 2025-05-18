import csv
import json
import numpy as np
import matplotlib.pyplot as plt 

faster_rcnn_noaug_map = []
faster_rcnn_aug_map = []
ssd_noaug_map = []
ssd_aug_map = []


with open('') as file:  
    lines = csv.reader(file, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        ssd_aug_map.append(float(row[2]))

with open('') as f:  
    lines = csv.reader(f, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        ssd_noaug_map.append(float(row[2]))

with open('') as f:
    result_file = json.load(f)
    for epoch in result_file.values():
        faster_rcnn_aug_map.append(float(epoch['validation_map']))

with open('') as f:
    result_file = json.load(f)
    for epoch in result_file.values():
        faster_rcnn_noaug_map.append(float(epoch['validation_map']))





plt.figure(figsize=(12.8, 4.8))
plt.plot(np.arange(50), faster_rcnn_aug_map, linestyle = 'solid', label = "Faster R-CNN with augmentation, map@50 = " + str(round(max(faster_rcnn_aug_map),2)) +"%") 
plt.plot(np.arange(50), faster_rcnn_noaug_map, linestyle = 'solid', label = "Faster R-CNN without augmentation, map@50 = " + str(round(max(faster_rcnn_noaug_map),2)) +"%") 
plt.plot(np.arange(50), ssd_aug_map[:50], linestyle = 'solid', label = "ssd with augmentation, map@50 = " + str(round(max(ssd_aug_map[:50]),2)) +"%") 
plt.plot(np.arange(50), ssd_noaug_map[:50], linestyle = 'solid', label = "ssd without augmentation, map@50 = " + str(round(max(ssd_noaug_map[:50]),2)) +"%") 

  
plt.minorticks_off()
plt.xlim(0)
plt.xscale('linear')
plt.ylim(0)
plt.yscale('linear')
plt.xlabel('Epoch') 
plt.ylabel('mAP@50') 
plt.legend() 

plt.savefig('')