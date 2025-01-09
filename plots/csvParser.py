import csv
import matplotlib.pyplot as plt 

steps = []
map = []
stepsBad = []
mapBad = []
stepsGood = []
mapGood = []

with open('/home/kornwolfd/ML_CV_practicalCourse/plots/plotData/goodBase.csv') as file:  
    lines = csv.reader(file, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        steps.append(int(row[1])) 
        map.append(float(row[2]))

with open('/home/kornwolfd/ML_CV_practicalCourse/plots/plotData/badAugmentation.csv') as f:  
    lines = csv.reader(f, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        stepsBad.append(int(row[1])) 
        mapBad.append(float(row[2]))

with open('/home/kornwolfd/ML_CV_practicalCourse/plots/plotData/lightning_logs_goodAug.csv') as f:  
    lines = csv.reader(f, delimiter=',') 
    for row in lines: 
        if 'Step' in row:
            continue
        stepsGood.append(int(row[1])) 
        mapGood.append(float(row[2]))



plt.figure(figsize=(12.8, 4.8))
#plt.plot(stepsGood, mapGood, color = 'b', linestyle = 'solid', label = "with good augmentation, map@50 = 62,11%")
plt.plot(steps, map, color = 'g', linestyle = 'solid', label = "without augmentation, map@50 = 54,42%") 
plt.plot(stepsBad, mapBad, color = 'r', linestyle = 'solid', label = "with bad augmentation, map@50 = 41,5%") 

  
plt.minorticks_off()
#plt.xlim(0.0, 32899.0)
plt.xlim(0)
plt.xscale('linear')
#plt.ylim(0.0, 0.6)
plt.ylim(0)
plt.yscale('linear')
plt.xlabel('Step') 
plt.ylabel('mAP@50') 
#plt.title('Weather Report', fontsize = 20) 
plt.legend() 

plt.savefig('/home/kornwolfd/ML_CV_practicalCourse/plots/badAug.png')