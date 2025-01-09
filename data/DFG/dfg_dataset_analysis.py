import json

data_dir = '/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/'
phase = 'test'

#boxes are provided in coco format bbx[x, y, w, h] where x & y are the coordinates of the upper left corner
#we need the boxes in bbx[x1, y1, x2, y2] where 1 is the upper left corner and y is the lower right corner of the box
def transform_bbox(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

with open(data_dir+'DFG-tsd-annot-json/'+phase+'.json') as f:
            labels_file = json.load(f)
            images = labels_file['images']
            print('length: ' + str(len(images)))

#the json is split into 
# - images[id, height, width, file_name]
# - categories[id, name, supercategory] 
# - annotations[id, area, bbox, category_id, segmentation, image_id, ignore, iscrowd]
#we need the labels in the following form: [List[Dict{str, Tensor}]]
# - where the list index corresponds to the index of the described image in the image list
# - the dictionary has 'boxes' and 'class'
# - 'boxes' is the key for a tensor containing the boxes
# - 'class' is the key for a tensor containing the classes for the boxes 
# annotations = labels_file['annotations']
# labels = {}
# classes = {}
# for annotation in annotations:
#     if classes.get(annotation['category_id'], None) is None:
#           classes[annotation['category_id']] = 1
#     else:
#           classes[annotation['category_id']] += 1

# sortedClasses = dict(sorted(classes.items(), key = lambda item: item[0]))
# print(sortedClasses) 

# rightOfWayClasses = []
# for x in range(38,82):
#       rightOfWayClasses.append(x)
# for x in range(161,171):
#       rightOfWayClasses.append(x)

# intermedClassMapping = {}
# for c in sortedClasses:
#       if c in rightOfWayClasses:
#             intermedClassMapping[c] = c
#       elif sortedClasses[c] <= 20:
#             intermedClassMapping[c] = 300
#       else:
#             intermedClassMapping[c] = c
# print(intermedClassMapping)

# newSet = set(intermedClassMapping.values())
# numDifferentClasses = len(newSet)
# print(numDifferentClasses)

# classMapping = {}
# counter = 0
# for c in intermedClassMapping:
#       if intermedClassMapping[c] != 300:
#             classMapping[c] = counter
#             counter += 1
#       elif intermedClassMapping[c] == 300 and counter == 11:
#             classMapping[c] = 11
#             counter += 1
#       else:
#             classMapping[c] = 11
# print(classMapping)
    
