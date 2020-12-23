import cv2
import numpy as np
import face_recognition
import os
import pickle
path="static"
myimages=[]
newlist=[]

newclassname=[]
mylist=os.listdir(path)

with open('classname.dat', 'rb') as f:
    oldclassname = pickle.load(f)
print("updating classnames.....")
for fn in mylist:
    cl=fn.split(".")[0]
    if cl not in oldclassname:
        newlist.append(fn)
print("getting new cv_images.......")
for cl in newlist:
    curimg=cv2.imread(f"{path}/{cl}")
    myimages.append(curimg)
    newclassname.append(cl.split(".")[0])

def findencodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist
print("Finding encodings.......")
newknown=findencodings(myimages)
print("encoding done.......")
print("updating datas.......")
with open('dataset_faces.dat', 'rb') as f:
    oldknown = pickle.load(f)
for en in newknown:
    oldknown.append(en)
for ncl in newclassname:
    oldclassname.append(ncl)

print("uploaing updates.......")
with open('classname.dat', 'wb') as f:
    pickle.dump(oldclassname, f)
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(oldknown, f)
print("Successfully data updated.......")
print(len(oldclassname), len(oldknown))
print("newly added faces : ",len(newclassname),newclassname)