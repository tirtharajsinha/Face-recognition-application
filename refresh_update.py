import cv2
import numpy as np
import face_recognition
import os
import pickle
path="static"
images=[]
classname=[]

mylist=os.listdir(path)

for cl in mylist:
    curimg=cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    classname.append(cl.split(".")[0])




with open('classname.dat', 'wb') as f:
    pickle.dump(classname, f)
print("classname updated........")

def findencodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(encodelist, f)
    return encodelist
print("updating encodings........")
known=findencodings(images)
if len(images)==len(known) and len(images)==len(classname):
    print("encoding sucessesfully done.")
else:
    print("run update again.....")