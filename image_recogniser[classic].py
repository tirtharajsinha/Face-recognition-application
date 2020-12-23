import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter.filedialog import askopenfile
import pickle
# getting all known data





# with open('cv_images.dat', 'rb') as f:
#     images = pickle.load(f)
with open('classname.dat', 'rb') as f:
    classnames = pickle.load(f)

print(classnames)
# def findencodings(images):
#     encodelist=[]
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodelist.append(encode)
#     with open('dataset_faces.dat', 'wb') as f:
#         pickle.dump(encodelist, f)
#     return encodelist


#main classifier

def classifier(img,encodeknown):
    img = cv2.resize(img, (544, 640),
                     interpolation=cv2.INTER_NEAREST)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgs)
    encodescurframe = face_recognition.face_encodings(imgs, facescurframe)
    for encodeface, faceloc in zip(encodescurframe, facescurframe):
        matches = face_recognition.compare_faces(encodeknown, encodeface)
        facedis = face_recognition.face_distance(encodeknown, encodeface)

        matchindex = np.argmin(facedis)
        if facedis[matchindex]>.7:
            print("no match found")
            break
        if matches[matchindex]:
            name = classnames[matchindex]
            print("result : ",name, min(facedis) * 100)
            y1, x2, y2, x1 = faceloc

            cv2.rectangle(img, (x1, y1), (x2, y2 + 20), (12, 240, 202), 2)
            cv2.rectangle(img, (x1, y2 - 15), (x2, y2 + 20), (12, 240, 202), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 + 20 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 58, 12), 2)

    cv2.imshow("face recogniser", img)
    cv2.waitKey(0)
# application engine for entire program
def engine(img):
    print("Getting encodings....")
    with open('dataset_faces.dat', 'rb') as f:
        encodeknown = pickle.load(f)


    classifier(img,encodeknown)

# getting query data
file = askopenfile()
permission=True
# checking compatebility
try:
    filename = str(file.name)
    img=cv2.imread(filename)

except:
    permission=False

# checking permission
if permission==True:
    engine(img)
else:
    print("no image uploaded")



