from PIL import Image
from tkinter.filedialog import askopenfile
import os
file=askopenfile()
filepath=str(file.name)
print(filepath)
im=Image.open(filepath)
filename=input("enter the subject of the image : ")
path="static"

classnames=[]
mylist=os.listdir(path)

for cl in mylist:
    classnames.append(cl.lower().split(".")[0])

if filename.lower() in classnames:
    print("our model is already trained with this data")
else:
    filename+=".jpg"
    filename="static/"+filename
    print("saving with ", filename)
    im.save(filename)


