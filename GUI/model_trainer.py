from PIL import Image
from tkinter.filedialog import askopenfile
import os
import pickle
from tkinter import messagebox,simpledialog



file=askopenfile()
print(file)
perm=True
if str(file) !="None":
    filepath = str(file.name)
    print(filepath)
    im = Image.open(filepath)

    filename = simpledialog.askstring("classname", "What is the classname of the image ?")

else:
    perm=False
    filename=""


with open('classname.dat', 'rb') as f:
    classnames = pickle.load(f)

if perm==False:
    messagebox.showinfo("worning", "no image got")
elif filename.strip()=="":
    print("no classname found !!!")
    messagebox.showinfo("worning", "no classname found")
elif filename.strip() in classnames:
    print("our model is already trained with this data")
    messagebox.showinfo("info", "our model is already trained with this data")

else:
    filename+=".jpg"

    msg=messagebox.askquestion('saving image', 'Are you sure you want to save the image with the name of '+filename,
                                       icon='question')
    if msg == 'yes':
        filename = "staticnew/" + filename
        im.save(filename)
    else:
        messagebox.showinfo("Information", "cancelled training process")




