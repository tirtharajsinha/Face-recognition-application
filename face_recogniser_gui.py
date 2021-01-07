############## LIBRARY IMPORT #####################

from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import cv2
import numpy as np
import face_recognition
#import os
from datetime import datetime
import pickle



############## Fetching encodings and classnames ###########

with open('classname.dat', 'rb') as f:
    classnames = pickle.load(f)

print("Getting encodings....")
with open('dataset_faces.dat', 'rb') as f:
    encodeknown = pickle.load(f)




############### marking ##################
def markings(name):
    with open("found_faces.csv","r+") as f:
        mydatalist=f.readlines()

        per=0
        now = datetime.now()
        timenow = str(now).split()
        date = timenow[0]
        time = timenow[1].split(".")[0]
        for line in mydatalist:
            entry=line.split(",")
            if entry[0]==name and entry[1]==date:
                per+=1
        if per==0:
            mandata=f"\n{name},{date},{time}"
            f.writelines(mandata)
            print("data recorded : ",mandata)

############# SUPPORTING FUNCTIONS ############
cnd=True
def cmnd():
    m.destroy()
def ext():
    global cnd
    cnd=False
    m.destroy()
def cononext():
    global cnd

    MsgBox = messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application',
                                       icon='warning')
    if MsgBox == 'yes':


        cnd=False


def changeOnHover(button, colorOnHover, colorOnLeave):
    # adjusting backgroung of the widget
    # background on entering widget
    button.bind("<Enter>", func=lambda e: button.config(
        background=colorOnHover))

    # background color on leving widget
    button.bind("<Leave>", func=lambda e: button.config(
        background=colorOnLeave))

def train_model():
    update = messagebox.askquestion('starting model update',
                                 'Do you want to update the model? ',
                                 icon='question')
    if update == 'yes':
        global classnames
        global encodeknown
        exec(open("model_trainer.py").read())
        exec(open("update_model.py").read())
        with open('classname.dat', 'rb') as f:
            classnames = pickle.load(f)

        print("Getting encodings....")
        with open('dataset_faces.dat', 'rb') as f:
            encodeknown = pickle.load(f)
        messagebox.showinfo("application update detected", "newly trained face_Recognition application is ready to use"
                                                           "Enjoy!!!")
    else:
        messagebox.showinfo("update Cancelled", "cancelled model training")

########### Application starter confermation window ##########

m=Tk()
m.title("LookAtMe - Face Recogniser")
m.geometry("250x100")
m.configure(bg="goldenrod")
m.minsize(height=100,width=250)
m.maxsize(height=100,width=250)
f=Frame(m,bg="cyan",height=150,width=200)
f.pack()
Label(f,bg="cyan",fg="goldenrod",text="Welcome to LookAtMe").pack()
Label(f,bg="cyan",fg="red",text="Do you want to launch the application ?").pack()
Button(m,fg="red", text="yes", command=cmnd,width=10).place(x=40,y=50)
Button(m,fg="red", text="no", command=ext,width=10).place(x=150,y=50)

m.mainloop()


######################### MAIN GUI WINDOW ##########################

root = Tk()
icn = PhotoImage(file = "faceicon.jpg")
root.iconphoto(False, icn)
root.title("LookAtMe - Face recogniser ")
root.geometry("1000x800")
root.minsize(width=1000, height=800)
root.maxsize(width=1000, height=800)
root.configure(bg="goldenrod2")
upbtn=Button(root,text="Update model",bg="green",fg="white", command=train_model, font="Consolas 20 bold")
upbtn.place(x=680,y=160,height=50,width=180)

extbtn=Button(root,text="close",bg="white",fg="red", command=cononext, font="Consolas 20 bold")
extbtn.place(x=890,y=160,height=50,width=80)

time = Label(root,bg="blue",fg="white", font="Consolas 20 bold")
time.place(x=720,y=220,height=50,width=200)
changeOnHover(extbtn,"green","white")
ft = Frame(root, bg="green", height=200)
ft.pack(fill="x", padx=10, pady=10)
Label(ft, bg="green", font="Algerian 20 bold underline", text="LookAtMe : THE FACE RECOGNISER", fg="red").pack()
des = "LookAtMe is a personal project , which can recognise face in less than seconds. \nBuilt using python3.8 with the help of opencv and face-recognition" \
          "\nlibrary. You have to train model with a person's image.filename should be that parson's name."
Label(ft, bg="green", font="Arial 10 bold", text=des, fg="yellow").pack(padx=30)
Label(ft, bg="green", font="Cooper 12 bold", text="By Tirtharaj Sinha", fg="white").pack(anchor="se", padx=20)
f3 = Frame(root, bg="white", height=100, width=640)
f3.pack(anchor="nw", padx=10, pady=10)
l2 = Label(f3, bg="white", font="Helvetica 20 bold", fg="goldenrod")
l2.place(x=170, y=20)
f1 = LabelFrame(root, bg="blue")
f1.pack(side=LEFT, padx=10, pady=10)
l1 = Label(f1, bg="red")
l1.pack(side=RIGHT)
f2 = Frame(root, bg="green", height=660, width=280)
f2.pack(side=RIGHT, padx=10, pady=10)
canvas = Canvas(f2, bg="orange", width=280, height=560, scrollregion=(0, 0, 280, 4000))
vbar = Scrollbar(f2, orient=VERTICAL)
vbar.pack(side=RIGHT, fill=Y)
vbar.config(command=canvas.yview)
canvas.config(yscrollcommand=vbar.set)
canvas.pack(anchor="ne")
canvas.create_text(150, 20, text="Today's Attendees", font="Jokerman 15 bold underline", fill="blue")

############################# MAIN application logic ##############################

cap = cv2.VideoCapture(0)
while True:
    if cnd==False:
        root.destroy()
        break


    sucess, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgs)
    encodescurframe = face_recognition.face_encodings(imgs, facescurframe)
    for encodeface, faceloc in zip(encodescurframe, facescurframe):
        matches = face_recognition.compare_faces(encodeknown, encodeface)
        facedis = face_recognition.face_distance(encodeknown, encodeface)

        matchindex = np.argmin(facedis)

        if facedis[matchindex] > 0.7:
            print("no match found")
            continue
        if matches[matchindex]:

            name = classnames[matchindex]

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            mandata = markings(name)

            try:
                l2["text"] = "------Recognised face------\n" + name
            except:
                break

    try:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img1))
        l1["image"] = img
    except:
        break
    now = datetime.now()
    timenow = str(now).split()
    t = timenow[1].split(".")[0]
    time["text"]=t
    with open("found_faces.csv", "r+") as f:
        mydatalist = f.readlines()
        ################# DATA RENDERING at GUI #################
        now = datetime.now()
        timenow = str(now).split()
        date = timenow[0]
        count = 0
        hight = 100
        for line in mydatalist:
            entry = line.split(",")
            if entry[1] == date:
                count += 1
                curdata = str(count) + " : " + line
                canvas.create_text(0, hight, anchor="nw", text=curdata, font="arial 10 bold underline")
                hight += 30

    root.update()



