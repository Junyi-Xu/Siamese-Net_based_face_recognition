from tkinter import *
from PIL import ImageTk, Image
import argparse,os
import main

model_file = "inference/face_verification/__model__"
params_file = "inference/face_verification/__params__"
image_path = "data/original_images/seats.png"   # 更改为PS完的座位照的路径
source_dirc = "data/source"
out_dirc = "data/out"


def drawGrid(canvas,sx,sy,w,h,row,col,dx,dy):
    canvas.create_rectangle(sx,sy,sx+w,sy+h,fill="white")
    for i in range(1,row):
        canvas.create_line(sx,sy+i*dy,sx+w,sy+i*dy)
    for j in range(1,col):
        canvas.create_line(sx+j*dx,sy,sx+j*dx,sy+h)

def toCoordinate(sx,sy,dx,dy,row,col):
    return sx+col*dx,sy+row*dy

def runDrawing(width=300, height=300):

    matchDict, seats = main.doRecognize(model_file,params_file,image_path,source_dirc,out_dirc,False)
    # print(matchDict)
    # print(seats)

    root = Tk()
    canvas = Canvas(root, width=width, height=height)
    canvas.pack()

    xMargin = yMargin = 20
    gridWidth = width - 2 * xMargin
    gridHeight = height - 3 * yMargin
    numRow = 4
    numColumn = 7
    dx = gridWidth // numColumn
    dy = gridHeight // numRow
    drawGrid(canvas,xMargin,yMargin,gridWidth,gridHeight,numRow,numColumn,dx,dy)
    deskWidth = yMargin*3
    deskHeight = yMargin
    canvas.create_rectangle(width//2-deskWidth//2,height-2*yMargin,width//2+deskWidth//2,height-yMargin,fill="grey")

    canvas.images = []
    for person in matchDict:
        img = Image.open(matchDict[person])
        img = img.resize((dx,dx))
        img = ImageTk.PhotoImage(img)
        row,col = seats[person]
        x,y = toCoordinate(xMargin,yMargin,dx,dy,row,col)
        name = matchDict[person].split('/')[-1].split('.')[0]
        canvas.create_image(x,y,anchor=NW,image=img)
        canvas.images.append(img)
        canvas.create_text(x+dx//2, y+dy, text=name, anchor=SW,fill="darkBlue", font="Times 14 bold italic")

    root.mainloop()
    print("bye!")

runDrawing(700, 500)
