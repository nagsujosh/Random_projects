from tkinter import *
from PIL import ImageTk, Image

root = Tk()


my_img1 = ImageTk.PhotoImage(Image.open("icon.png"))
my_img2 = ImageTk.PhotoImage(Image.open("icon.png"))
my_img3 = ImageTk.PhotoImage(Image.open("icon.png"))
my_img4 = ImageTk.PhotoImage(Image.open("icon.png"))
my_img5 = ImageTk.PhotoImage(Image.open("icon.png"))

image_list = [my_img1, my_img2, my_img3, my_img4, my_img5]

my_status = Label(root, text="Image 1 of " + str(len(image_list)), bd=1, relief=SUNKEN, anchor=E)

my_label = Label(image=my_img1)
my_label.grid(row=0, column=0, columnspan=3)


def forward(image_number):
    global my_label
    global button_forward
    global button_back
    global my_status

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number + 1))
    button_back = Button(root, text="<<", command=lambda: back(image_number - 1))

    if image_number == 5:
        button_forward = Button(root, text=">>", state=DISABLED)

    my_status = Label(root, text="Image "+str(image_number)+"of " + str(len(image_list)), bd=1, relief=SUNKEN, anchor=E)
    my_status.grid(row=2, column=0, columnspan=3, sticky=W + E)

    button_forward.grid(row=1, column=2)
    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)


def back(image_number):
    global my_label
    global button_forward
    global button_back
    global my_status

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number + 1))
    button_back = Button(root, text="<<", command=lambda: back(image_number - 1))

    if image_number == 1:
        button_back = Button(root, text="<<", state=DISABLED)

    my_status = Label(root, text="Image " + str(image_number) + "of " + str(len(image_list)), bd=1, relief=SUNKEN,
                      anchor=E)
    my_status.grid(row=2, column=0, columnspan=3, sticky=W + E)

    button_forward.grid(row=1, column=2)
    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)


button_back = Button(root, text="<<", command=lambda: back, state=DISABLED)
button_exit = Button(text="Exit", command=root.quit)
button_forward = Button(root, text=">>", command=lambda: forward(2))

button_back.grid(row=1, column=0)
button_exit.grid(row=1, column=1)
button_forward.grid(row=1, column=2, pady=10)
my_status.grid(row=2, column=0, columnspan=3, sticky=W+E)

root.mainloop()
