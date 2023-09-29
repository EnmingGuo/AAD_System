import re
from tkinter import font
import json
import Register
import ttkbootstrap as tk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
import tkinter
import tkinter.messagebox as messagebox
import Database
import step1
import AidedDiagnosis
from pathlib import Path
from tkinter import PhotoImage
PATH = Path(__file__).parent / 'assets'
class DataEntryForm(tk.Frame):
    def __init__(self, master,database=None,data=None):
        master.title('Login')
        master.geometry('1000x650')
        super().__init__(master,padding=15)
        self.grid(row=0, column=0, sticky='nsew')
        self.images = [
            PhotoImage(
                name='brain',
                file=PATH / 'magic_mouse/brain_title.png'
            ),
            PhotoImage(
                name='syringe',
                file=PATH / 'magic_mouse/sharpicons_syringe.png'
            ),
            PhotoImage(
                name='doctornotes',
                file=PATH / 'magic_mouse/sharpicons_doctor-notes.png'
            ),
            PhotoImage(
                name='Login',
                file=PATH/'magic_mouse/Login_Interface.png'
            )
        ]
        self.database = database
        if(data!=None):
            username_rtn = data.get('username')
            username_str_var = tk.StringVar(value = username_rtn)
        else:
            username_str_var = tk.StringVar()
        password_str_var = tk.StringVar()

        tk.Label(self, image='Login').grid(row=1,column=0,columnspan=2,sticky=tk.W,pady=50)
        # 账户信息
        tk.Label(self, text='Username: ',width =20).grid(row=2, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=username_str_var,width=30).grid(row=2, column=1, sticky=tk.W)
        tk.Label(self, text='Password: ',width =20).grid(row=3, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=password_str_var,width=30,show='*').grid(row=3, column=1, sticky=tk.W)

        def hit_login_button():
            login_button.configure(state='disable')
            if(check_null()!=True):
                login_button.configure(state='normal')
                return
            if(submit_data()==True):
                login_button.configure(state='normal')
                result = messagebox.showinfo(title='Successful Login', message='You Login successfully!')
                if result == 'ok':
                    data={
                        'username': username_str_var.get(),
                        'database': self.database,
                        'lastpage':"Login"
                    }
                    if(Database.get_authority(self.database,username_str_var.get())=="admin"):
                        self.destroy()
                        step1.open_train_file(master,data = data)
                    else:
                        self.destroy()
                        AidedDiagnosis.AidedDiagnosis(master, data=data)
                return
            else:
                login_button.configure(state='normal')
                return

        def hit_register_button():
            self.destroy();
            Register.DataEntryForm(master,database = self.database)

        # 提交按钮
        login_button = tk.Button(self, text='Login', width=20)
        login_button.grid(row=4, column=0,sticky=tk.W,pady=50)
        login_button.configure(command=hit_login_button)
        register_button = tk.Button(self, text='Register', width=20)
        register_button.grid(row=4, column=1, sticky=tk.W, padx=65,pady=50)
        register_button.configure(command=hit_register_button)

        self.place(relx=0.55, rely=0.5, anchor=CENTER)


        def submit_data():
            login_button.configure(state="disable")
            data = {
                "username": username_str_var.get(),
                "password": password_str_var.get()
            }
            ans = Database.check_Login(self.database,data)
            if(ans==0):
                messagebox.showwarning(title='Login error', message="Username doesn't exist.")
                return False
            elif(ans == 2):
                messagebox.showwarning(title='Login error', message="Password is wrong.")
                return False;
            else :
                return True;

        def check_null():
            password = password_str_var.get()
            username = username_str_var.get()
            if(password == "" or username == "" ):
                messagebox.showwarning(title='Information Incomplete',message ='Please complete the information')
                return False
            else:
                return True

if __name__ == "__main__":

    Alzheimer = Database.init()
    app = tk.Window(resizable=(False,False),themename= "yeti")
    DataEntryForm(app,database=Alzheimer)
    app.mainloop()









