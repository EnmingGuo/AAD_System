import re
from tkinter import font
import json
import ttkbootstrap as tk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import DateEntry
from ttkbootstrap import utility
import tkinter
import tkinter.messagebox as messagebox
import Database
import Login
import AidedDiagnosis
import user_record
import datetime
class upgrade(tk.Frame):
    def __init__(self, master,data = None):
        master.title('Upgrade')
        master.geometry('1000x650')
        super().__init__(master, padding=15)
        self.grid(row=0, column=0, sticky='nsew')
        self.database  = data.get('database')
        self.previous_data = data
        username_str_var = tk.StringVar()


        # 0 女 1 男 -1 保密
        authority_str_var = tk.IntVar(value =1)

        # 账户信息
        # tk.Label(self, width=10).grid(row=0, column=0)
        tk.Label(self, text='Username: ',width =20).grid(row=1, column=0, sticky=tk.W, pady=50)
        tk.Entry(self, textvariable=username_str_var,width=50).grid(row=1, column=1, sticky=tk.W)


        # 权限 单选框
        tk.Label(self, text='Authority: ',width =20).grid(row=5, column=0, sticky=tk.W, pady=50)
        radio_frame = tk.Frame(self)
        radio_frame.grid(row=5, column=1, sticky=tk.W)
        tk.Radiobutton(radio_frame, text='normal', variable=authority_str_var, value=1).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(radio_frame, text='advanced', variable=authority_str_var, value=0).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(radio_frame, text='admin', variable=authority_str_var, value=-1).pack(side=tk.LEFT, padx=5)


        # 提交按钮
        submit_button = tk.Button(self, text='Submit', width=20)
        submit_button.grid(row=10, column=0,sticky=tk.W,pady=100)
        return_button = tk.Button(self, text='Return', width=20)
        return_button.grid(row=10, column=1, sticky=tk.W, padx = 200,pady=100)
        self.place(relx=0.6, rely=0.5, anchor=CENTER)

        def hit_return_button():
            previous_data = self.previous_data
            self.destroy();
            user_record.UserRecord(master,data=previous_data)
        return_button.configure(command=hit_return_button)

        def hit_submit_button():
            username = username_str_var.get()
            submit_button.configure(state='disabled')
            if(Database.is_username_available(self.database,username)==False):
                au = authority_str_var.get()

                au_str = ""
                if(au == -1): ## admin
                    au_str = "admin"
                elif(au == 0): ## advanced
                    au_str = "advanced"
                else:    ## normal
                    au_str = "normal"
                data ={
                    "username":username,
                    "authority":au_str
                }
                print(au_str)
                result = Database.update_authority(self.database,data)
                if(result == True):
                    messagebox.showinfo(title='Successful Upgrade', message='User has been upgraded successfully!')

            else:
                messagebox.showwarning(title='Username error', message="Username doesn't exist.")
            submit_button.configure(state='normal')
        submit_button.configure(command=hit_submit_button)

if __name__ == "__main__":

    app = tk.Window(resizable=(False,False))
    upgrade(app)
    app.mainloop()









