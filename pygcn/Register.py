import re
from tkinter import font
import json
import ttkbootstrap as tk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
import tkinter
import tkinter.messagebox as messagebox
import Login
import Database
class DataEntryForm(tk.Frame):
    def __init__(self, master,database = None):
        master.title('Register')
        master.geometry('1000x650')
        super().__init__(master, padding=15)
        self.grid(row=0, column=0, sticky='nsew')
        self.database  = database
        username_str_var = tk.StringVar()
        password_str_var = tk.StringVar()
        confirm_password_str_var = tk.StringVar()
        email_str_var = tk.StringVar()

        # 0 女 1 男 -1 保密
        gender_str_var = tk.IntVar(value =1)
        # 兴趣爱好
        hobby_list = [
            [tk.IntVar(), 'Education'],
            [tk.IntVar(), 'Business'],
            [tk.IntVar(), 'Mecical Diagnosis'],
            [tk.IntVar(), 'Others'],
        ]

        # 账户信息
        # tk.Label(self, width=10).grid(row=0, column=0)
        tk.Label(self, text='Username: ',width =20).grid(row=1, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=username_str_var,width=50).grid(row=1, column=1, sticky=tk.W)
        tk.Label(self, text='Password: ',width =20).grid(row=2, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=password_str_var,width=50,show='*').grid(row=2, column=1, sticky=tk.W)
        tk.Label(self, text='Confirm Password: ',width =20).grid(row=3, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=confirm_password_str_var,width=50,show='*',).grid(row=3, column=1, sticky=tk.W)
        tk.Label(self, text='Email: ', width=20).grid(row=4, column=0, sticky=tk.W, pady=20)
        tk.Entry(self, textvariable=email_str_var, width=50).grid(row=4, column=1, sticky=tk.W)

        # 性别 单选框
        tk.Label(self, text='Sex: ',width =20).grid(row=5, column=0, sticky=tk.W, pady=20)
        radio_frame = tk.Frame(self)
        radio_frame.grid(row=5, column=1, sticky=tk.W)
        tk.Radiobutton(radio_frame, text='Male', variable=gender_str_var, value=1).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(radio_frame, text='Female', variable=gender_str_var, value=0).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(radio_frame, text='Nonbinary', variable=gender_str_var, value=-1).pack(side=tk.LEFT, padx=5)

        # Interest
        tk.Label(self, text='Interest: ',width =20).grid(row=6, column=0, sticky=tk.W, pady=20)
        check_frame = tk.Frame(self)
        check_frame.grid(row=6, column=1, sticky=tk.W)
        tk.Checkbutton(check_frame, text=hobby_list[0][1], variable=hobby_list[0][0], bootstyle="round-toggle").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(check_frame, text=hobby_list[1][1], variable=hobby_list[1][0], bootstyle="round-toggle").pack(
            side=tk.LEFT, padx=5)
        tk.Checkbutton(check_frame, text=hobby_list[2][1], variable=hobby_list[2][0], bootstyle="round-toggle").pack(
            side=tk.LEFT, padx=5)
        tk.Checkbutton(check_frame, text=hobby_list[3][1], variable=hobby_list[3][0], bootstyle="round-toggle").pack(side=tk.LEFT, padx=5)

        # 生日
        tk.Label(self, text='Birthday: ',width =20).grid(row=7, column=0, sticky=tk.W, pady=20)
        data_entry = tk.DateEntry(self,width =45)
        data_entry.grid(row=7, column=1, sticky=tk.W, pady=20)
        print(data_entry.entry.get())

        # 提交按钮
        submit_button = tk.Button(self, text='Submit', width=20)
        submit_button.grid(row=10, column=0,sticky=tk.W,pady=20)
        return_button = tk.Button(self, text='Return', width=20)
        return_button.grid(row=10, column=1, sticky=tk.W, padx = 200,pady=20)
        self.place(relx=0.6, rely=0.5, anchor=CENTER)
        def hit_return_button():
            self.destroy();
            Login.DataEntryForm(master,database=self.database)
        return_button.configure(command=hit_return_button)
        def hit_submit_button():
            submit_button.configure(state="disable")
            if(check_null()!=True):
                submit_button.configure(state="normal")
                return False;
            if(check_password_validation()!=True):
                submit_button.configure(state="normal")
                return False;
            if(check_password()!=True):
                submit_button.configure(state="normal")
                return False;
            if(submit_data()==True):
                submit_button.configure(state='normal')
                result = messagebox.showinfo(title='Successful Registration', message='User has been added successfully!')
                if result == 'ok':
                    self.destroy()
                    data={
                        "username": username_str_var.get()
                    }
                    Login.DataEntryForm(master,data = data,database=self.database)

            else:
                messagebox.showwarning(title='Username error', message="Username already exists.")
                submit_button.configure(state='normal')

        submit_button.configure(command=hit_submit_button)

        def submit_data():
            submit_button.configure(state="disable")
            selected_hobbies = [hobby[1] for hobby in hobby_list if hobby[0].get() == 1]
            identity = "normal";
            if(username_str_var.get()=="admin"):
                identity = "admin"
            data = {
                "username": username_str_var.get(),
                "password": password_str_var.get(),
                "email": email_str_var.get(),
                "gender": gender_str_var.get(),
                "birthday":data_entry.entry.get(),
                'interests':selected_hobbies,
                "authority":identity
            }
            if(Database.insert_mydata(self.database,data)==True):
                return True
            else:
                return False;

        def check_password():
            password = password_str_var.get()
            confirm_password = confirm_password_str_var.get()
            if password != confirm_password:
                messagebox.showwarning(title='Password Error', message='Passwords do not match!')
                return False
            else:
                return True
        def check_password_validation():
            password = password_str_var.get()
            if len(password) < 8:
                messagebox.showwarning(title='Password Validation', message="Password should be at least eight characters long.")
                return False
            if not re.match("^[a-zA-Z0-9!@#$%^&*()_+-=\[\]{};':\",./<>?\\\\]*$", password):
                messagebox.showwarning(title='Password Validation', message="Password cannot contain special characters.")
                return False
            if not re.search("[a-z]", password):
                messagebox.showwarning(title='Password Validation',
                                       message="Password must contain at least one lowercase letter.")
                return False
            if not re.search("[A-Z]", password):
                messagebox.showwarning(title='Password Validation',
                                       message="Password must contain at least one uppercase letter.")
                return False
            if not re.search("[0-9]", password):
                messagebox.showwarning(title='Password Validation',
                                       message="Password must contain at least one digit.")
                return False
            return True

        def check_null():
            password = password_str_var.get()
            confirm_password = confirm_password_str_var.get()
            username = username_str_var.get()
            email = email_str_var.get()
            if(password == "" or confirm_password =="" or username == "" or email == ""):
                messagebox.showwarning(title='Information Incomplete',message ='Please complete the information')
                return False
            else:
                return True

if __name__ == "__main__":

    app = tk.Window(resizable=(False,False))
    DataEntryForm(app)
    app.mainloop()









