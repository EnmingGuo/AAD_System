import ttkbootstrap as ttk
from ttkbootstrap.tableview import Tableview
from ttkbootstrap.constants import *
import Database
import AidedDiagnosis
import step1
import upgrade
class UserRecord(ttk.Frame):
    def __init__(self, master, data=None):
        master.geometry("1000x650")  # width and height
        master.title('User Record')
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)
        self.previous_data = data
        self.database = data.get('database')
        self.colors = master.style.colors
        self.l1 = [
            {"text": "username", "stretch": True},
            {"text": "email","stretch":True},
            {"text": "gender", "stretch": True},
            {"text": "birthday", "stretch": True},
            {"text": "interests", "stretch": True},
            {"text": "authority", "stretch": True},
        ]
        r_set = self.getdata();

        self.dv = ttk.tableview.Tableview(
            master=self,
            paginated=True,
            coldata=self.l1,
            rowdata=r_set,
            searchable=True,
            bootstyle=INFO,
            pagesize=26,
            height=20,
            stripecolor=(self.colors.light, None),
        )
        self.dv.pack(expand=True, fill='both')  # 使用pack()方法并指定expand=True和fill='both'
        self.dv.autofit_columns() # Fit in current view
        self.dv.load_table_data() # Load all data rows
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side='bottom', pady=20)

        self.b1 = ttk.Button(self.button_frame, text='Export',command=lambda: self.dv.export_current_selection(),width=20)
        self.b1.pack(side='left',padx=(20, 0),pady=(5,0))

        def hit_b2_button():
            submited_data = self.previous_data;
            self.destroy();
            step1.open_train_file(master,data=submited_data)

        self.b2 = ttk.Button(self.button_frame, text='Back',width=20,command = hit_b2_button)
        self.b2.pack(side='left',padx=(20, 0),pady=(5,0))
        def hit_b3_button():
            previous_data = self.previous_data;
            self.destroy();
            upgrade.upgrade(master,data = previous_data)


        self.b3 = ttk.Button(self.button_frame, text='Upgrade', width=20,command =  hit_b3_button)
        self.b3.pack(side='left', padx=(20, 0), pady=(5, 0))

    def getdata(self):
        result  = Database.get_all_user(self.database)
        back_result =[]
        for item in result:
            sex_str = ""
            interest = item.get('interests')
            joined_string = ", ".join(interest)
            # 移除字符串末尾的逗号
            if joined_string.endswith(","):
                joined_string = joined_string[:-1]
            sex = item.get('gender')
            if sex == 1:
                sex_str = "Male"
            elif sex == 0:
                sex_str = "Female"
            else:
                sex_str = "Non-binary"
            tuple_temp=(item.get('username'),item.get('email'),sex_str,
                       item.get('birthday'),joined_string,item.get('authority'))
            back_result.append(tuple_temp)
        return back_result


if __name__ == '__main__':
    my_w = ttk.Window('User Record')
    UserRecord(my_w,[])
    my_w.mainloop()

