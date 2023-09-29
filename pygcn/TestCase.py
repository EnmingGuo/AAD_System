import ttkbootstrap as ttk
from ttkbootstrap.tableview import Tableview
from ttkbootstrap.constants import *
import Database
import AidedDiagnosis
class TestCase(ttk.Frame):
    def __init__(self, master, data=None):
        master.geometry("1000x650")  # width and height
        master.title('Test Case ')
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)
        self.previous_data = data
        self.database = data.get('database')
        self.colors = master.style.colors
        self.l1 = [
            {"text": "datetime", "stretch": True},
            {"text": "username","stretch":True},
            {"text": "case","stretch":True},
            {"text": "selected_feature", "stretch": True},
            {"text": "Normal Case Probability", "stretch": True},
            {"text": "Alzheimer\'s Case Probability", "stretch": True},
        ]
        r_set = self.getdata()

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
            AidedDiagnosis.AidedDiagnosis(master,data=submited_data)

        self.b2 = ttk.Button(self.button_frame, text='Back',width=20,command = hit_b2_button)
        self.b2.pack(side='left',padx=(20, 0),pady=(5,0))

    def getdata(self):
        data = {
            'username':self.previous_data.get('username')
        }
        result  = Database.get_all_testcase(self.database,data)
        back_result =[]
        for item in result:
            tuple_temp=(item.get('datetime'),item.get('username'),item.get('case'),item.get('selected_model'),
                       item.get('Normal Case Probability'),item.get('Alzheimer\'s Case Probability'))
            back_result.append(tuple_temp)
        return back_result


if __name__ == '__main__':
    my_w = ttk.Window('Request Record')
    TestCase(my_w,[])
    my_w.mainloop()
