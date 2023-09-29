import ttkbootstrap as ttk
from ttkbootstrap.tableview import Tableview
from ttkbootstrap.constants import *
import Database
import modelselection
import step1
class Recordrequest(ttk.Frame):
    def __init__(self, master, data=None):
        master.geometry("1000x650")  # width and height
        master.title('Training Record')
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)
        self.previous_data = data
        self.database = data.get('database')
        self.colors = master.style.colors
        self.l1 = [
            {"text": "datetime", "stretch": True},
            {"text": "selected_model","stretch":True},
            {"text": "device","stretch":True},
            {"text": "batch_size","stretch":True},
            {"text": "test_ratio", "stretch": True},
            {"text": "epoch", "stretch": True},
            {"text": "optimizer", "stretch": True},
            {"text": "selected_feature", "stretch": True},
            {"text": "train_accuracy", "stretch": True},
            {"text": "test_accuracy", "stretch": True},

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
            if(self.previous_data.get('last_page')=='modelselection'):
                submit_data = self.previous_data
                self.destroy()
                modelselection.ModelSelection(master, data=submit_data)
            elif(self.previous_data.get('last_page')=='step1'):
                submit_data = self.previous_data
                self.destroy()
                step1.open_train_file(master, data=submit_data)


        self.b2 = ttk.Button(self.button_frame, text='Back',width=20,command = hit_b2_button)
        self.b2.pack(side='left',padx=(20, 0),pady=(5,0))

    def getdata(self):
        data = {
            'username':self.previous_data.get('username')
        }
        result  = Database.get_all_Training(self.database,data)
        back_result =[]
        for item in result:
            train = round(item.get('train_accuracy'),4)
            test = round(item.get('test_accuracy'),4)
            str=""
            for _ in item.get('selected_feature'):
                str += _
                str += ','
            tuple_temp=(item.get('datetime'),item.get('selected_model'),item.get('device'),item.get('batch_size'),item.get('test_ratio'),
                        item.get('epoch'),item.get('optimizer'),str,train,test)
            back_result.append(tuple_temp)
        return back_result


if __name__ == '__main__':
    my_w = ttk.Window('Request Record')
    Recordrequest(my_w,[])
    my_w.mainloop()
