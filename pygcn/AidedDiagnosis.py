import datetime
import pathlib
from queue import Queue
from threading import Thread
from tkinter import messagebox, PhotoImage
from tkinter.filedialog import askdirectory
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
import Login
import AidedDiagnosis
import Prediction
import TestCase
import Leave_Message
import Database
import step1
from pathlib import Path
import User_modification
class AidedDiagnosis(ttk.Frame):
    queue = Queue()
    searching = False
    def __init__(self, master,data):
        master.title("Aided Diagnosis")
        master.geometry('1000x650')
        super().__init__(master, padding=15)
        self.previous_data = data
        self.database = data.get('database')
        self.pack(fill=BOTH, expand=YES)
        PATH = Path(__file__).parent / 'assets'
        images = [
            PhotoImage(
                name='user',
                file= PATH / 'magic_mouse/user.png'),
            PhotoImage(
                name = 'message',
                file = PATH/'magic_mouse/message.png'
            ),
            PhotoImage(
                name='transfer',
                file=PATH / 'magic_mouse/switch.png'
            ),
            PhotoImage(
                name='record',
                file=PATH/'magic_mouse/Record.png'
            )
        ]
        self.images = images
        # application variables
        self.path_var = ttk.StringVar(value='C:/Users/lijl7/Desktop/AAD_System/Test_Mat')
        self.term_var = ttk.StringVar(value='mat')
        self.type_var = ttk.StringVar(value='endswidth')
        self.model_var = ttk.StringVar(value='sgcn')
        self.sure_var = ttk.StringVar(value='no file')

        # header and labelframe option container
        option_text = "It's time to make aided diagnosis!"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=20)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        def user_btn_clicked():
            # 处理用户单击按钮的操作
            previous_data = self.previous_data;
            self.destroy()
            User_modification.DataEntryForm(master,data=previous_data,database=previous_data['database'])

        def hit_button_record():
            submit_data = self.previous_data
            self.destroy()
            TestCase.TestCase(master, data=submit_data)

        user_btn = ttk.Button(
            master=self,
            image='user',
            bootstyle=LINK,
            command=user_btn_clicked
        )
        user_btn.place(x=930, y=0)

        record_btn = ttk.Button(
            master =self,
            image='record',
            bootstyle=LINK,
            command=hit_button_record
        )
        record_btn.place(x=900,y=0)
        username = self.previous_data.get("username")
        my_authority = Database.get_authority(self.database, username)
        if my_authority == "advanced":
            def hit_button_message():
                submit_data = self.previous_data
                self.destroy()
                Leave_Message.MessageBoard(master,data=submit_data)
            message_btn = ttk.Button(
                master = self,
                image='message',
                bootstyle=LINK,
                command=hit_button_message
            )
            message_btn.place(x=870,y=0)
        elif my_authority == 'admin':
            def hit_button_switch():
                submit_data = self.previous_data
                submit_data['lastpage'] = "AidedDiagnosis"
                self.destroy()
                step1.open_train_file(master,data=submit_data)
            switch_btn = ttk.Button(
                master=self,
                image='transfer',
                bootstyle=LINK,
                command = hit_button_switch
            )
            switch_btn.place(x=870, y=0)

        self.create_path_row()
        self.create_term_row()
        self.create_type_row()
        self.create_model_row()
        self.create_results_view()

        self.progressbar = ttk.Progressbar(
            master=self,
            mode=INDETERMINATE,
            bootstyle=(STRIPED, SUCCESS)
        )

        '''添加确认框'''
        self.create_sure(self)

        '''添加部分之最下面的两个按钮'''
        button_frame = ttk.Frame(self)
        button_frame.pack()
        button_front = ttk.Button(button_frame, text='Back', width=20)  # 指定父框架，提示文本和宽度
        button_front.pack(side=ttk.LEFT, padx=5)
        button_next = ttk.Button(button_frame, text='Next', width=20)
        button_next.pack(side=ttk.LEFT, padx=5)

        def hit_button_backstep():
            if (self.previous_data['lastpage'] == 'step1'):
                self.previous_data['lastpage'] = ''
                submit_data = self.previous_data
                self.destroy()
                step1.open_train_file(master,data=submit_data)
            else:
                submit_data = self.previous_data
                self.destroy()
                Login.DataEntryForm(master, data=submit_data, database=submit_data.get('database'))

        button_front.configure(command=hit_button_backstep)
        def hit_button_next():
            path = self.path_var.get()+'/'+self.sure_var.get()
            if self.sure_var.get() != "no file":
                submit_data =self.previous_data;
                submit_data['lastpath'] = path
                submit_data['model'] = self.model_var.get()
                self.destroy()
                Prediction.predict(master,data=submit_data)
            else:
                messagebox.showwarning(title='Error', message="You forget to choose a test case!")
        button_next.configure(command=hit_button_next)


    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Path", width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent = ttk.Entry(path_row, textvariable=self.path_var)
        path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        browse_btn = ttk.Button(
            master=path_row,
            text="Browse",
            command=self.on_browse,
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def create_sure(self,master):
        container = ttk.Frame(master)
        container.pack(fill=X, pady=15)
        ttk.Label(container).pack(side=LEFT, padx=150)
        sure_lbl = ttk.Label(container, text="Selected file:", width=11)
        sure_lbl.pack(side=LEFT, padx=15)
        sure_ent = ttk.Entry(container, textvariable=self.sure_var,width=20)
        sure_ent.pack(side=LEFT, fill=X, padx=5)

    def create_term_row(self):
        """Add term row to labelframe"""
        term_row = ttk.Frame(self.option_lf)
        term_row.pack(fill=X, expand=YES, pady=15)
        term_lbl = ttk.Label(term_row, text="Term", width=8)
        term_lbl.pack(side=LEFT, padx=(15, 0))
        term_ent = ttk.Entry(term_row, textvariable=self.term_var)
        term_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)
        search_btn = ttk.Button(
            master=term_row,
            text="Search",
            command=self.on_search,
            bootstyle=OUTLINE,
            width=8
        )
        search_btn.pack(side=LEFT, padx=5)

    def create_type_row(self):
        """Add type row to labelframe"""
        type_row = ttk.Frame(self.option_lf)
        type_row.pack(fill=X, expand=YES,pady=10)
        type_lbl = ttk.Label(type_row, text="Type", width=8)
        type_lbl.pack(side=LEFT, padx=(15, 0))

        contains_opt = ttk.Radiobutton(
            master=type_row,
            text="Contains",
            variable=self.type_var,
            value="contains"
        )
        contains_opt.pack(side=LEFT,padx=15)

        startswith_opt = ttk.Radiobutton(
            master=type_row,
            text="StartsWith",
            variable=self.type_var,
            value="startswith"
        )
        startswith_opt.pack(side=LEFT, padx=15)

        endswith_opt = ttk.Radiobutton(
            master=type_row,
            text="EndsWith",
            variable=self.type_var,
            value="endswith"
        )
        endswith_opt.pack(side=LEFT,padx=15)
        endswith_opt.invoke()

    def create_model_row(self):
        """Add model row to labelframe"""
        type_row = ttk.Frame(self.option_lf)
        type_row.pack(fill=X, expand=YES,pady=10)
        type_lbl = ttk.Label(type_row, text="Model", width=8)
        type_lbl.pack(side=LEFT, padx=(15, 0))

        sgcn_opt = ttk.Radiobutton(
            master=type_row,
            text="Static GCN",
            variable=self.model_var,
            value="sgcn"
        )
        sgcn_opt.pack(side=LEFT, padx=15)

        sgat_opt = ttk.Radiobutton(
            master=type_row,
            text="Static GAT",
            variable=self.model_var,
            value="sgat"
        )
        sgat_opt.pack(side=LEFT, padx=15)

        ssage_opt = ttk.Radiobutton(
            master=type_row,
            text="Static GraphSage",
            variable=self.model_var,
            value="ssage"
        )
        ssage_opt.pack(side=LEFT, padx=15)

        dgcn_opt = ttk.Radiobutton(
            master=type_row,
            text="Evolving GCN",
            variable=self.model_var,
            value="dgcn"
        )
        dgcn_opt.pack(side=LEFT,padx=15)

        dgat_opt = ttk.Radiobutton(
            master=type_row,
            text="Dynamic GAT",
            variable=self.model_var,
            value="dgat"
        )
        dgat_opt.pack(side=LEFT,padx=15)

        dsage_opt = ttk.Radiobutton(
            master=type_row,
            text="Evolving GraphSage",
            variable=self.model_var,
            value="dsage"
        )
        dsage_opt.pack(side=LEFT,padx=15)


    def create_results_view(self):
        """Add result treeview to labelframe"""
        self.resultview = ttk.Treeview(
            master=self,
            bootstyle=INFO,
            columns=[0, 1, 2, 3, 4],
            show=HEADINGS
        )
        self.resultview.pack(fill=BOTH, expand=YES, pady=10)

        # setup columns and use `scale_size` to adjust for resolution
        self.resultview.heading(0, text='Name', anchor=W)
        self.resultview.heading(1, text='Modified', anchor=W)
        self.resultview.heading(2, text='Type', anchor=E)
        self.resultview.heading(3, text='Size', anchor=E)
        self.resultview.heading(4, text='Path', anchor=W)
        self.resultview.column(
            column=0,
            anchor=W,
            width=utility.scale_size(self, 125),
            stretch=False
        )
        self.resultview.column(
            column=1,
            anchor=W,
            width=utility.scale_size(self, 140),
            stretch=False
        )
        self.resultview.column(
            column=2,
            anchor=E,
            width=utility.scale_size(self, 50),
            stretch=False
        )
        self.resultview.column(
            column=3,
            anchor=E,
            width=utility.scale_size(self, 50),
            stretch=False
        )
        self.resultview.column(
            column=4,
            anchor=W,
            width=utility.scale_size(self, 300)
        )
        def treeviewClick(event):  # 单击
            for item in self.resultview.selection():
                item_text = self.resultview.item(item, "values")
                self.sure_var.set(item_text[0]+'.'+self.term_var.get())
        self.resultview.bind('<ButtonRelease-1>', treeviewClick)

    def on_browse(self):
        """Callback for directory browse"""
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var.set(path)

    def on_search(self):
        """Search for a term based on the search type"""
        search_term = self.term_var.get()
        search_path = self.path_var.get()
        search_type = self.type_var.get()
        if search_term == '':
            return

        # start search in another thread to prevent UI from locking
        Thread(
            target=AidedDiagnosis.file_search,
            args=(search_term, search_path, search_type),
            daemon=True
        ).start()
        self.progressbar.start(10)

        iid = self.resultview.insert(
            parent='',
            index=END,
        )
        self.resultview.item(iid, open=True)
        self.after(100, lambda: self.check_queue(iid))

    def check_queue(self, iid):
        """Check file queue and print results if not empty"""
        if all([
            AidedDiagnosis.searching,
            not AidedDiagnosis.queue.empty()
        ]):
            filename = AidedDiagnosis.queue.get()
            self.insert_row(filename, iid)
            self.update_idletasks()
            self.after(100, lambda: self.check_queue(iid))
        elif all([
            not AidedDiagnosis.searching,
            not AidedDiagnosis.queue.empty()
        ]):
            while not AidedDiagnosis.queue.empty():
                filename = AidedDiagnosis.queue.get()
                self.insert_row(filename, iid)
            self.update_idletasks()
            self.progressbar.stop()
        elif all([
            AidedDiagnosis.searching,
            AidedDiagnosis.queue.empty()
        ]):
            self.after(100, lambda: self.check_queue(iid))
        else:
            self.progressbar.stop()

    def insert_row(self, file, iid):
        """Insert new row in tree search results"""
        try:
            _stats = file.stat()
            _name = file.stem
            _timestamp = datetime.datetime.fromtimestamp(_stats.st_mtime)
            _modified = _timestamp.strftime(r'%m/%d/%Y %I:%M:%S%p')
            _type = file.suffix.lower()
            _size = AidedDiagnosis.convert_size(_stats.st_size)
            _path = file.as_posix()
            iid = self.resultview.insert(
                parent='',
                index=END,
                values=(_name, _modified, _type, _size, _path)
            )
            self.resultview.selection_set(iid)
            self.resultview.see(iid)
        except OSError:
            return

    @staticmethod
    def file_search(term, search_path, search_type):
        """Recursively search directory for matching files"""
        AidedDiagnosis.set_searching(1)
        if search_type == 'contains':
            AidedDiagnosis.find_contains(term, search_path)
        elif search_type == 'startswith':
            AidedDiagnosis.find_startswith(term, search_path)
        elif search_type == 'endswith':
            AidedDiagnosis.find_endswith(term, search_path)

    @staticmethod
    def find_contains(term, search_path):
        """Find all files that contain the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if term in file:
                        record = pathlib.Path(path) / file
                        AidedDiagnosis.queue.put(record)
        AidedDiagnosis.set_searching(False)

    @staticmethod
    def find_startswith(term, search_path):
        """Find all files that start with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.startswith(term):
                        record = pathlib.Path(path) / file
                        AidedDiagnosis.queue.put(record)
        AidedDiagnosis.set_searching(False)

    @staticmethod
    def find_endswith(term, search_path):
        """Find all files that end with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.endswith(term):
                        record = pathlib.Path(path) / file
                        AidedDiagnosis.queue.put(record)
        AidedDiagnosis.set_searching(False)

    @staticmethod
    def set_searching(state=False):
        """Set searching status"""
        AidedDiagnosis.searching = state

    @staticmethod
    def convert_size(size):
        """Convert bytes to mb or kb depending on scale"""
        kb = size // 1000
        mb = round(kb / 1000, 1)
        if kb > 1000:
            return f'{mb:,.1f} MB'
        else:
            return f'{kb:,d} KB'


if __name__ == '__main__':
    app = ttk.Window("Aided Diagnosis")
    AidedDiagnosis(app,[])
    app.mainloop()