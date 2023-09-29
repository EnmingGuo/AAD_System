import datetime
import pathlib
from pathlib import Path
from queue import Queue
from threading import Thread
from tkinter import messagebox,PhotoImage
from tkinter.filedialog import askdirectory
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
import threading
import time
import user_record
import modelselection
import AidedDiagnosis
from Prepare_Dataset import *
import request_surface
import Login
class open_train_file(ttk.Frame):
    queue = Queue()
    names = []
    open_file_name = []
    searching = False

    def __init__(self, master,data=None):
        master.title('Training Data Preparation')
        master.geometry('1000x650')
        print("nihao")
        super().__init__(master, padding=15)
        self.previous_data = data
        if(self.previous_data!=None):
            self.username = data.get('username');
        self.pack(fill=BOTH, expand=YES)

        # application variables
        _path = pathlib.Path().absolute().as_posix()
        self.path_var = ttk.StringVar(value='C:/Users/lijl7/Desktop/AAD_System/Mat')
        self.store_var = ttk.StringVar(value='C:/Users/lijl7/Desktop/AAD_System/pygcn/Static')
        self.term_var = ttk.StringVar(value='mat')
        self.type_var = ttk.StringVar(value='endswidth')
        PATH = Path(__file__).parent / 'assets'
        images = [
            PhotoImage(
                name='user',
                file=PATH / 'magic_mouse/user_find.png'),
            PhotoImage(
                name = 'record',
                file=PATH/'magic_mouse/Record.png'
            ),
            PhotoImage(
                name = 'message',
                file = PATH/'magic_mouse/message.png'
            ),
            PhotoImage(
                name = 'transfer',
                file=PATH/'magic_mouse/switch.png'
            )
        ]
        self.images = images
        # header and labelframe option container
        option_text = "Welcome "+self.username+"! Complete the form to begin your search"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)
        def user_btn_clicked():
            previous_data = self.previous_data;
            self.destroy();
            user_record.UserRecord(master,data=previous_data)

        user_btn = ttk.Button(
            master=self,
            image='user',
            bootstyle=LINK,
            command=user_btn_clicked
        )
        user_btn.place(x=930, y=0)

        def hit_button_record():
            submit_data = self.previous_data
            data['last_page'] = 'step1'
            self.destroy()
            request_surface.Recordrequest(master,data=submit_data)

        record_btn = ttk.Button(
            master=self,
            image='record',
            bootstyle=LINK,
            command=hit_button_record
        )
        record_btn.place(x=900,y=0)

        def hit_button_switch():
            submit_data = self.previous_data
            submit_data['lastpage'] = "step1"
            self.destroy()
            AidedDiagnosis.AidedDiagnosis(master,data = submit_data)

        switch_btn = ttk.Button(
            master =  self,
            image = 'transfer',
            bootstyle = LINK,
            command=hit_button_switch
        )
        switch_btn.place(x=870, y=0)

        self.create_path_row()
        self.create_store_path_row()
        self.create_term_row()
        self.create_type_row()
        self.create_progress_bar()
        self.create_results_view()

        self.progressbar = ttk.Progressbar(
            master=self,
            mode=INDETERMINATE,
            bootstyle=(STRIPED, SUCCESS)
        )

        def run_process():
            rawPath = self.path_var.get()
            storePath = self.store_var.get()
            print(rawPath, storePath)
            time.sleep(2)
            # 创建 DatasetProgress 实例
            dataset_progress = DatasetProgress()
            # 调用 create_static_dataset 方法，并传入 storePath 和 rawPath
            dataset = dataset_progress.create_static_dataset(storePath, rawPath)

            if(self.previous_data!=None):
                self.previous_data['train_data']=dataset;
            for i in range(0, 2):
                answer = dataset_progress.get_progress()
                if answer == 100:
                    update_progress(100)  # 更新进度条
                    button_Import.configure(state="normal")  # 恢复按钮状态
                    break

        def update_progress(value):
            self.progress_bar.configure(value = value)
            self.pb_lbl.configure(text='100%')

        '''添加部分之最下面的两个按钮'''
        def button_import_hit():
            content = self.path_var.get()
            content2 = self.store_var.get()
            print(content,content2)
            self.threading_num = 0
            print("start")
            button_Import.configure(state="disabled")
            thread = threading.Thread(target=run_process)
            thread.start()


        button_frame = ttk.Frame(self)
        button_frame.pack()
        button_Import = ttk.Button(button_frame, text='Import', width=15)  # 指定父框架，提示文本和宽度
        button_Import.pack(side=ttk.LEFT, padx=5)
        button_Import.config(command = button_import_hit)
        button_nextstep = ttk.Button(button_frame, text='Next', width=15)
        button_nextstep.pack(side=ttk.LEFT, padx=5)
        
        def hit_button_nextstep():
            if 'train_data' in self.previous_data and self.previous_data['train_data']:
                submit_data = self.previous_data
                self.destroy()
                modelselection.ModelSelection(master,data=submit_data)
            else:
                messagebox.showwarning(title='Error', message="You forget to import the dataset!")
        button_nextstep.configure(command=hit_button_nextstep)

        button_backstep = ttk.Button(button_frame, text='Back', width=15)
        button_backstep.pack(side=ttk.LEFT, padx=5)

        def hit_button_backstep():
            last = self.previous_data['lastpage']
            print(last)
            if(last == 'AidedDiagnosis'):
                self.previous_data['lastpage']=''
                submit_data = self.previous_data
                self.destroy()
                AidedDiagnosis.AidedDiagnosis(master,data=submit_data)
            else:
                submit_data = self.previous_data
                self.destroy()
                Login.DataEntryForm(master,data=submit_data,database=submit_data.get('database'))

        button_backstep.config(command=hit_button_backstep)


    def create_store_path_row(self):
        store_path_row = ttk.Frame(self.option_lf)
        store_path_row.pack(fill=X,expand=YES, pady=15)
        store_path_lbl = ttk.Label(store_path_row,text="StorePath",width=8)
        store_path_lbl.pack(side=LEFT,padx=(15,0))
        store_path_ent = ttk.Entry(store_path_row,textvariable=self.store_var)
        store_path_ent.pack(side=LEFT,fill=X,expand=YES,padx=5)
        broswer2_btn = ttk.Button(
            master=store_path_row,
            text="Browse",
            command=self.on_browse2,
            width=8
        )
        broswer2_btn.pack(side=LEFT, padx=5)

    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES,pady = 15)
        path_lbl = ttk.Label(path_row, text="RawPath", width=8)
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
        type_row.pack(fill=X, expand=YES,pady =10)
        type_lbl = ttk.Label(type_row, text="Type", width=8)
        type_lbl.pack(side=LEFT, padx=(15, 0))

        contains_opt = ttk.Radiobutton(
            master=type_row,
            text="Contains",
            variable=self.type_var,
            value="contains"
        )
        contains_opt.pack(side=LEFT)

        startswith_opt = ttk.Radiobutton(
            master=type_row,
            text="StartsWith",
            variable=self.type_var,
            value="startswith"
        )
        startswith_opt.pack(side=LEFT, padx=10)

        endswith_opt = ttk.Radiobutton(
            master=type_row,
            text="EndsWith",
            variable=self.type_var,
            value="endswith"
        )
        endswith_opt.pack(side=LEFT)
        endswith_opt.invoke()

    def create_progress_bar(self):
        progress_bar_row = ttk.Frame(self.option_lf)
        progress_bar_row.pack(fill=X, expand=YES, pady=5)
        progress_bar_lbl = ttk.Label(progress_bar_row, text="Progress", width=8)
        progress_bar_lbl.pack(side=LEFT, padx=(15, 0))
        self.progress_bar = ttk.Progressbar(progress_bar_row,value =0)
        self.progress_bar.pack(fill=X, pady=10, padx=5)
        self.pb_lbl = ttk.Label(self.progress_bar, text='0%', bootstyle=(PRIMARY, INVERSE))
        self.pb_lbl.pack()

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

    def on_browse(self):
        """Callback for directory browse"""
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var.set(path)
    def on_browse2(self):
        path = askdirectory(title="Browse directory")
        if path:
            self.store_var.set(path)
    def on_search(self):
        """Search for a term based on the search type"""
        search_term = self.term_var.get()
        search_path = self.path_var.get()
        search_type = self.type_var.get()
        self.names = []
        if search_term == '':
            return

        # start search in another thread to prevent UI from locking
        Thread(
            target=open_train_file.file_search,
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
            open_train_file.searching,
            not open_train_file.queue.empty()
        ]):
            filename = open_train_file.queue.get()
            self.insert_row(filename, iid)
            self.update_idletasks()
            self.after(100, lambda: self.check_queue(iid))
        elif all([
            not open_train_file.searching,
            not open_train_file.queue.empty()
        ]):
            while not open_train_file.queue.empty():
                filename = open_train_file.queue.get()
                self.insert_row(filename, iid)
            self.update_idletasks()
            self.progressbar.stop()
        elif all([
            open_train_file.searching,
            open_train_file.queue.empty()
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
            _size = open_train_file.convert_size(_stats.st_size)
            _path = file.as_posix()
            self.names.append(_name)
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
        open_train_file.set_searching(1)
        if search_type == 'contains':
            open_train_file.find_contains(term, search_path)
        elif search_type == 'startswith':
            open_train_file.find_startswith(term, search_path)
        elif search_type == 'endswith':
            open_train_file.find_endswith(term, search_path)

    @staticmethod
    def find_contains(term, search_path):
        """Find all files that contain the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if term in file:
                        record = pathlib.Path(path) / file
                        open_train_file.queue.put(record)
        open_train_file.set_searching(False)

    @staticmethod
    def find_startswith(term, search_path):
        """Find all files that start with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.startswith(term):
                        record = pathlib.Path(path) / file
                        open_train_file.queue.put(record)
        open_train_file.set_searching(False)

    @staticmethod
    def find_endswith(term, search_path):
        """Find all files that end with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.endswith(term):
                        record = pathlib.Path(path) / file
                        open_train_file.queue.put(record)
        open_train_file.set_searching(False)

    @staticmethod
    def set_searching(state=False):
        """Set searching status"""
        open_train_file.searching = state

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
    app = ttk.Window("Import Train Dataset")
    open_train_file(app)
    app.mainloop()