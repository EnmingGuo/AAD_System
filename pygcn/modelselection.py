import math
import copy
from pathlib import Path
from tkinter import PhotoImage, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import Database
import datetime
import request_surface
import static_GraphSAGE
import step1
import static_GCN
import static_GAT
import dynamic_GCN
import dynamic_GraphSAGE
import uuid
class ModelSelection(ttk.Frame):
    def __init__(self,master,data=None):
        PATH = Path(__file__).parent / 'assets'
        print(PATH)
        master.title('Model Selection')
        master.geometry('1000x650')
        super().__init__(master)
        self.previous_data = data
        self.database = self.previous_data.get('database')
        self.pack(fill=BOTH, expand=YES)
        images = [
            PhotoImage(
                name='reset',
                file=PATH / 'magic_mouse/icons8_reset_24px.png'),
            PhotoImage(
                name='confirm',
                file=PATH/'magic_mouse/confirm.png'
            ),
            PhotoImage(
                name='reset-small',
                file=PATH / 'magic_mouse/icons8_reset_16px.png'),
            PhotoImage(
                name='submit',
                file=PATH / 'magic_mouse/icons8_submit_progress_24px.png'),
            PhotoImage(
                name='brain',
                file=PATH/'magic_mouse/brain_title.png'
            ),
            PhotoImage(
                name='syringe',
                file=PATH/'magic_mouse/sharpicons_syringe.png'
            ),
            PhotoImage(
                name='doctornotes',
                file=PATH/'magic_mouse/sharpicons_doctor-notes.png'
            ),
            PhotoImage(
                name='back',
                file=PATH / 'magic_mouse/rtn.png'
            )
        ]
        self.images  =images
        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(0, weight=1)

        # column 1
        col1 = ttk.Frame(self, padding=10)
        col1.grid(row=0, column=0, sticky=NSEW)

        # device info
        dev_info = ttk.Labelframe(col1, text='Main Panel', padding=10)
        dev_info.pack(side=TOP, fill=BOTH, expand=YES)

        # header
        dev_info_header = ttk.Frame(dev_info, padding=5)
        dev_info_header.pack(fill=X)

        btn = ttk.Button(
            master=dev_info_header,
            image='syringe',
            bootstyle=LINK,
        )
        btn.pack(side=LEFT)

        lbl = ttk.Label(dev_info_header, text='Model 2023, Alzheimer\'s Diagnosis')
        lbl.pack(side=LEFT, fill=X, padx=30)

        btn = ttk.Button(
            master=dev_info_header,
            image='doctornotes',
            bootstyle=LINK,
        )
        btn.pack(side=LEFT)

        # image
        ttk.Label(dev_info, image='brain').pack(fill=X)

        # progressbar
        self.pb = ttk.Progressbar(dev_info, value=0)
        self.pb.pack(fill=X, pady=30, padx=5)
        self.pb_lbl= ttk.Label(self.pb, text='0%', bootstyle=(PRIMARY, INVERSE))
        self.pb_lbl.pack()

        # progress message
        self.progress_message = 'Model has not been selected !'
        self.pm_lbl = ttk.Label(
            master=dev_info,
            text=self.progress_message,
            font='Helvetica 10',
            anchor=CENTER
        )
        self.pm_lbl.pack(fill=X)

        # licence info
        lic_info = ttk.Labelframe(col1, text='Training Switch', padding=20)
        lic_info.pack(side=TOP, fill=BOTH, expand=YES, pady=(10, 0))
        lic_info.rowconfigure(0, weight=2)
        lic_info.columnconfigure(0, weight=2)

        first_row_frame = ttk.Frame(master = lic_info)
        first_row_frame.pack(side=TOP,fill=X,pady=(0,20))
        self.lic_info_model = ''
        self.lic_title = ttk.Label(
            master=first_row_frame,
            width=15,
            text='Selectd Model: ',
            anchor=CENTER,
            font = 'Helvetica 10'
        )
        self.lic_title.pack(side=LEFT,padx=(20, 0))

        self.lic_Model = ttk.Label(
            master=first_row_frame,
            width=20,
            text=self.lic_info_model,
            bootstyle=PRIMARY,
            font='Helvetica 10',
            anchor=CENTER

        )
        self.lic_Model.pack(side=LEFT, padx=(20, 0))

        second_row_frame = ttk.Frame(master=lic_info)
        second_row_frame.pack(side=TOP, fill=X, pady=(0, 20))
        self.second_lbl = ttk.Label(
            master=second_row_frame,
            width=15,
            text='Username:',
            anchor=CENTER,
            font='Helvetica 10'
        )
        self.second_lbl.pack(side=LEFT, padx=(20, 0))
        username_given = "Unknown"
        if (self.previous_data != None):
            username_given = self.previous_data.get('username')

        self.lic_num = ttk.Label(
            master=second_row_frame,
            width=20,
            text=username_given,
            bootstyle=PRIMARY,
            font='Helvetica 10',
            anchor=CENTER
        )
        self.lic_num.pack(side=LEFT, padx=(20, 0))

        self.model_pb = ttk.Progressbar(lic_info, value=0)
        self.model_pb .pack(fill=X, pady=25, padx=5)
        self.mpb_lbl = ttk.Label(self.model_pb, text='0%', bootstyle=(PRIMARY, INVERSE))
        self.mpb_lbl.pack()
        def portal():
            submit_data = self.previous_data;
            self.destroy();
            step1.open_train_file(master,data=submit_data)

        def trainModel():
            if(self.lic_info_model==""):
                messagebox.showwarning(title='Request error', message="You have not choose a model!")
                return ;
            self.previous_data['model'] = self.lic_info_model;
            self.previous_data['is_shuffle'] = self.is_DataShuffle.get();
            self.previous_data['device'] = self.cbo.get()
            self.previous_data['optimizer'] = self.optimizer_cbo.get()
            self.previous_data['Batch_size'] = self.Batch_scale_value.get();
            self.previous_data['Test_ratio'] = self.Test_Ratio_scale_value.get() / 100;
            self.previous_data['Epoch'] = self.epoch_scale_value.get();
            selected_features = [item[1] for item in self.feature_list if item[0].get() == 1]
            if(len(selected_features)== 0):
                messagebox.showwarning(title='Request error', message="You have not choose features!")
                return;
            self.confirm_now.configure(state='disabled');
            self.previous_data['feature'] = selected_features
            # 生成一个 UUID
            unique_id = str(uuid.uuid4())
            self.previous_data['requestId'] = unique_id
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
            data = {
                'username': self.previous_data['username'],
                'request_Id': unique_id,
                'selected_model': self.lic_info_model,
                'is_shuffle': self.previous_data['is_shuffle'],
                'device': self.previous_data['device'],
                'batch_size': self.previous_data['Batch_size'],
                'test_ratio': self.previous_data['Test_ratio'],
                'epoch': self.previous_data['Epoch'],
                'selected_feature': self.previous_data['feature'],
                'optimizer': self.previous_data['optimizer'],
                'datetime': time_string
            }
            if (self.submit_data(data) == True):
                result = messagebox.showinfo(title='Successful Request',
                                             message='The Training is successfully requested!')
                if result == 'ok':
                    rtn = None
                    used_data = self.previous_data
                    if(self.lic_info_model=='STATIC GCN'):
                        rtn = static_GCN.static_gcn(used_data)
                    elif(self.lic_info_model=='STATIC GraphSage'):
                        rtn = static_GraphSAGE.graphSage(used_data)
                    elif(self.lic_info_model=="STATIC GAT" or self.lic_info_model=="Dynamic GAT"):
                        rtn = static_GAT.static_gat(used_data)
                    elif (self.lic_info_model == "Dynamic GCN"):
                        rtn = dynamic_GCN.evolving_gcn(used_data)
                    elif (self.lic_info_model == "Dynamic GraphSage"):
                        rtn = dynamic_GraphSAGE.evolving_graphSage(used_data)
                    if (len(rtn) == 2):
                        self.model_pb.configure(value=100)
                        self.mpb_lbl.configure(text="100%")
                        result2 = messagebox.showinfo(title='Successful Training',
                                                      message='The result is already!')
                        print(rtn)
                        if result2 == 'ok':
                            self.previous_data['train_accuracy'] = rtn[0]
                            self.previous_data['test_accuracy'] = rtn[1]
                            second_data = {
                                'request_Id': unique_id,
                                'train_accuracy': rtn[0],
                                'test_accuracy': rtn[1]
                            }
                            result3 = Database.update_Training(self.database,second_data)
                            if result3 == True:
                                result4 = messagebox.showinfo(title='Answer Already',
                                                              message='The Training result is already!')
                                if result4 == 'ok':
                                    submit_data = self.previous_data
                                    submit_data['last_page'] = 'modelselection'
                                    self.destroy()
                                    request_surface.Recordrequest(master,data=submit_data)
            else:
                messagebox.showwarning(title='Request error', message="The training request is failed!")
                return;
        self.confirm_now = ttk.Button(
            master=lic_info,
            text='Confirm',
            image='confirm',
            compound=BOTTOM,
            command=trainModel
        )
        self.confirm_now.pack(padx=10, fill=X,pady=10)

        self.back_now = ttk.Button(
            master=lic_info,
            text='Back',
            image='back',
            compound=BOTTOM,
            command = portal
        )
        self.back_now.pack(padx=10, fill=X,pady=10)

        # Column 2
        col2 = ttk.Frame(self, padding=10)
        col2.grid(row=0, column=1, sticky=NSEW)

        # scrolling
        scrolling = ttk.Labelframe(col2, text='Model Selection', padding=(15, 10))
        scrolling.pack(side=TOP, fill=BOTH, expand=YES)

        # set the selected status
        selected = ttk.IntVar(value = 1)
        unselected = ttk.IntVar(value = 0)

        self.is_SGCN = ttk.IntVar(value = 0)
        self.is_SGAT = ttk.IntVar(value = 0)
        self.is_SGraphSage= ttk.IntVar(value = 0)
        self.is_DGCN = ttk.IntVar(value = 0)
        self.is_DGAT = ttk.IntVar(value = 0)
        self.is_DGraphSage = ttk.IntVar(value = 0)

        # Is Data shuffled?
        self.is_DataShuffle = ttk.IntVar(value = 1)


        op1 = ttk.Checkbutton(scrolling, text='Static Model', variable=selected, state='disabled')
        op1.pack(fill=X, pady=10)

        def set_checked(var,checkbutton):
            if var.get() == 1:
                child_text = checkbutton.cget("text")
                self.lic_info_model=child_text;
                self.lic_Model.configure(text=self.lic_info_model)
                var.set(1)
                self.pb.configure(value =100)
                self.pb_lbl.configure(text='100%')
                self.progress_message='Model is ready!'
                self.pm_lbl.configure(text=self.progress_message)
                for child in checkbutton.master.winfo_children():
                    if isinstance(child, ttk.Checkbutton) and child is not checkbutton:
                        child.config(state='disabled')
            else:
                var.set(0)
                self.pb.configure(value=0)
                self.pb_lbl.configure(text='0%')
                self.progress_message = 'Model has not been selected!'
                self.pm_lbl.configure(text=self.progress_message)
                self.lic_info_model = '';
                self.lic_Model.configure(text=self.lic_info_model)
                for child in checkbutton.master.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        child_text = child.cget("text");
                        if(child_text == 'Static Model' or child_text == 'Dynamic Model'):
                            continue;
                        child.config(state='normal')
        def set_check_simplfied(var):
            if var.get() == 1:
                var.set(1)
            else:
                var.set(0)

        # Static GCN
        op2 = ttk.Checkbutton(
            master=scrolling,
            text='STATIC GCN',
            variable=self.is_SGCN,
            command=lambda: set_checked(self.is_SGCN,op2)
        )
        op2.pack(fill=X, padx=(20, 0), pady=10)

        # STATIC GAT
        op3 = ttk.Checkbutton(
            master=scrolling,
            text='STATIC GAT',
            variable=self.is_SGAT,
            command=lambda: set_checked(self.is_SGAT,op3)
        )
        op3.pack(fill=X, padx=(20, 0), pady=10)

        # Scroll only vertical or horizontal
        op4 = ttk.Checkbutton(
            master=scrolling,
            text='STATIC GraphSage',
            variable=self.is_SGraphSage,
            command=lambda: set_checked(self.is_SGraphSage,op4)
        )
        op4.pack(fill=X, padx=(20, 0), pady=10)

        # Label of Dynamic Tick
        op8 = ttk.Checkbutton(scrolling, text='Dynamic Model', variable=selected, state='disabled')
        op8.pack(fill=X, pady=10)


        # DYNAMIC - GCN
        op5 = ttk.Checkbutton(
            master=scrolling,
            text='Dynamic GCN',
            variable=self.is_DGCN,
            command=lambda: set_checked(self.is_DGCN,op5)
        )
        op5.pack(fill=X, padx=(20, 0), pady=10)
        # DYNAMIC - GAT
        op6 = ttk.Checkbutton(
            master=scrolling,
            text='Dynamic GAT',
            variable=self.is_DGAT,
            command=lambda: set_checked(self.is_DGAT,op6)
        )
        op6.pack(fill=X, padx=(20, 0), pady=10)

        dynamic_graph_cb = ttk.Checkbutton(
            master=scrolling,
            text='Dynamic GraphSage',
            variable=self.is_DGraphSage,
            command=lambda: set_checked(self.is_DGraphSage,dynamic_graph_cb)
        )
        dynamic_graph_cb.pack(fill=X, padx=(20, 0), pady=10)

        # 1 finger gestures
        data_structure = ttk.Labelframe(
            master=col2,
            text='Data Structure',
            padding=(15, 10)
        )
        data_structure.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )

        op9 = ttk.Checkbutton(
            master=data_structure,
            text='Data Shuffle',
            variable=self.is_DataShuffle,
            command=lambda: set_check_simplfied(self.is_DataShuffle)
        )
        op9.pack(fill=X, padx=(20, 0), pady=5)

        # Get Test Ratio
        Test_Ratio = ttk.Frame(data_structure)
        Test_Ratio.pack(fill=X, padx=(20, 0), pady=(5, 0))
        def update_TestRatio_scale_label(Test_Ratio_scale_value):
            temp = int(float(Test_Ratio_scale_value))
            xiaoshu = temp / 100
            xiaoshu_formatted = "{:.2f}".format(xiaoshu)
            self.Test_Ratio_label_value.configure(text=str(xiaoshu_formatted))

        ttk.Label(Test_Ratio, text='Test Ratio:').pack(side=LEFT)
        self.Test_Ratio_scale_value = ttk.IntVar(value = 20)
        self.Test_Ratio_label_value = ttk.Label(Test_Ratio,text='0.20')
        self.Test_Ratio_Scale = ttk.Scale(Test_Ratio, variable=self.Test_Ratio_scale_value, from_=10, to=30,command=update_TestRatio_scale_label)
        self.Test_Ratio_Scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        self.Test_Ratio_label_value.pack(side="left")
        Test_Ratio_Btn = ttk.Button(
            master=Test_Ratio,
            image='reset-small',
            bootstyle=LINK,
            command=lambda: self.reset_scale_value(self.Test_Ratio_Scale)
        )
        Test_Ratio_Btn.pack(side=LEFT)

        # gest Batch
        Batch_Size = ttk.Frame(data_structure)
        Batch_Size.pack(fill=X, padx=(20, 0), pady=(5, 0))
        def update_batch_scale_label(batch_scale_value):
            temp = int(float(batch_scale_value))
            self.Batch_label_value.configure(text=str(temp))
        ttk.Label(Batch_Size, text='Batch Size:').pack(side=LEFT)
        self.Batch_scale_value = ttk.IntVar(value = 60)
        self.Batch_label_value = ttk.Label(Batch_Size , text ='60')
        self.Batch_Size_Scale = ttk.Scale(Batch_Size, variable=self.Batch_scale_value, from_=10, to=100,command=update_batch_scale_label)
        self.Batch_Size_Scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        self.Batch_label_value.pack(side="left")

        Batch_Size_Btn = ttk.Button(
            master=Batch_Size,
            image='reset-small',
            bootstyle=LINK,
            command=lambda :self.reset_scale_value(self.Batch_Size_Scale)
        )
        Batch_Size_Btn.pack(side=LEFT)


        # middle click
        middle_click = ttk.Labelframe(
            master=col2,
            text='Device Selection',
            padding=(15, 10)
        )
        middle_click.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )
        self.cbo = ttk.Combobox(
            master=middle_click,
            values=['CPU', 'GPU', 'Other']
        )
        self.cbo.current(0)
        self.cbo.pack(fill=X)

        # Column 3
        col3 = ttk.Frame(self, padding=10)
        col3.grid(row=0, column=2, sticky=NSEW)

        # two finger gestures
        Training_parameter = ttk.Labelframe(
            master=col3,
            text='Training Information',
            padding=10
        )
        Training_parameter.pack(side=TOP, fill=BOTH)

        lbl = ttk.Label(
            master=Training_parameter,
            text='Training Epoch: '
        )
        lbl.pack(fill=X, pady = 5)

        op8 = ttk.Checkbutton(
            master=Training_parameter,
            text='Iterative Training',
            variable='op8',
            state = 'disabled'
        )
        op8.pack(fill=X, padx=(20, 0), pady=5)

        # gest sense
        def update_epoch_scale_label(epoch_scale_value):
            temp = int(float(epoch_scale_value))
            self.epoch_label_value.config(text=str(temp))

        gest_sense_frame = ttk.Frame(Training_parameter)
        gest_sense_frame.pack(fill=X, padx=(20, 0), pady=(5, 0))
        ttk.Label(gest_sense_frame, text='Epoch:').pack(side=LEFT)
        self.epoch_scale_value = ttk.IntVar(value =20)
        self.epoch_label_value = ttk.Label(gest_sense_frame, text='20')
        self.Epoch_scale = ttk.Scale(gest_sense_frame, variable=self.epoch_scale_value,from_=10, to=50,command=update_epoch_scale_label)
        self.Epoch_scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        self.epoch_label_value.pack(side="left")

        btn = ttk.Button(
            master=gest_sense_frame,
            image='reset-small',
            bootstyle=LINK,
            command=lambda :self.reset_scale_value(self.Epoch_scale)
        )
        btn.pack(side=LEFT)


        # Training Parameters

        lbl = ttk.Label(
            master=Training_parameter,
            text='Optimizer Information: '
        )
        lbl.pack(fill=X, pady=(10, 5))

        tmp1 = ttk.Checkbutton(
            master=Training_parameter,
            text='Cross-entropy Function',
            variable = selected,
            state='disabled'
        )
        tmp1.pack(fill=X, padx=(20, 0), pady=5)

        tmp2 = ttk.Checkbutton(
            master=Training_parameter,
            text='Adaptive Moment Estimation',
            variable=selected,
            state='disabled'

        )
        tmp2.pack(fill=X, padx=(20, 0), pady=5)

        self.optimizer_cbo = ttk.Combobox(
            master=Training_parameter,
            values=['Average Epoch', 'Optimal Epoch','Latest Epoch']
        )
        self.optimizer_cbo.current(1)
        self.optimizer_cbo.pack(fill=X, padx=(20, 0), pady=5)

        # mouse options
        self.feature_list = [
            [ttk.IntVar(), 'Degree'],
            [ttk.IntVar(), 'Betweeness Centrality'],
            [ttk.IntVar(), 'Closeness Centrality'],
            [ttk.IntVar(), 'Degree Centrality'],
            [ttk.IntVar(), 'Kernel Shell'],
            [ttk.IntVar(), 'Pagerank'],
            [ttk.IntVar(), 'Flow Coefficiency'],
            [ttk.IntVar(), 'Local Efficiency']
        ]

        mouse_options = ttk.Labelframe(
            master=col3,
            text='Feature Selection',
            padding=(15, 10)
        )
        mouse_options.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )

        feature1 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[0][1],
            variable=self.feature_list[0][0],
            bootstyle="round-toggle"
        )
        feature1.pack(fill=X, pady=8)

        feature2 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[1][1],
            variable=self.feature_list[1][0],
            bootstyle="round-toggle"
        )
        feature2.pack(fill=X, pady=8)

        feature3 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[2][1],
            variable=self.feature_list[2][0],
            bootstyle="round-toggle"
        )
        feature3.pack(fill=X, pady=8)

        feature4 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[3][1],
            variable=self.feature_list[3][0],
            bootstyle="round-toggle"
        )
        feature4.pack(fill=X, pady=8)

        feature5 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[4][1],
            variable=self.feature_list[4][0],
            bootstyle="round-toggle"
        )
        feature5.pack(fill=X, pady=8)

        feature6 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[5][1],
            variable=self.feature_list[5][0],
            bootstyle="round-toggle"
        )
        feature6.pack(fill=X, pady=8)

        feature7 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[6][1],
            variable=self.feature_list[6][0],
            bootstyle="round-toggle"
        )
        feature7.pack(fill=X, pady=8)

        feature8 = ttk.Checkbutton(
            master=mouse_options,
            text=self.feature_list[7][1],
            variable=self.feature_list[7][0],
            bootstyle="round-toggle"
        )
        feature8.pack(fill=X, pady=8)

        # base speed
        base_speed_sense_frame = ttk.Frame(mouse_options)
        base_speed_sense_frame.pack(fill=X, padx=(20, 0), pady=(20, 0))
        base_speed_sense_btn = ttk.Button(master=base_speed_sense_frame,text='Fully Set',width=20)
        base_speed_sense_btn.configure(command=self.fullyset)
        base_speed_sense_btn.pack(side=LEFT)

    def fullyset(self):
        for feature in self.feature_list:
            feature[0].set(1)


        # turn on all checkbuttons
        for i in range(1, 14):
            self.setvar(f'op{i}', 1)

        # turn off select buttons
        for j in [2, 9, 12, 13]:
            self.setvar(f'op{j}', 0)

    def callback(self):
        """Demo callback"""
        Messagebox.ok(
            title='Button callback',
            message="You pressed a button."
        )


    def submit_data(self,data):
        if (Database.insert_Training(self.database,data) == True):
            return True
        else:
            return False

    def reset_scale_value(self,scale):
        middle_value = (scale['from'] + scale['to']) // 2
        scale.set(middle_value)

if __name__ == '__main__':

    app = ttk.Window("Model Setting", "yeti")
    ModelSelection(app)
    app.mainloop()
