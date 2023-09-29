import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from random import randint
import gspan_function
import step1
import AidedDiagnosis
import modelselection
from Prepare_Dataset import *
_flag = True
#_flag = False

class train(ttk.Frame):

    def __init__(self, master,filenames:list):
        master.title('Import Raw Data')
        master.geometry('1000x650')
        self.filenames = list(filenames)
        print(self.filenames)
        super().__init__(master, padding=20)
        self.pack(fill=BOTH, expand=YES)
        container1 = ttk.Frame(self)
        container1.pack(side=TOP, fill=X, padx=60,pady=(80,20))
        self.create_Meter(container1)
        container = ttk.Frame(self)
        container.pack(side=TOP, fill=X, padx=60, pady=100)
        ttk.Label(container).pack(side=LEFT, padx=70)
        button_front = ttk.Button(container, text='Return', bootstyle="danger-outline",width=20)
        button_front.pack(side=LEFT, padx=50)
        button_next = ttk.Button(container, text='Next', bootstyle="danger-outline",width=20)
        button_next.pack(side=LEFT,padx = 100)

        def hit_button_front():
            self.destroy()
            step1.open_train_file(master)
        button_front.configure(command=hit_button_front)
        def hit_button_next():
            self.destroy()
            #step3.open_train_file(master,self.filenames)
            modelselection.ModelSelection(master)
        button_next.configure(command=hit_button_next)

    def create_band(self, master, dic):
        """Create and pack an equalizer band"""
        value = randint(1, 99)
        self.setvar(dic['name'], value)

        container = ttk.Frame(self.option_lf)
        container.pack(side=TOP, fill=X, padx=10)

        #调整三个进度条的左边和框框的距离
        ttk.Label(container).pack(side=LEFT,padx=30)

        # header label
        hdr = ttk.Label(container, text=dic['name'], anchor=CENTER)
        hdr.pack(side=LEFT, fill=Y, padx=20, pady=20)

        # volume scale

        scale = ttk.Scale(
            master=container,
            orient=HORIZONTAL,
            from_=0,
            to=100,
            value=value,
            command=lambda x=value, y=dic['name']: self.update_value(x, y),
            bootstyle=INFO,
            length=200,
            #resolution=dic['resolution'],
        )
        scale.pack(side=LEFT,fill=Y, padx=20, pady=20)

        # value label
        val = ttk.Label(master=container, textvariable=dic['name'])
        val.pack(side=LEFT, padx=20, pady=20)

    def update_value(self, value, name):
        self.setvar(name, f"{float(value):.0f}")

    #设置进度条
    def create_Meter(self,container):
        meter = ttk.Meter(
            master=container,
            metersize=250,
            padding=10,
            amountused=25,
            metertype="semi",
            subtext="%",
            interactive=True,
        )
        meter.pack(side=LEFT,padx=120)
        meter.configure(amountused=0)
        #设置开始训练按钮
        sty = ttk.Style()
        sty.configure('my.TButton',font=('Helvetica', 20))
        button = ttk.Button(container,text='start',style='my.TButton',width=10)
        button.pack(side=LEFT)
        #设置多线程
        import threading
        import time
        def start_process(self):
            self.button.configure(state="disabled")  # 禁用按钮

            # 启动另一个线程来模拟进度更新
            thread = threading.Thread(target=self.run_process)
            thread.start()

        def run_process():
            rawPath = self.filenames[0]
            storePath = self.filenames[1]
            print(rawPath, storePath)
            time.sleep(2)
            # 创建 DatasetProgress 实例
            dataset_progress = DatasetProgress()
            # 调用 create_static_dataset 方法，并传入 storePath 和 rawPath
            dataset = dataset_progress.create_static_dataset(storePath, rawPath)
            for i in range(0,2):
                answer = dataset_progress.get_progress()
                if answer == 100:
                    update_progress(100)  # 更新进度条
                    button.configure(state="normal")  # 恢复按钮状态
                    break
                    
        def update_progress(value):
            meter.step(value)

        def hit():
            meter.configure(amountused=0)
            self.threading_num = 0
            print("start")
            button.configure(state="disabled")
            thread = threading.Thread(target=run_process)
            thread.start()
            # def fun_timer():
            #     global timer
            #     self.threading_num = self.threading_num + 1
            #     if self.threading_num > 100: return
            #     meter.step(1)
            #     timer = threading.Timer(randint(1,50)*0.01, fun_timer)
            #     timer.start()
            # timer = threading.Timer(randint(1,50)*0.01, fun_timer)
            # timer.start()
        button.config(command=hit)

if __name__ == "__main__":
    app = ttk.Window("ImportRawData")
    app.geometry('800x600')
    train(app,[])
    app.mainloop()