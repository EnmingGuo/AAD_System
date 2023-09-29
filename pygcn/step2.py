import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from random import randint
import gspan_function
import step1
import AidedDiagnosis

_flag = True
#_flag = False

class train(ttk.Frame):

    def __init__(self, master,filenames:list):
        master.title('第二步:调整参数并训练')
        master.geometry('800x600')
        self.filenames = list(filenames)
        super().__init__(master, padding=20)
        self.pack(fill=BOTH, expand=YES)

        info = [{'name':'频繁边最小支持度\t','from':0.15,'to':0.25,'resolution':0.005},
                {'name': '频繁子图最小支持度\t', 'from':0.06, 'to':0.09, 'resolution':0.001},
                {'name': '频繁序列最小支持度\t', 'from':3, 'to':8, 'resolution':1}]

        option_text = "Adjust your parameter values"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        for dic in info:
            self.create_band(self, dic)

        container1 = ttk.Frame(self)
        container1.pack(side=TOP, fill=X, padx=10)
        self.create_Meter(container1)
        container = ttk.Frame(self)
        container.pack(side=TOP, fill=X, padx=10, pady=15)
        ttk.Label(container).pack(side=LEFT, padx=70)
        button_front = ttk.Button(container, text='上一步', bootstyle="danger-outline",width=10)
        button_front.pack(side=LEFT, padx=80)
        button_next = ttk.Button(container, text='下一步', bootstyle="danger-outline",width=10)
        button_next.pack(side=LEFT)
        def hit_button_front():
            self.destroy()
            step1.open_train_file(master)
        button_front.configure(command=hit_button_front)
        def hit_button_next():
            self.destroy()
            step3.open_train_file(master,self.filenames)
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
            metersize=180,
            padding=5,
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
        def hit():
            meter.configure(amountused = 0)
            self.threading_num = 0
            # 多线程
            def fun_timer():
                global timer
                self.threading_num = self.threading_num + 1
                if self.threading_num > 100: return
                meter.step(1)
                timer = threading.Timer(randint(1,50)*0.01, fun_timer)
                timer.start()
            timer = threading.Timer(randint(1,50)*0.01, fun_timer)
            timer.start()
            if _flag:
                gspan_function.train(self.getvar('频繁边最小支持度\t'), self.getvar('频繁子图最小支持度\t'), self.getvar('频繁序列最小支持度\t'),
                                 self.filenames)
        button.config(command=hit)

if __name__ == "__main__":
    app = ttk.Window("第二步:调整参数并训练")
    app.geometry('800x600')
    train(app,[])
    app.mainloop()