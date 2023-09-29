import uuid
from tkinter import messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import static_GCN
import static_GAT
import dynamic_GCN
import dynamic_GraphSAGE
import static_GraphSAGE
import torch
from torch_geometric.data import Data
from  feature_capture import *
import AidedDiagnosis
import datetime
import Database
class predict(ttk.Frame):
    def __init__(self, master,data):
        master.title("Case Prediction")
        master.geometry('1000x650')
        super().__init__(master, padding=20)
        self.previous_data = data
        self.database = data.get('database')
        self.resz_var = ttk.StringVar (value='Normal Case Probability: -- %')
        self.resh_var = ttk.StringVar (value='Alzheimer\'s Case Probability: -- %')
        self.pack(fill=BOTH, expand=YES)

        option_text = "Start your prediction process"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_Meter(self.option_lf)

        '''添加部分之最下面的两个按钮'''
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=60)
        button_front = ttk.Button(button_frame, text='Back', width=20)  # 指定父框架，提示文本和宽度
        button_front.pack(side=ttk.LEFT, padx=15)
        button_next = ttk.Button(button_frame, text='Finish', width=20)
        button_next.pack(side=ttk.LEFT, padx=15)
        def hit_button_front():
            submit_data  = self.previous_data;
            self.destroy();
            AidedDiagnosis.AidedDiagnosis(master,data=submit_data)

        button_front.configure(command=hit_button_front)
        def hit_button_next():
            self.quit()
        button_next.configure(command=hit_button_next)

    def getdata(self, path):
        IndexF = []
        IndexT = []
        load_data = sio.loadmat(path)
        matrix = load_data['features']
        for row in range(0, 90):
            for col in range(0, 90):
                if (matrix[row][col] == 1):
                    IndexF.append(row)
                    IndexT.append(col)
        Edge_List = [IndexF, IndexT]
        edge_index = torch.tensor(Edge_List, dtype=torch.int64)
        degree = get_degree(matrix)
        degree_centrality = get_degree_centrality(matrix)
        betweeness_centrality = get_betweeness_centrality(matrix)
        pagerank = get_pagerank(matrix)
        closeness_centrality = get_closeness_centrality(matrix)
        flow_efficencies = get_flow_efficencies(matrix)
        KS = get_KS(matrix)[0]
        local_efficiency = get_local_efficiency(matrix)
        x_list = []
        for x_num in range(0, 90):
            x_temp = []
            x_temp.append(degree[x_num])
            x_temp.append(degree_centrality[x_num])
            x_temp.append(betweeness_centrality[x_num])
            x_temp.append(pagerank[x_num])
            x_temp.append(closeness_centrality[x_num])
            x_temp.append(flow_efficencies[x_num])
            x_temp.append(KS[x_num])
            x_temp.append(local_efficiency[x_num])
            x_list.append(x_temp)
        X = torch.tensor(x_list, dtype=torch.float)
        Y = torch.tensor([1], dtype=torch.long)
        data = Data(x=X, edge_index=edge_index, y=Y)
        return data

    #设置进度条
    def create_Meter(self,master):
        container = ttk.Frame(master)
        container.pack(side=LEFT, padx=5)
        meter = ttk.Meter(
            master=container,
            metersize=180,
            padding=6,
            amountused=25,
            metertype="full",
            subtext="%",
            interactive=True,
            bootstyle='info',
        )
        meter.pack(side=LEFT,padx=120)
        meter.configure(amountused=0)
        #设置预测结果显示
        container2 = ttk.Frame(master)
        container2.pack(side=LEFT, padx=5)
        resz_lbl = ttk.Label(container2, textvariable=self.resz_var,bootstyle="success",font=15)
        resz_lbl.pack(side=TOP, padx=5,pady=15)
        resh_lbl = ttk.Label(container2, textvariable=self.resh_var,bootstyle="danger",font=15)
        resh_lbl.pack(side=TOP,  padx=5,pady=15)
        button = ttk.Button(container2, text='predict',width=15)
        button.pack(pady=30)

        def hit_button():
            button.configure(state="disabled")
            path = self.previous_data.get('lastpath')
            mydata = self.getdata(path)
            model = self.previous_data.get('model')
            result = None
            if model == "sgcn":
                result = static_GCN.test_case(mydata)
            elif model == "sgat" or model == "dgat":
                result = static_GAT.test_case(mydata)
            elif model =="ssage":
                result = static_GraphSAGE.test_case(mydata)
            elif model == 'dgcn':
                result = dynamic_GCN.test_case(mydata)
            elif model == 'dsage':
                result =dynamic_GraphSAGE.test_case(mydata)

            value1 = result[0][0].item()
            value2 = result[0][1].item()
            value1_submit = "{:.1f}%".format(value1 * 100)
            value2_submit = "{:.1f}%".format(value2 * 100)
            unique_id = str(uuid.uuid4())
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
            data = {
                'username': self.previous_data['username'],
                'case':self.previous_data['lastpath'],
                'request_Id': unique_id,
                'selected_model': self.previous_data['model'],
                'datetime': time_string,
                'Normal Case Probability':value2_submit,
                'Alzheimer\'s Case Probability':value1_submit
            }
            result = Database.insert_case(self.database,data)
            if result:
                result2 = messagebox.showinfo(title='Successful Request',
                                             message='The Prediction Result is ready!')
                if result2 == 'ok':
                    value1_percent = round(value1 * 100)
                    value2_percent = round(value2 * 100)
                    # 使用四舍五入后的值进行配置和显示
                    meter.configure(amountused=value2_percent)
                    self.resz_var.set('Normal Case Probability: ' + str(value2_percent) + ' %')
                    self.resh_var.set('Alzheimer\'s Case Probability: ' + str(value1_percent) + ' %')

            else:
                messagebox.showwarning(title='Database Breakdown', message="You failed to insert the case prediction data!")
            # 转化为百分比形式的字符串
            button.configure(state='normal')

        button.configure(command=hit_button)


if __name__ == "__main__":
    app = ttk.Window("第四步:进行测试")
    predict(app,[],[])
    app.mainloop()