import uuid
import datetime
from tkinter import messagebox

import Database
import ttkbootstrap as ttk
import AidedDiagnosis
from ttkbootstrap.constants import *
class MessageBoard(ttk.Frame):
    def __init__(self, master, data=None):
        master.geometry("1000x650")
        master.title("Message")
        super().__init__(master)
        # Create the ttkbootstrap style
        self.pack(fill=BOTH, expand=YES)
        self.previous_data  = data
        # Create the main frame
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(main_frame, text="Message", style="Title.TLabel", font=("Comic Sans MS", 14, "bold"))
        title_label.grid(row=0, column=0, pady=20)

        # function
        def submit_message():
            # TODO: Implement the logic for submitting the message
            text_content = message_entry.get("1.0", "end-1c")
            text_size = len(text_content)

            if(text_content=="" or text_size <=10):
                messagebox.showwarning(title='Message error', message="The message should contain at least 10 characters.")
                return;

            unique_id = str(uuid.uuid4())
            username = self.previous_data['username']
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
            data = {
                'username': username,
                'message_id':unique_id,
                'datetime':time_string,
                'content':text_content
            }
            if(Database.insert_message(self.previous_data['database'],data) == True):
                messagebox.showinfo(title='Successful Request',message='Leaving message successed!')
            else:
                messagebox.showwarning(title='Request error', message="Leaving message failed!")
                return;


        def go_back():
            # TODO: Implement the logic for going back
            submit_data = self.previous_data
            self.destroy()
            AidedDiagnosis.AidedDiagnosis(master,submit_data)
        # Create the message entry
        message_entry = ttk.Text(main_frame, width=100, height=20, font=("Arial", 12))
        message_entry.grid(row=1, column=0, pady=20)
        message_entry.focus_set()

        # Create a frame for the buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=20)

        # Create the submit button
        submit_button = ttk.Button(button_frame, text="Submit", style="Primary.TButton", command=submit_message, width=20)
        submit_button.pack(side="left", padx=10)

        # Create the back button
        back_button = ttk.Button(button_frame, text="Back", style="Secondary.TButton", command=go_back, width=20)
        back_button.pack(side="left", padx=10)

        # Center the main frame horizontally
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Center the main frame vertically
        main_frame.pack_propagate(False)
        main_frame.place(relx=0.5, rely=0.5, anchor="center")





if __name__ == "__main__":
    my_w = ttk.Window('Message')
    message_board = MessageBoard(my_w, [])
    my_w.mainloop()
