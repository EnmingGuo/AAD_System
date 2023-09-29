import ttkbootstrap as ttk
from ttkbootstrap.constants import *

app = ttk.Window()

meter = ttk.Meter(
    metersize=180,
    padding=6,
    amountused=25,
    metertype="full",
    subtext="miles per hour",
    interactive=True,
    bootstyle='info',
)
meter.pack()

# update the amount used directly
meter.configure(amountused = 50)

# update the amount used with another widget
entry = ttk.Entry(textvariable=meter.amountusedvar)
entry.pack(fill=X)

# increment the amount by 10 steps
meter.step(10)

# decrement the amount by 15 steps
meter.step(-15)

# update the subtext
meter.configure(subtext="loading...")

app.mainloop()
