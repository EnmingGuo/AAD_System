import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def cal(x,y):
    ans=0
    for i in range(0,len(x)-1):
        ans+=(x[i+1]-x[i])*y[i];
    print(ans)
    return ans
def plot_scatter(x, y,x2,y2):
    roc_auc = cal(x,y)
    roc_auc2 =cal(x2,y2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.step(x, y, where='post', color='red', linestyle='--', label='Evolving GraphSage(AUC = %0.2f)' % roc_auc)
    plt.step(x2, y2, where='post', color='blue', linestyle='--', label='Static GraphSage(AUC = %0.2f)' % roc_auc2)
    plt.xlabel('False positive rate (1 - specificity)')
    plt.ylabel('True positive rate (sensitivity)')
    plt.title('ROC Curve')

    plt.legend()
    plt.show()

x = [0.001727116,0.027633851, 0.050086356, 0.070811744, 0.139896373,0.214162349,0.452504318,0.561312608,0.69775475,0.768566494,1]
y = [0.070689655,0.470689655, 0.7637931, 0.835862069, 0.875862069,0.927586207,0.963793103,0.977586207,0.987931034,1.00,1]

x2=[0.001727116,0.04906736,0.065630397,0.110535406,0.134438687,0.284784111,0.465768566,0.666666667,0.813471503,1]
y2=[0.075862069,0.443103448,0.695517241,0.793103448,0.837931034,0.89137931,0.910344828,0.944827586,0.998275862,1]

plot_scatter(x, y,x2,y2)

# 0.001727116	0.070689655
# 0.027633851	0.370689655
# 0.050086356	0.44137931
# 0.070811744	0.775862069
# 0.139896373	0.875862069
# 0.214162349	0.927586207
# 0.452504318	0.963793103
# 0.561312608	0.977586207
# 0.69775475	0.987931034
# 0.768566494	1.003448276


