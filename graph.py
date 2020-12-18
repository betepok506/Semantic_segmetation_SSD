from matplotlib import pyplot as plt
import numpy as np


if __name__=='__main__':
    file='save/t.txt'
    epoch=[]
    val_loss=[]
    loss_val=[]
    with open(file,'r') as f:
        for st in f:
            tt=f.readline()[:-1].split(',')
            try:
                epoch.append((int(tt[0])+1)*1000)
                loss_val.append(float(tt[1]))
                val_loss.append(float(tt[2]))
            except:
                pass
    m = max((1 + 1) // 2, 2)
    n = 2

    fig, cells = plt.subplots()
    cells.plot(epoch, loss_val, label='Training loss',color='blue', linewidth=1.0)
    cells.plot(epoch, val_loss, label='Validation loss',color='red', linewidth=1.0)
    cells.legend()
    cells.set_xlabel('Training steps', fontsize=14)
    cells.set_ylabel('Loss', fontsize=14)
    cells.grid(True)
    cells.set_xticks(np.linspace(0, 60*1000,7))
    cells.set_yticks(np.linspace(0,int(max(val_loss))+2,7))
    plt.show()