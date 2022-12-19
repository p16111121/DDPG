import matlab.engine
import numpy as np

eng=matlab.engine.start_matlab()
#init = [0.5 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]
init = [0.68724342	,0	,-0.016318482	,0	,-0.012160639	,0	,15.12535317	,0	,0.120144841	,0	,-0.086991665	,0]
#init = [0.678035764	,0	,0.000128504	,0	,0.000517704	,0	,11.70890025	,0	,0.011566594	,0	,0.0000879516213244234	,0]
StateVec=[]
PressureSensor=[]
StateVec.append(init)
stern=4
stern_recode=[4,4,4,6,6,4,4,4,4,4,4,4,4,4,4,2,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0]
for i in range(2):
    print("i=",i)
    [newstate,pressure]=eng.step(matlab.double([init]),matlab.double([stern]),matlab.double([i]),nargout=2)
    #print(type(newstate))
    init=newstate
    
    newstate=np.array(newstate)
    print(newstate,newstate.flatten()[10])
    StateVec.append(newstate.flatten().tolist())
    
    pressure=np.array(pressure)
    PressureSensor.append(pressure.flatten().tolist())
    
    stern=stern_recode[i]
"""
import csv

with open("state30sec_test.csv","w",newline="") as f:
    cw=csv.writer(f)
    cw.writerows(StateVec[j] for j in range(len(StateVec)))

with open("PressureSensor30sec_test.csv","w",newline="") as f:
    cw=csv.writer(f)
    cw.writerows(PressureSensor[j] for j in range(len(PressureSensor)))"""
