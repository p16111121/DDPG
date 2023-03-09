function [newstate,PressureSensor]=step(oldstate,stern_angle,times)
warning('off')
global Mmat Props X Y Z K M N inputs;
global var1;

var1 = 1;

save_msg = 0; test_msg = 'PositiveB_1en5pcnt'; case_msg = 'RollStab';

props = 0; delR = 0; delS = 0;
Xprops =0.8; % 推力
inputs = [Xprops;delR;delS];

[Mmat,Props,X,Y,Z,K,M,N] = RemusAUV1();

told1 = times;
told = 0;%似乎和t無關
%tend = 200;
% tspan = told:1:tend; %可隱藏

%init = [0.5 0 0 0 0 0 0 0 0 0 0 0];
init=oldstate;
% init = [u,v,w,p,q,r,x,y,z,phi,theta,psi]; 給予船初始直
%StateVec = [init];
%TVec = told;

%舵板角度初始值
rud=0; % 垂直舵板 控制左右
%stern=0 %水平舵板 控制上下
stern=stern_angle;
%for i=1:3
% input 為舵板角度的紀錄
input(told+1,1)= Xprops;
input(told+1,2)= 0*pi/180;%垂直舵板保持0
input(told+1,3)= stern*pi/180;%將水平舵板角度換算成徑度

%while (told < i)
tspan = told:0.5:told+1;%每秒之間積分間隔 [189,189.5,190]

inputs=input(told+1,:);%最新的舵板角度
%tspan 積分間隔 init 初始條件
%ode45解 y` 在 tspan的積分
[t,states] = ode45('odefunc',tspan,init); % Each row in the solution array y corresponds to a value returned in column vector t.
u = states(end,1);%縱移速度
v = states(end,2);%橫移速度
w = states(end,3);%起伏速度
p = states(end,4);%橫搖角速度
q = states(end,5);%縱搖角速度
r = states(end,6);%平擺角速度
x = states(end,7);%縱向位移
y = states(end,8);%橫向位移
z = states(end,9);%起伏位移
phi = states(end,10);%roll angle
theta = states(end,11);%pitch angle
psi = states(end,12);%yaw angle
%told = t(end);
init = [u,v,w,p,q,r,x,y,z,phi,theta,psi];
%told=times;
%StateVec = vertcat(StateVec,init);% 顯示(累積)每一時間之init之狀況
    %TVec = vertcat(TVec,told);%時間陣列

%PressureSensor(i,1) = StateVec(i+1,9)-0.85*sin(StateVec(end,11))%深度值
    
    
%end
PressureSensor=init(9)-0.85*sin(init(11));
newstate=init;
end