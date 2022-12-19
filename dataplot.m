clc,clear,close all;

PressureSensor=readtable("./test/20221102132521/PressureSensor10.csv");
pressure=table2array(PressureSensor);

originpresurre=xlsread("matlab對照data.xlsx",1,'O1:O60');
oristern = xlsread("matlab對照data.xlsx",1,"Q1:Q61");

stern=readtable("./test\20221102132521\stern10.csv");
stern=table2array(stern);



figure(1);
plot(-1*pressure,'LineWidth',1.5);
axis([0,200,-75,5]); 
xticks([0:25:200]);

xlabel({'\bf  \it Time [s]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
ylabel({'\bf  \it depth [m]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
set(gcf,'position',[100 100 1400 500 ]);   % 100 50 380 750， 100和50是螢幕和出圖的位置間的距離 ； 出圖大小，280:圖寬，400圖高
set(gca,'Position',[.10 .30 .80 .40]);   % .a .b .c .d， a左、b下 c右 d上

figure(2);
plot(-1*originpresurre,'LineWidth',1.5);
axis([0,200,-2,0.2]); 
yticks([-1.2:0.2:0.2]);
xlabel({'\bf  \it Time [s]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
ylabel({'\bf  \it depth [m]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
set(gcf,'position',[100 100 1400 500 ]);   % 100 50 380 750， 100和50是螢幕和出圖的位置間的距離 ； 出圖大小，280:圖寬，400圖高
set(gca,'Position',[.10 .30 .80 .40]);   % .a .b .c .d， a左、b下 c右 d上

figure(3);
plot(oristern);
xlabel({'\bf  \it Time [s]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
ylabel({'\bf  \it stern angle{\theta }'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
set(gcf,'position',[100 100 1400 500 ]);   % 100 50 380 750， 100和50是螢幕和出圖的位置間的距離 ； 出圖大小，280:圖寬，400圖高
set(gca,'Position',[.10 .30 .80 .40]);

figure(4);
plot(stern);
xlabel({'\bf  \it Time [s]'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
ylabel({'\bf  \it stern angle{\theta }'},'fontsize',24,'FontName','Times New Roman','HorizontalAlignment','center');
axis([0,200,-40,40]); 

set(gcf,'position',[100 100 1400 500 ]);   % 100 50 380 750， 100和50是螢幕和出圖的位置間的距離 ； 出圖大小，280:圖寬，400圖高
set(gca,'Position',[.10 .30 .80 .40]);