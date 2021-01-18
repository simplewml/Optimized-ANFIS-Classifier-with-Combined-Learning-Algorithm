%% Compare our algorithms with Matlab's ANFIS on two small datasets
clc; clearvars; close all; rng('default'); warning off all;

nMFs=2;% number of MFs in each input domain
alpha=.01;% learning rate
rr=.05;
nAlgs=2;%使用的算法个数
batchSize=64;
P=0.55;
maxFeatures=8; % maximum number of features to use输入特征维度
nIt=100;%训练次数
nRepeats=5;%重复训练次数
datasets={'data'};%.mat数据矩阵 

lineStyles={'k-','k--','r-','r--','g-','g--','b-','b--','c-','c--','m-','m--'};

times=cell(nAlgs,length(datasets)); usedDim=zeros(nAlgs,length(datasets));%初始化训练集、测试集均方差矩阵、训练时间、实际使用维度
RCEtrain=times; RCEtest=times;RCETr=nan(length(datasets),nIt); RCETe=nan(length(datasets),nIt);RTr=nan(nAlgs,nIt);RTe=RTr;
BCEtrain=times; BCEtest=times;BCETr=nan(length(datasets),nIt); BCETe=nan(length(datasets),nIt);BTr=nan(nAlgs,nIt);BTe=BTr;
SS=times; stdSS=SS;%离差平方和、
BT=cell(1,nAlgs);CT=BT;SigmaT=BT;

for s=1:length(datasets)%读取每一个.mat矩阵dw
    temp=load(['dataset\' datasets{s} '.mat']);
    truedata=temp.truedata;
    falsedata=temp.falsedata;
    X0=[truedata;falsedata];y0=[ones(length(truedata),1);zeros(length(falsedata),1)];% y_m=mean(y0); y0=y0-y_m;%X0提取除去最后一列数据的数据，y0提取最后一列数据：数据对应的结果
%     X0=X0(:,1:end);%using 2fsr+5IMU
%     X0=X0(:,1:end-6);%using 2fsr+3IMU
%     X0=X0(:,[1,2,6:end-6]);%using 2fsr+2IMU
%     X0=X0(:,1:end-12);%using 2fsr+1IMU
    
    %% 数据预处理
    %输入z标准化处理
    X0 = zscore(X0); [N0,M]=size(X0);
    
    %PCA降维
    if M>maxFeatures%输入的维度大于最大输入特征维度，用PCA降维
        [~,XPCA,latent]=pca(X0);%XPCA：原始数据在新生成的主成分空间里的坐标值，latent：X0协方差矩阵特征值（XPCA每一列的方差）从大到小排列
        realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');%实际使用特征最大数据维度
        usedDim(s)=min(maxFeatures,realDim98);
        X0=XPCA(:,1:usedDim(s)); [N0,M]=size(X0);%实际使用的数据集
    end
     %% 手势数据散点图
% %     散点图
%     figure;
%     scatter3(X0(:,1),X0(:,2),X0(:,3),3,y0);
%     title(datasets{s});
    %% 训练&测试
    N=round(N0*.6);%每一次训练用到的数据量为总体的60%
    times{s}=nan(nAlgs,nRepeats);%对评价指标nan处理：方法数*训练次数*重复训练次数
    RCEtrain{s}=nan(nAlgs,nIt,nRepeats); RCEtest{s}=nan(nAlgs,nIt,nRepeats);
    for r=1:nRepeats
        idsTrain=datasample(1:N0,N,'replace',false);%随机抽取1~N0中不同的N个数
        XTrain=X0(idsTrain,:); yTrain=y0(idsTrain); %训练集为随机抽取的index对应的个体
        XTest=X0; XTest(idsTrain,:)=[]; yTest=y0; yTest(idsTrain)=[]; %测试集为剩下的所有个体

        %% 1: Matlab's ANFIS, GD
        tic;     
        [RCEtrain{s}(1,:,r),RCEtest{s}(1,:,r),BCEtrain{s}(1,:,r),BCEtest{s}(1,:,r),BT{1},CT{1},SigmaT{1}]=...
            BGD_R_ORG(XTrain,yTrain,XTest,yTest,alpha,rr,nMFs,nIt,N);%核心
        times{s}(1,r)=toc;%计时

        %% 2: our GD-R  %核心 alpha 指数调 nMFs不用调
        tic;     
        [RCEtrain{s}(2,:,r),RCEtest{s}(2,:,r),BCEtrain{s}(2,:,r),BCEtest{s}(2,:,r),BT{2},CT{2},SigmaT{2}]=...
            MBGD_RDA_ORG(XTrain,yTrain,XTest,yTest,alpha,rr,P,nMFs,nIt,batchSize);%核心
        times{s}(2,r)=toc;%计时
        
    end  
end
%% 绘制每个数据集上六种手势单独训练的RCA变化曲线
for i=1:nAlgs
    RCE=0;
    BCE=0;
    figure;%RCA
    set(gcf, 'Position', 1/3*get(0, 'Screensize'));hold on;
    time_total=0;
    for s=1:length(datasets) 
        RTr=nanmean(RCEtrain{s}(i,:,:),3);
        RTe=nanmean(RCEtest{s}(i,:,:),3);
        time_total=time_total+sum(times{s}(i,:));
        RCETr(s,:)=RTr;
        RCETe(s,:)=RTe;
        RCE=RCE+RCETe(s,end);
    end
    for s=1:length(datasets)
        plot(RCETr(s,:),lineStyles{2*s-1},'linewidth',2);
        plot(RCETe(s,:),lineStyles{2*s},'linewidth',2);
    end
    if i==1
        set(gca,'yscale','log'); xlabel('Iteration'); ylabel('ANFIS BGD-R RCE');
    else
        set(gca,'yscale','log'); xlabel('Iteration'); ylabel('ANFIS MBGD-RDA RCE');
    end
    legend('TrainRCE','TestRCE','location','northeast');    

    figure;%BCA
    set(gcf, 'Position', 1/3*get(0, 'Screensize'));hold on;
    for s=1:length(datasets)
        BTr=nanmean(BCEtrain{s}(i,:,:),3);
        BTe=nanmean(BCEtest{s}(i,:,:),3);
        time_total=time_total+sum(times{s}(i,:));
        BCETr(s,:)=BTr;
        BCETe(s,:)=BTe;
        BCE=BCE+BCETe(s,end);
    end
    for s=1:length(datasets)
        plot(smooth(BCETr(s,:),10),lineStyles{2*s-1},'linewidth',2);
        plot(smooth(BCETe(s,:),10),lineStyles{2*s},'linewidth',2);
    end
    if i==1
        set(gca,'yscale','log'); xlabel('Iteration'); ylabel('ANFIS BGD-R BCE');
    else
        set(gca,'yscale','log'); xlabel('Iteration'); ylabel('ANFIS MBGD-RDA BCE');
    end
    legend('TrainBCE ','TestBCE ','location','northeast');  
    
    disp(['Average_RCE:' num2str(RCE/length(datasets))]);
    disp(['Average_BCE:' num2str(BCE/length(datasets))]);
    disp(['Average_time:' num2str(time_total/length(datasets))]);
end
save('parameters.mat','BT','CT','SigmaT');
% save(['OAACE' num2str(alpha) '.mat'],'RCETr','RCETe','RTr','RTe','RCEtrain','RCEtest','BCETr','BCETe','BTr','BTe','BCEtrain','BCEtest');