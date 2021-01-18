%% Compare our algorithms with Matlab's ANFIS on two small datasets
clc; clearvars; close all; rng('default'); warning off all;

nMFs=2;% number of MFs in each input domain
alpha=.01;% learning rate
rr=.05;
nAlgs=2;%ʹ�õ��㷨����
batchSize=64;
P=0.55;
maxFeatures=8; % maximum number of features to use��������ά��
nIt=100;%ѵ������
nRepeats=5;%�ظ�ѵ������
datasets={'data'};%.mat���ݾ��� 

lineStyles={'k-','k--','r-','r--','g-','g--','b-','b--','c-','c--','m-','m--'};

times=cell(nAlgs,length(datasets)); usedDim=zeros(nAlgs,length(datasets));%��ʼ��ѵ���������Լ����������ѵ��ʱ�䡢ʵ��ʹ��ά��
RCEtrain=times; RCEtest=times;RCETr=nan(length(datasets),nIt); RCETe=nan(length(datasets),nIt);RTr=nan(nAlgs,nIt);RTe=RTr;
BCEtrain=times; BCEtest=times;BCETr=nan(length(datasets),nIt); BCETe=nan(length(datasets),nIt);BTr=nan(nAlgs,nIt);BTe=BTr;
SS=times; stdSS=SS;%���ƽ���͡�
BT=cell(1,nAlgs);CT=BT;SigmaT=BT;

for s=1:length(datasets)%��ȡÿһ��.mat����dw
    temp=load(['dataset\' datasets{s} '.mat']);
    truedata=temp.truedata;
    falsedata=temp.falsedata;
    X0=[truedata;falsedata];y0=[ones(length(truedata),1);zeros(length(falsedata),1)];% y_m=mean(y0); y0=y0-y_m;%X0��ȡ��ȥ���һ�����ݵ����ݣ�y0��ȡ���һ�����ݣ����ݶ�Ӧ�Ľ��
%     X0=X0(:,1:end);%using 2fsr+5IMU
%     X0=X0(:,1:end-6);%using 2fsr+3IMU
%     X0=X0(:,[1,2,6:end-6]);%using 2fsr+2IMU
%     X0=X0(:,1:end-12);%using 2fsr+1IMU
    
    %% ����Ԥ����
    %����z��׼������
    X0 = zscore(X0); [N0,M]=size(X0);
    
    %PCA��ά
    if M>maxFeatures%�����ά�ȴ��������������ά�ȣ���PCA��ά
        [~,XPCA,latent]=pca(X0);%XPCA��ԭʼ�����������ɵ����ɷֿռ��������ֵ��latent��X0Э�����������ֵ��XPCAÿһ�еķ���Ӵ�С����
        realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');%ʵ��ʹ�������������ά��
        usedDim(s)=min(maxFeatures,realDim98);
        X0=XPCA(:,1:usedDim(s)); [N0,M]=size(X0);%ʵ��ʹ�õ����ݼ�
    end
     %% ��������ɢ��ͼ
% %     ɢ��ͼ
%     figure;
%     scatter3(X0(:,1),X0(:,2),X0(:,3),3,y0);
%     title(datasets{s});
    %% ѵ��&����
    N=round(N0*.6);%ÿһ��ѵ���õ���������Ϊ�����60%
    times{s}=nan(nAlgs,nRepeats);%������ָ��nan����������*ѵ������*�ظ�ѵ������
    RCEtrain{s}=nan(nAlgs,nIt,nRepeats); RCEtest{s}=nan(nAlgs,nIt,nRepeats);
    for r=1:nRepeats
        idsTrain=datasample(1:N0,N,'replace',false);%�����ȡ1~N0�в�ͬ��N����
        XTrain=X0(idsTrain,:); yTrain=y0(idsTrain); %ѵ����Ϊ�����ȡ��index��Ӧ�ĸ���
        XTest=X0; XTest(idsTrain,:)=[]; yTest=y0; yTest(idsTrain)=[]; %���Լ�Ϊʣ�µ����и���

        %% 1: Matlab's ANFIS, GD
        tic;     
        [RCEtrain{s}(1,:,r),RCEtest{s}(1,:,r),BCEtrain{s}(1,:,r),BCEtest{s}(1,:,r),BT{1},CT{1},SigmaT{1}]=...
            BGD_R_ORG(XTrain,yTrain,XTest,yTest,alpha,rr,nMFs,nIt,N);%����
        times{s}(1,r)=toc;%��ʱ

        %% 2: our GD-R  %���� alpha ָ���� nMFs���õ�
        tic;     
        [RCEtrain{s}(2,:,r),RCEtest{s}(2,:,r),BCEtrain{s}(2,:,r),BCEtest{s}(2,:,r),BT{2},CT{2},SigmaT{2}]=...
            MBGD_RDA_ORG(XTrain,yTrain,XTest,yTest,alpha,rr,P,nMFs,nIt,batchSize);%����
        times{s}(2,r)=toc;%��ʱ
        
    end  
end
%% ����ÿ�����ݼ����������Ƶ���ѵ����RCA�仯����
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