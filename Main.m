clc; clearvars; close all; rng('default'); warning off all;

nMFs=2;% number of MFs in each input domain
alpha=.1;% learning rate
rr=.05;% regularization coefficient 
nAlgs=2;%numbers of used algorithms
batchSize=64; %size of mini-batch
P=0.5;%drop rate
maxFeatures=8; % maximum dimention of sellected PCA features
nIt=50;%iteration times

datasets={'chenrong','lixiaolong','wangenkai','wanglei','yangbo','yanshihao'};% six subjects dataset

lineStyles={'k-','k--','r-','r--','g-','g--','b-','b--','c-','c--','m-','m--'};

%initialize evaluating indicators
times=cell(length(datasets),1); usedDim=zeros(length(datasets),1);
RCEtrain=times; RCEvalidation=times; RCEtest=times; RCETr=nan(length(datasets),nIt); RCEVa=nan(length(datasets),nIt); RCETe=nan(length(datasets),nAlgs);%RTr=nan(nAlgs,nIt);RTe=RTr;
BCEtrain=times; BCEvalidation=times; BCEtest=times; BCETr=nan(length(datasets),nIt); BCEVa=nan(length(datasets),nIt); BCETe=nan(length(datasets),nAlgs);%BTr=nan(nAlgs,nIt);BTe=BTr;
BT=cell(1,nAlgs);CT=BT;SigmaT=BT;

data=[];label=[];
for s=1:length(datasets)
    temp=load(['dataset\' datasets{s} '.mat']);
    truedata=temp.truedata;
    falsedata=temp.falsedata;
    X0{s}=[truedata;falsedata];y0{s}=[ones(length(truedata),1);zeros(length(falsedata),1)];
    data=[data;X0{s}]; label=[label;y0{s}];
end

for s=1:length(datasets)%Each person's data is used as a separate testing set to evaluate classifiers trained with other subjects combined data
    XTest=X0{s};yTest=y0{s};
    XTr=[];yTr=[];
    for i=1:length(datasets)
        if i~=s
            XTr=[XTr;X0{i}];
            yTr=[yTr;y0{i}];
        end
    end
    %z-score normlization
    Me=mean(XTr); St=std(XTr); XTr=(XTr-repmat(Me,size(XTr,1),1))./repmat(St,size(XTr,1),1); [N0,M] = size(XTr);
    %PCA dimention reduction
    if M>maxFeatures
        [COEFF,XPCA,latent,~,EXPLAINED,MU]=pca(XTr);
        realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
        usedDim(s)=min(maxFeatures,realDim98); PCAM=COEFF(:,1:usedDim(s));
        XTr=XPCA(:,1:usedDim(s)); [N0,M]=size(XTr);
    end
    XTest=((XTest-Me)./St-MU)*PCAM;
    
    %% trianing & testing
    N=round(N0*.6);%the ratio of trianing set to the validation set is 60%/40%
    times{s}=nan(nAlgs,1);
    RCEtrain{s}=nan(nAlgs,nIt); RCEvalidation{s}=nan(nAlgs,nIt); RCEtest{s}=nan(nAlgs,1);
    BCEtrain{s}=RCEtrain{s}; BCEvalidation{s}=RCEvalidation{s}; BCEtest{s}=RCEtest{s};
    
    idsTrain=datasample(1:N0,N,'replace',false);
    XTrain=XTr(idsTrain,:); yTrain=yTr(idsTrain); 
    XValidation=XTr; XValidation(idsTrain,:)=[]; yValidation=yTr; yValidation(idsTrain,:)=[];
    %% 1: our BGD-R
    tic;     
    [RCEtrain{s}(1,:),RCEvalidation{s}(1,:),RCEtest{s}(1),BCEtrain{s}(1,:),BCEvalidation{s}(1,:),BCEtest{s}(1),BT{1},CT{1},SigmaT{1}]=...
        BGD_R(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,nMFs,nIt,N);%core algorithm of ANFIS
    times{s}(1)=toc;%计时

    %% 2: our MBGD-RDA  
    tic;     
    [RCEtrain{s}(2,:),RCEvalidation{s}(2,:),RCEtest{s}(2),BCEtrain{s}(2,:),BCEvalidation{s}(2,:),BCEtest{s}(2),BT{2},CT{2},SigmaT{2}]=...
        MBGD_RDA(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,P,nMFs,nIt,batchSize);%core of proposed algorithm
    times{s}(2)=toc;%计时

end
%% computing RCA and BCA
for i=1:nAlgs
    RCE=0;
    BCE=0;
    %RCA
    time_total=0;
    for s=1:length(datasets) 
        RTe=RCEtest{s}(i);
        time_total=time_total+sum(times{s}(i));
        RCETe(s,i)=RTe;
    end

    %BCA
    for s=1:length(datasets)
        BTe=BCEtest{s}(i);
        time_total=time_total+sum(times{s}(i));
        BCETe(s,i)=BTe;
    end
    switch i
        case 1
            disp(['Average_RCE of BGD_R:' num2str(mean(RCETe(:,i))) '+-' num2str(std(RCETe(:,i)))]);
            disp(['Average_BCE of BGD_R:' num2str(mean(BCETe(:,i))) '+-' num2str(std(BCETe(:,i)))]);
        case 2
            disp(['Average_RCE of MBGD_RDA:' num2str(mean(RCETe(:,i))) '+-' num2str(std(RCETe(:,i)))]);
            disp(['Average_BCE of MBGD_RDA:' num2str(mean(BCETe(:,i))) '+-' num2str(std(BCETe(:,i)))]);
    end
    disp(['Average_time:' num2str(time_total/length(datasets))]);
end
% save('parameters.mat','BT','CT','SigmaT','PCAMTX','Mean','Std','MU');