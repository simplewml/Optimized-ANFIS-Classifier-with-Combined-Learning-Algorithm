function [RCEtrain,RCEtest,BCEtrain,BCEtest,BT,CT,SigmaT]=BGD_R(XTrain,yTrain,XTest,yTest,alpha,rr,numMFs,numIt,batchSize)
%XTrain、yTrain训练数据集；XTest、yTest测试数据集；alpha学习率；numMFs隶属函数个数；numIt训练次数；batchSize训练数据量

% GD for FLS training
% alpha: learning rate
% numMFs: number of MFs in each input domain
% numIt: maximum number of iterations

[N,M]=size(XTrain); NTest=size(XTest,1);
numMFsVec=numMFs*ones(M,1);
R=numMFs^M; % number of rules
C=zeros(M,numMFs); Sigma0=C; B=zeros(R,M+1);%C、Sigma每个隶属函数的mu、sigma，前向参数；B：x的系数和常数项，后向参数
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),numMFs);%C初值
    Sigma0(m,:)=std(XTrain(:,m));%每一列的标准差
end
Sigma=Sigma0;

%% Iterative update
mu=zeros(M,numMFs); CEtrain=ones(N,1);
yPred=nan(batchSize,1); ycPred=yPred;
numDown=0; numUpDown=0;
for it=1:numIt %numIt次训练
    deltaC=zeros(M,numMFs); deltaSigma=deltaC;    deltaB=rr*B; deltaB(:,1)=0; % consequent
    f=ones(batchSize,R); % firing level of rules
    idsTrain=datasample(1:N,batchSize,'replace',false);%训练集的训练数据index
    for n=1:batchSize%一次训练使用的数据量
        for m=1:M % layer 1: membership grades of MFs 
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));%高斯隶属函数模糊化结果
        end
        for r=1:R
            idsMFs=idx2vec(r,numMFsVec);
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));% layer 2: firing strength w 
            end
        end
        fBar=f(n,:)/sum(f(n,:));% layer 3: normalize w 
        yR=[1 XTrain(idsTrain(n),:)]*B';% layer 4: compute result of rules 
        yPred(n)=fBar*yR'; % layer 5: sum the whole rules
       
         % Compute delta sigmoid
        ycPred(n)=1./(1+exp(-yPred(n))); %layer 6: sigmoid prediction
        for r=1:R%对于每个规则
            temp=(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);%第n组样本训练的误差*（规则r的结果-yPred)*规则r占的触发强度比重
            if ~isnan(temp) && abs(temp)<inf%没有数据缺失&&触发强度之和不为0
                vec=idx2vec(r,numMFsVec);
                % delta of c, sigma, and b
                for m=1:M%输入特征维度
                    deltaC(m,vec(m))=deltaC(m,vec(m))-temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))-temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaB(r,m+1)=deltaB(r,m+1)-(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*fBar(r)*XTrain(idsTrain(n),m);%deltaB：2:m+1为第n组样本训练的误差*规则r的触发强度比重*训练数据
                end
                % delta of br0
                deltaB(r,1)=deltaB(r,1)-(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*fBar(r);%deltaB：1为第n组样本训练的误差*规则r的触发强度比重
            end
        end
    end
    
     % Training error sigmoid
    error0=0;
    error1=0;
    yyPred=round(ycPred);
    for i=1:length(yyPred)
        if(yTrain(idsTrain(i))==1&&yyPred(i)~=1)
            error1=error1+1;
        elseif (yTrain(idsTrain(i))==0&&yyPred(i)~=0)
            error0=error0+1;
        end
    end
    CEtrain(it)=1-sum((yyPred-yTrain(idsTrain))==0)/length(yPred);
    RCEtrain(it)=1-sum((yyPred-yTrain(idsTrain))==0)/length(yPred);
    BCEtrain(it)=1/2*(error1/sum(yTrain(idsTrain)==1)+error0/sum(yTrain(idsTrain)==0));
    
    % Test error
    f=ones(NTest,R); % firing level of rules
    for n=1:NTest
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        
        for r=1:R % firing levels of rules
            idsMFs=idx2vec(r,numMFsVec);
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));
            end
        end
    end
    yR=[ones(NTest,1) XTest]*B';
    yPredTest=sum(f.*yR,2)./sum(f,2); 
    ycPredTest=1./(1+exp(-yPredTest)); % prediction
    %compute error
    error0=0;
    error1=0;
    yyyPred=round(ycPredTest);
    for i=1:length(ycPredTest)
        if(yTest(i)==1&&yyyPred(i)~=1)
            error1=error1+1;
        elseif (yTest(i)==0&&yyyPred(i)~=0)
            error0=error0+1;
        end
    end
    RCEtest(it)=1-sum((yyyPred-yTest)==0)/length(yPredTest);
    BCEtest(it)=1/2*(error1/sum(yTest==1)+error0/sum(yTest==0));
    
    % Adaptive learning rate
    if it>1 && CEtrain(it)<CEtrain(it-1)%如果这次的RMSE<上次RMSE，记为连续下降次数+1，否则，连续下降次数置0
        numDown=numDown+1;
    else
        numDown=0;
    end
    if numDown==4%如果连续下降次数达到4，增加学习率，连续下降次数置0
        alpha=1.1*alpha; numDown=0;
    end
    if it>2 && ~mod(it,2) % evaluate every two iterations
        if CEtrain(it)<CEtrain(it-1) && CEtrain(it-1)>CEtrain(it-2)
            numUpDown=numUpDown+1;
        else
            numUpDown=0;
        end
    end
    if numUpDown==2
        alpha=.9*alpha;  numUpDown=0;
    end
    
    % update C, Sigma and b
    den=sqrt(sum(deltaC(:).^2)+sum(deltaSigma(:).^2)+sum(deltaB(:).^2));
    C=C-alpha*deltaC/den;
    Sigma=max(.1*Sigma0,Sigma-alpha*deltaSigma/den);
    B=B-alpha*deltaB/den;
end

BT=B;
CT=C;
SigmaT=Sigma;
