function [RCEtrain,RCEtest,BCEtrain,BCEtest,BT,CT,SigmaT]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,rr,P,numMFs,numIt,batchSize)
%XTrain��yTrainѵ�����ݼ���XTest��yTest�������ݼ���alphaѧϰ�ʣ�numMFs��������������numItѵ��������batchSizeѵ��������

% GD for FLS training
% alpha: learning rate
% numMFs: number of MFs in each input domain
% numIt: maximum number of iterations

beta1=0.9; beta2=0.999;

[N,M]=size(XTrain); NTest=size(XTest,1);
numMFsVec=numMFs*ones(M,1);
R=numMFs^M; % number of rules
C=zeros(M,numMFs); Sigma0=C; B=zeros(R,M+1);%C��Sigmaÿ������������mu��sigma��ǰ�������B��x��ϵ���ͳ�����������
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),numMFs);%C��ֵ
    Sigma0(m,:)=std(XTrain(:,m));%ÿһ�еı�׼��
end
Sigma=Sigma0;

%% Iterative update
mu=zeros(M,numMFs);
mStepSize=zeros(1,numIt); stdStepSize=mStepSize;
mC=0; vC=0; mB=0; mSigma=0; vSigma=0; vB=0; yPred=nan(batchSize,1); ycPred=yPred;

for it=1:numIt %numIt��ѵ��
    deltaC=zeros(M,numMFs); deltaSigma=deltaC;    deltaB=rr*B; deltaB(:,1)=0; % consequent
    f=ones(batchSize,R); % firing level of rules
    idsTrain=datasample(1:N,batchSize,'replace',false);%ѵ������ѵ������index
    for n=1:batchSize%һ��ѵ��ʹ�õ�������
        for m=1:M % membership grades of MFs ��1
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));%��˹��������ģ�������
        end
        % droprule
        for r=1:R
            if rand<P
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            else
                f(n,r)=0;
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            f(n,:)=1;
            for r=1:R
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        fBar=f(n,:)/sum(f(n,:));%�õ���һ������ǿ��w ��3
        yR=[1 XTrain(idsTrain(n),:)]*B';%����ÿ������Ľ������y ��4
        yPred(n)=fBar*yR'; % predictionÿ������Ľ����Ȩ�����Ϊһ��������Ԥ��ֵ ��5
       
         % Compute delta sigmoid
        ycPred(n)=1./(1+exp(-yPred(n))); %sigmoid prediction
        for r=1:R%����ÿ������
            temp=(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);%��n������ѵ�������*������r�Ľ��-yPred)*����rռ�Ĵ���ǿ�ȱ���
            if ~isnan(temp) && abs(temp)<inf%û������ȱʧ&&����ǿ��֮�Ͳ�Ϊ0
                vec=idx2vec(r,numMFsVec);
                % delta of c, sigma, and b
                for m=1:M%��������ά��
                    deltaC(m,vec(m))=deltaC(m,vec(m))-temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))-temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaB(r,m+1)=deltaB(r,m+1)-(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*fBar(r)*XTrain(idsTrain(n),m);%deltaB��2:m+1Ϊ��n������ѵ�������*����r�Ĵ���ǿ�ȱ���*ѵ������
                end
                % delta of br0
                deltaB(r,1)=deltaB(r,1)-(yTrain(idsTrain(n))/ycPred(n)-(1-yTrain(idsTrain(n)))/(1-ycPred(n)))*(ycPred(n)*(1-ycPred(n)))*fBar(r);%deltaB��1Ϊ��n������ѵ�������*����r�Ĵ���ǿ�ȱ���
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
    % Adam
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    mB=beta1*mB+(1-beta1)*deltaB;
    vB=beta2*vB+(1-beta2)*deltaB.^2;
    mBHat=mB/(1-beta1^it);
    vBHat=vB/(1-beta2^it);
    % update C, Sigma and B, using AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(.1*Sigma0,Sigma-lrSigma.*mSigmaHat);
    lrB=min(ub,max(lb,alpha./(sqrt(vBHat)+10^(-8))));
    B=B-lrB.*mBHat;
    lr=[lrC(:); lrSigma(:); lrB(:)];
    mStepSize(it)=mean(lr); stdStepSize(it)=std(lr);
    
end

BT=B;
CT=C;
SigmaT=Sigma;
