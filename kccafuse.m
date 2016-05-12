clear all;clc;
run('kmbox\install')
%%
%train feature for all views
load('features\ftrain0.mat');
load('features\ftrain15.mat');
load('features\ftrain30.mat');
load('features\ftrain45.mat');
load('features\ftrainn15.mat');
load('features\ftrainn30.mat');
load('features\ftrainn45.mat');

%%
%test features for all views
load('features\ftest0.mat');
load('features\ftest15.mat');
load('features\ftest30.mat');
load('features\ftest45.mat');
load('features\ftestn15.mat');
load('features\ftestn30.mat');
load('features\ftestn45.mat');

%%
%Actual train and test label
load('features\trainY.mat');
load('features\testY.mat');


train0x = ftrain0; 
train15x = ftrain15;
train30x = ftrain30;
train45x = ftrain45; 
trainn15x = ftrainn15; 
trainn30x = ftrainn30; 
trainn45x = ftrainn45; 

testy = testY;
trainy = trainY;

test0x = ftest0;
test15x = ftest15;
test30x = ftest30;
test45x = ftest45;
testn15x = ftestn15;
testn30x = ftestn30;
testn45x = ftestn45;

%%
%predicted values from different channel
ypred0 = k_cca(train0x,trainy,test0x,testy);
ypred15 = k_cca(train15x,trainy,test15x,testy);
ypred30 = k_cca(train30x,trainy,test30x,testy);
ypred45 = k_cca(train45x,trainy,test45x,testy);
ypredn15 = k_cca(trainn15x,trainy,testn15x,testy);
ypredn30 = k_cca(trainn30x,trainy,testn30x,testy);
ypredn45 = k_cca(trainn45x,trainy,testn45x,testy);

%%
%Voting Scheme
ypred = ypred0 + ypred15 + ypred30 + ypred45 + ypredn30 + ypredn45 %+ ypredn15;  
label = zeros(size(ypred,1),7);
count = 0;
for i=1:size(testy,1)
  [m,id] = max(ypred(i,:));
  label(i,id) = 1;
  if(label(i,id) == testy(i,id))
   count = count + 1;
  end
end
Accuracy = count/size(testy,1)*100

%%
%confusion matrix

confmat = cell(8,9);
confmat{1,1} = 'Emotions';
confmat{2,1} = 'Anger';confmat{3,1} = 'Disgust';confmat{4,1} = 'Fear'; confmat{5,1} = 'Happy'; 
confmat{6,1} = 'Neutral' ; confmat{7,1} = 'Sad'; confmat{8,1} = 'Surprise';
confmat{1,2} = 'Anger';confmat{1,3} = 'Disgust';confmat{1,4} = 'Fear'; confmat{1,5} = 'Happy'; 
confmat{1,6} = 'Neutral' ; confmat{1,7} = 'Sad'; confmat{1,8} = 'Surprise'; confmat{1,9} = 'Total';
for i=2:8
    t1 = find(testy(:,i-1)==1);
    t2 = sum(label(t1,:),1);
    for j=2:8
    confmat{i,j}  = t2(j-1);
    end
    confmat{i,j+1} = sum(t2);
end


