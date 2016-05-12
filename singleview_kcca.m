clear all;clc;
%single view testing
run('kmbox\install')
load('features\ftrain0.mat');
load('features\ftest0.mat');

load('features\trainY.mat');
load('features\testY.mat');


trainx = ftrain0; trainy = trainY;
testx = ftest0;testy = testY;


%%
%KCCA Model
[x1,y1,b,al1,al2] = km_kcca(trainx,trainy,'gauss',1.9,.03,7);

k = sum(x1.*y1)/(x1'*x1);
px = al1;
label = zeros(size(testy,1),7);
count = 0;
for i=1:size(testy,1)
Ktest = km_kernel(trainx,testx(i,:),'gauss',1.9);
atest = px'*Ktest;
btest = k.*atest';
wy = trainy'*al2;
ypred = pinv(wy')*btest';
[m,id] = max(ypred);
label(i,id) = 1;
if(testy(i,id) == 1)
   count = count + 1;
end
end
count/size(testy,1)*100


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




