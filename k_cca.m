function[ytest] = k_cca(trainx,trainy,testx,testy)

%calling the km_kcca function 
%input are training images and its corresponding labels
%gaussian kernel is used
%other inputs are kernel parameter, regularization parameter and low rank respectively
%outputs are the projection of x and y 
% b is the correlation
%al1 and al2 are alpha1 and aplha2

[x1,y1,b,al1,al2] = km_kcca(trainx,trainy,'gauss',1.9,.03,7);

k = sum(x1.*y1)/(x1'*x1);
px = al1;
label = zeros(size(testy,1),7);
count = 0;
ytest = [];
%estimation of ytest

    for i=1:size(testy,1)
        Ktest = km_kernel(trainx,testx(i,:),'gauss',1.9);
        atest = px'*Ktest;
        btest = k.*atest';
        wy = trainy'*al2;
        ytest = [ytest; (pinv(wy')*btest')'];   
    end
end