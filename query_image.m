
clc;
clear all;
close all;

%% Database Loading
hog_feature_vector;
load dbfeat dbfeat

%% function query_image=query_image(k)
cd test                              % changing image files to needed directory
    b =num2str(10);
    imagefile=strcat('(',b,')','.png');
    im = imread(imagefile);
cd ..
im=imresize(im,[128,128]);
imshow(im);
title('Input Image');
%% HOG Process & Feature Extraction Process

im=double(im);
rows=size(im,1);
cols=size(im,2);
Ix=im; %Basic Matrix assignment
Iy=im; %Basic Matrix assignment

% Gradients in X and Y direction. Iy is the gradient in X direction and Iy
% is the gradient in Y direction
for i=1:rows-2
    Iy(i,:)=(im(i,:)-im(i+2,:));
end

% imshow(Iy);
for i=1:cols-2
    Ix(:,i)=(im(:,i)-im(:,i+2));
end
% imshow(Ix);

gauss=fspecial('gaussian',8); %% Initialized a gaussian filter with sigma=0.5 * block width.    

angle=atand(Ix./Iy); % Matrix containing the angles of each edge gradient
angle=imadd(angle,90); %Angles in range (0,180)
magnitude=sqrt(Ix.^2 + Iy.^2);
figure,imshow(uint8(angle));title('HOG Angle Image');
figure,imshow(uint8(magnitude));title('HOG Magnitude Image');
% Remove redundant pixels in an image. 
angle(isnan(angle))=0;
magnitude(isnan(magnitude))=0;

feature=[]; %initialized the feature vector

% Iterations for Blocks
for i = 0: rows/8 - 2
    for j= 0: cols/8 -2
        %disp([i,j])
        
        mag_patch = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        %mag_patch = imfilter(mag_patch,gauss);
        ang_patch = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        
        block_feature=[];
        
        %Iterations for cells in a block
        for x= 0:1
            for y= 0:1
                angleA =ang_patch(8*x+1:8*x+8, 8*y+1:8*y+8);
                magA   =mag_patch(8*x+1:8*x+8, 8*y+1:8*y+8); 
                histr  =zeros(1,9);
                
                %Iterations for pixels in one cell
                for p=1:8
                    for q=1:8
%                       
                        alpha= angleA(p,q);
                        
                        % Binning Process (Bi-Linear Interpolation)
                        if alpha>10 && alpha<=30
                            histr(1)=histr(1)+ magA(p,q)*(30-alpha)/20;
                            histr(2)=histr(2)+ magA(p,q)*(alpha-10)/20;
                        elseif alpha>30 && alpha<=50
                            histr(2)=histr(2)+ magA(p,q)*(50-alpha)/20;                 
                            histr(3)=histr(3)+ magA(p,q)*(alpha-30)/20;
                        elseif alpha>50 && alpha<=70
                            histr(3)=histr(3)+ magA(p,q)*(70-alpha)/20;
                            histr(4)=histr(4)+ magA(p,q)*(alpha-50)/20;
                        elseif alpha>70 && alpha<=90
                            histr(4)=histr(4)+ magA(p,q)*(90-alpha)/20;
                            histr(5)=histr(5)+ magA(p,q)*(alpha-70)/20;
                        elseif alpha>90 && alpha<=110
                            histr(5)=histr(5)+ magA(p,q)*(110-alpha)/20;
                            histr(6)=histr(6)+ magA(p,q)*(alpha-90)/20;
                        elseif alpha>110 && alpha<=130
                            histr(6)=histr(6)+ magA(p,q)*(130-alpha)/20;
                            histr(7)=histr(7)+ magA(p,q)*(alpha-110)/20;
                        elseif alpha>130 && alpha<=150
                            histr(7)=histr(7)+ magA(p,q)*(150-alpha)/20;
                            histr(8)=histr(8)+ magA(p,q)*(alpha-130)/20;
                        elseif alpha>150 && alpha<=170
                            histr(8)=histr(8)+ magA(p,q)*(170-alpha)/20;
                            histr(9)=histr(9)+ magA(p,q)*(alpha-150)/20;
                        elseif alpha>=0 && alpha<=10
                            histr(1)=histr(1)+ magA(p,q)*(alpha+10)/20;
                            histr(9)=histr(9)+ magA(p,q)*(10-alpha)/20;
                        elseif alpha>170 && alpha<=180
                            histr(9)=histr(9)+ magA(p,q)*(190-alpha)/20;
                            histr(1)=histr(1)+ magA(p,q)*(alpha-170)/20;
                        end
                        
                
                    end
                end
                block_feature=[block_feature histr]; % Concatenation of Four histograms to form one block feature
                                
            end
        end
        % Normalize the values in the block using L1-Norm
        block_feature=block_feature/sqrt(norm(block_feature)^2+.01);
               
        feature=[feature block_feature]; %Features concatenation
    end
end

feature(isnan(feature))=0; %Removing Infinitiy values

% Normalization of the feature vector using L2-Norm
feature=feature/sqrt(norm(feature)^2+.001);
for z=1:length(feature)
    if feature(z)>0.2
         feature(z)=0.2;
    end
end
qfeat=feature/sqrt(norm(feature)^2+.001);        
qfeat=abs(sum(qfeat));

%% Database Trainning

%%%Assigning target to each class features
M = 5; N =1;
for i = 1:1:size(dbfeat,2)
if M==0
N = N+1;
M = 2;
else
M = M-1;
end
tv(1:size(dbfeat,1),i) = N;
end

TV=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2];
load dbfeat;
pv = round(abs(dbfeat));
svmstruct = svmtrain(pv,TV);

%% SVM Classification

cout = svmclassify(svmstruct,abs(qfeat));
if isequal(cout,1)
    msgbox('pedestrian');
    query_image11=1;
elseif isequal(cout,2)
    msgbox('non pedestrian');
    query_image11=0;
end











