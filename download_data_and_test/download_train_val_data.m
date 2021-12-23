clc;
clear;
close all;

load Adobe_MIT_train_imgs.mat

A=train_imgs;

if(~exist('..\data\train','dir'))
    mkdir('..\data\train')
end

for i=1:size(A,2)
    
  
    if(~exist(['..\data\train\' A{1,i}],'file'))       
   
   url = ['https://data.csail.mit.edu/graphics/fivek/img/tiff16_' A{1,i}(end-4) '/' A{1,i}(1:end-6) '.tif'];
   try
   tic;
   data=webread(url);
   data1=imresize(data,0.25,'bicubic');
   
   imwrite(data1,['..\data\train\' A{1,i}]);   
   tt=toc;
   
   fprintf('%.4d - %f sec \n',i,tt);
   catch
       fprintf('Error - %d',i);
   end
   

    end
   
end


clear train_imgs
load Sintel_train_imgs.mat

for i=1:size(train_imgs,2)
    filename=['..\data\train\' train_imgs{1,i}];
   url = ['https://media.xiph.org/sintel/sintel-1k-png16/' train_imgs{1,i}];
   
   try
   tic;
   outfilename = websave(filename,url);
   tt=toc;
   
   fprintf('%.4d - %f sec \n',i,tt);
   catch
       fprintf('Error - %d',i);    
   end 

   
end

load Adobe_MIT_val_imgs.mat
clear A
A=val_imgs;

if(~exist('..\data\val','dir'))
    mkdir('..\data\val')
end

for i=1:size(A,2)
    
  
    if(~exist(['..\data\val\' A{1,i}],'file'))       
   
   url = ['https://data.csail.mit.edu/graphics/fivek/img/tiff16_' A{1,i}(end-4) '/' A{1,i}(1:end-6) '.tif'];
   try
   tic;
   data=webread(url);
   data1=imresize(data,0.25,'bicubic');
   
   imwrite(data1,['..\data\val\' A{1,i}]);   
   tt=toc;
   
   fprintf('%.4d - %f sec \n',i,tt);
   catch
       fprintf('Error - %d',i);
   end
   

    end
   
end


