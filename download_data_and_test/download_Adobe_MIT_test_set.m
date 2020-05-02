clc;
clear;
close all;

load Adobe_MIT_test_imgs.mat

A=test_imgs;

if(~exist('..\data\Test\Adobe_MIT','dir'))
    mkdir('..\data\Test\Adobe_MIT')
end

for i=1:size(test_imgs,2)
    
  
    if(~exist(['..\data\Test\Adobe_MIT\' A{1,i}],'file'))       
   
   url = ['https://data.csail.mit.edu/graphics/fivek/img/tiff16_e/' A{1,i}(1:end-6) '.tif'];
   try
   tic;
   data=webread(url);
   data1=imresize(data,0.25,'bicubic');
   
   imwrite(data1,['..\data\Test\Adobe_MIT\' A{1,i}]);   
   tt=toc;
   
   fprintf('%.4d - %f sec \n',i,tt);
   catch
       fprintf('Error - %d',i);
   end
   

    end
   
end
