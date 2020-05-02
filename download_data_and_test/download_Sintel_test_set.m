clc;
clear;
close all;

load Sintel_test_imgs.mat
mkdir('..\data\Test\Sintel\');

for i=1:size(test_imgs,2)
    filename=['..\data\Test\Sintel\' test_imgs{1,i}];
   url = ['https://media.xiph.org/sintel/sintel-1k-png16/' test_imgs{1,i}];
   
   try
   tic;
   outfilename = websave(filename,url);
   tt=toc;
   
   fprintf('%.4d - %f sec \n',i,tt);
   catch
       fprintf('Error - %d',i);    
   end 

   
end

   

