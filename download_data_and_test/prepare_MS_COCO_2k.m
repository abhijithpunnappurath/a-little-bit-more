clc;
clear;
close all;

unzipped_coco_path = '.\test2014\test2014\';
mkdir('..\data\Test\MS_COCO_2K\');

load MS_COCO_test_imgs.mat

for i=1:size(test_imgs,2)
    fprintf('%d \n',i)
   source=fullfile(unzipped_coco_path,test_imgs{1,i});
   destination=fullfile('..\data\Test\MS_COCO_2K\',test_imgs{1,i});
   copyfile(source,destination)
end