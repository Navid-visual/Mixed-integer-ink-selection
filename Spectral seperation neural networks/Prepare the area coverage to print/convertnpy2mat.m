addpath('npy_read')
[file,path] = uigetfile('*.npy','choose the file to convert');
filename = [path file];
matfile = double(readNPY(filename));
save(file(1:end-4),'matfile')
