
completehalftone=[];
path='.\image\';
for i=1:10
    file_name=sprintf('test_im%d.npy',i-1);
    path_name=[path file_name];
    
    approx_halftone = readNPY(path_name);
    completehalftone=[completehalftone;approx_halftone];
end

painting_reshape_back=reshape(double(completehalftone),[2344 1968 8]); % find the image dimention from Mixed-integer-ink-selection\Dataset\Spectral paintings\image dimensions.txt
painting_reshape_back=permute(painting_reshape_back,[2 1 3]);
painting_reshape_back(painting_reshape_back>1)=1;

name={'Yellow', 'LightBlack','LightMagenta','LightCyan','LightLightBlack','Magenta','Cyan','Black'};% Order of color channels
for i=1:8
    ss = name{i};
    imwrite(1-painting_reshape_back(:,:,i),sprintf('%s.tiff',ss),'tiff');
    
end