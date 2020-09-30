# Neural network
 Spectral separation network consists of two different neural networks:
*	Forward model: Halftone to reflectance
*	Backward model: Reflectance to halftone

We should start by training the forward network first then we freeze the weights of the forward and train the backward model.

Once you have trained the general backward model you start the adaptive learning (Adaptive learning.py), This will improve the loss of reproduction.

One the adaptive model is trained you can generate the halftone of the painting (generate_halftone.py).

The codes are tuned for reproducing the painting of the flower but you can load other paintings from the “Mixed-integer-ink-selection/Dataset/Spectral paintings/*.mat”
Due too the large size of the images we have to process them in smaller batches and attach them together later. 

The output of the generate halftone is a series of *.npy files containing the halftone of the painting in a n x 8 structure which you need to reshape later to get you 8 layers of halftone maps. Each for one ink channel in the printer. 

