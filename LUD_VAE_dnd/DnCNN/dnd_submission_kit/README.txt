Submission Kit for the Darmstadt Noise Dataset

Version 1.1

# Content

This software package contains code for preparing submissions to the Darmstadt Noise Dataset benchmark described in the CVPR17 paper:

Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.

# Dependencies

The python code requires 
- numpy
- scipy
- h5py

# Usage of Matlab code

First, you have to obtain your denoising results.
- For denoising raw images use the template code given in *denoise_raw*. 
- For denoising sRGB image use the template code given in *denoise_srgb*.
Either way, your denoised results will be put in <submission_folder>.

After having computed the denoised image you need to bundle your submission. 
- If you denoised in raw space, please use *bundle_raw_submission* to bundle your submission. 
- If you denoised in sRGB space, please use *bundle_srgb_submission* to bundle your submission. 

Bundling your submission will create a new folder <submission_folder>/bundled/. Please upload the contents of this folder to our website https://noise.vinsinf.tu-darmstadt.de

# Usage of python code

You can use the python code analogously to the Matlab code. The template code in the functions *denoise_raw* and *denoise_srgb* uses numpy arrays. Please see *pytorch_wrapper.py* for passing the data to and from PyTorch. 

Feel free to provide a wrapper for other frameworks as well!

# Contact

For questions and suggestions please contact Tobias Plötz (tobias.ploetz (at) visinf.tu-darmstadt.de).

Good luck!