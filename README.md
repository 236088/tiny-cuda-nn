# Tiny CUDA Neural Networks ![](https://github.com/NVlabs/tiny-cuda-nn/workflows/CI/badge.svg)

fork from https://github.com/NVlabs/tiny-cuda-nn

This is a small, self-contained framework for training and querying neural networks. Most notably, it contains a lightning fast ["fully fused" multi-layer perceptron](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/fully-fused-mlp-diagram.png) ([technical paper](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf)), a versatile [multiresolution hash encoding](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/multiresolution-hash-encoding-diagram.png) ([technical paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)), as well as support for various other input encodings, losses, and optimizers.



## License and Citation

This framework is licensed under the BSD 3-clause license. Please see `LICENSE.txt` for details.

If you use it in your research, we would appreciate a citation via
```bibtex
@software{tiny-cuda-nn,
	author = {M\"uller, Thomas},
	license = {BSD-3-Clause},
	month = {4},
	title = {{tiny-cuda-nn}},
	url = {https://github.com/NVlabs/tiny-cuda-nn},
	version = {1.7},
	year = {2021}
}
```

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)


## Publications

Among others, this framework powers the following publications:

> __Instant Neural Graphics Primitives with a Multiresolution Hash Encoding__  
> [Thomas Müller](https://tom94.net), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _ACM Transactions on Graphics (__SIGGRAPH__), July 2022_  
> __[Website](https://nvlabs.github.io/instant-ngp/)&nbsp;/ [Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)&nbsp;/ [Code](https://github.com/NVlabs/instant-ngp)&nbsp;/ [Video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)&nbsp;/ [BibTeX](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.bib)__

> __Extracting Triangular 3D Models, Materials, and Lighting From Images__  
> [Jacob Munkberg](https://research.nvidia.com/person/jacob-munkberg), [Jon Hasselgren](https://research.nvidia.com/person/jon-hasselgren), [Tianchang Shen](http://www.cs.toronto.edu/~shenti11/), [Jun Gao](http://www.cs.toronto.edu/~jungao/), [Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Thomas Müller](https://tom94.net), [Sanja Fidler](https://www.cs.toronto.edu/~fidler/)  
> __CVPR (Oral)__, June 2022  
> __[Website](https://nvlabs.github.io/nvdiffrec/)&nbsp;/ [Paper](https://nvlabs.github.io/nvdiffrec/assets/paper.pdf)&nbsp;/ [Video](https://nvlabs.github.io/nvdiffrec/assets/video.mp4)&nbsp;/ [BibTeX](https://nvlabs.github.io/nvdiffrec/assets/bib.txt)__

> __Real-time Neural Radiance Caching for Path Tracing__  
> [Thomas Müller](https://tom94.net), [Fabrice Rousselle](https://research.nvidia.com/person/fabrice-rousselle), [Jan Novák](http://jannovak.info), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _ACM Transactions on Graphics (__SIGGRAPH__), August 2021_  
> __[Paper](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf)&nbsp;/ [GTC talk](https://gtc21.event.nvidia.com/media/Fully%20Fused%20Neural%20Network%20for%20Radiance%20Caching%20in%20Real%20Time%20Rendering%20%5BE31307%5D/1_liqy6k1c)&nbsp;/ [Video](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.mp4)&nbsp;/ [Interactive results viewer](https://tom94.net/data/publications/mueller21realtime/interactive-viewer/)&nbsp;/ [BibTeX](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.bib)__


## Acknowledgments

Special thanks go to the NRC authors for helpful discussions and to [Nikolaus Binder](https://research.nvidia.com/person/nikolaus-binder) for providing part of the infrastructure of this framework, as well as for help with utilizing TensorCores from within CUDA.
