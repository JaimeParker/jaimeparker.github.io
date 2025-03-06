---
title: "The Bottleneck of Numpy due to Different Version"
categories: tech
tags: [Python]
use_math: true
---

Check the MKL or OpenBLAS version of NumPy.

The version of **numpy** may cause slow training speeds. For more information, refer to [AutoDL Help Documentation: Performance Section - numpy version issue](https://www.autodl.com/docs/perf/#numpy) CN.

> Preliminary identification method: If CPU usage is very high, all cores are fully utilized, and even after upgrading to more cores, the CPU is still fully utilized while GPU usage remains low, and if an Intel CPU is being used, there's a significant chance that the performance issue is caused by the numpy version.


This observation is not entirely consistent. In Stable Baselines3 with PyTorch-GPU training, when running multiple parallel environments, CPU cores often only work on a few cores in rotation, with CPU usage reaching its peak and then lowering, after which different cores are used in turn. This process repeats itself.

NumPy uses OpenBLAS or MKL for computation acceleration. Intel CPUs support MKL, while AMD CPUs only support OpenBLAS. When using Intel CPUs, MKL provides a significant performance boost (several times faster in some matrix computations), which greatly impacts the overall performance. Generally, AMD CPUs using OpenBLAS perform faster than Intel CPUs using OpenBLAS, so there's no need to overly worry about the performance difference between Intel and AMD CPUs when using OpenBLAS.

If you are using an Intel CPU, first check whether the version of NumPy you're using is MKL or OpenBLAS.

```python
import numpy as np
print(np.__config__.show())
```

If the output contains the word **mkl**, it indicates the MKL version, as shown below:

```
Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)] on win32
import numpy as np
np.__config__.show()
blas_armpl_info:
  NOT AVAILABLE
blas_mkl_info:
    libraries = ['mkl_rt']
    library_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\include']
blas_opt_info:
    libraries = ['mkl_rt']
    library_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\include']
lapack_armpl_info:
  NOT AVAILABLE
lapack_mkl_info:
    libraries = ['mkl_rt']
    library_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\include']
lapack_opt_info:
    libraries = ['mkl_rt']
    library_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['D:/Software/anaconda3/envs/sb3\\Library\\include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
```

When using domestic(China) Conda mirrors like **Tsinghua**, NumPy is typically installed with the **OpenBLAS** acceleration scheme by default. If you install NumPy using conda install numpy, you will see **OpenBLAS-related** packages being installed.