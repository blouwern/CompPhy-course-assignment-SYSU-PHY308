# 上机实验报告：MPI/OpenMP 并行计算矩阵乘法

王志超 23343062 2026年4月16日
- [上机实验报告：MPI/OpenMP 并行计算矩阵乘法](#上机实验报告mpiopenmp-并行计算矩阵乘法)
  - [实践内容介绍](#实践内容介绍)
  - [上机结果与讨论](#上机结果与讨论)
    - [算法原理](#算法原理)
    - [程序实例使用说明](#程序实例使用说明)
      - [性能测试程序](#性能测试程序)
      - [综合性展示程序](#综合性展示程序)
    - [性能测试——不同算法在不同矩阵规模下的性能对比](#性能测试不同算法在不同矩阵规模下的性能对比)
      - [CBLAS 系列算法](#cblas-系列算法)
      - [Rough Simple (手写串行) 系列算法](#rough-simple-手写串行-系列算法)
      - [综合分析](#综合分析)
    - [性能测试——不同节点数量下MPI(OpenMP)的运行速度](#性能测试不同节点数量下mpiopenmp的运行速度)
  - [附录程序说明](#附录程序说明)


## 实践内容介绍
利用MPI(OpenMP)进行矩阵乘以矩阵加速计算，对比(自己写的)串行和MPI(OpenMP)并行运行时间，并与 CBLAS 库中的 `dgemm`（Python 中 NumPy 或 SciPy 的类似函数）串行计算进行比对。
## 上机结果与讨论
### 算法原理
计算式 $A \cdot B = C$

MPI实现采用最简单的分批计算的方法
- 对所有进程进行集体通信，从主节点(rank=0)将B矩阵广播
- 主进程一次性将矩阵A进行分块，分发给所有工作进程
- 工作进程计算出各自分块结果，送回给主进程

OpenMP实现与MPI实现类似，区别在于 OpenMP 并行计算时不需要广播矩阵B或分发矩阵A。每个线程都共享同一块内存，可以直接访问矩阵A的不同部分进行计算，且每个线程计算完成后直接将结果写回矩阵C，没有收集结果的过程。

### 程序实例使用说明
#### 性能测试程序
项目根据并行方法分为 MPI、OpenMP 和串行；而每个并行节点/线程的计算方法又分为原始手写串行和调用 CBLAS 库函数两种。根据不同的组合，项目有六个可执行文件以供性能测试，分别为：
- MM_cblas_mpi：MPI并行计算，工作节点计算采用CBLAS串行
- MM_cblas_openmp：OpenMP并行计算，工作线程计算采用CBLAS串行
- MM_cblas_seq：串行计算，计算采用CBLAS串行
- MM_RS_mpi：MPI并行计算，工作节点计算采用原始手写(RS即Rough Simple缩写)串行
- MM_RS_openmp：OpenMP并行计算，工作线程计算采用原始手写串行
- MM_RS_seq：串行计算，计算采用原始手写串行

调用方法：
```bash
<git-repo>$ mkdir build
<git-repo>/build$ cd build
<git-repo>/build$ cmake -G Ninja ..
<git-repo>/build$ ninja
<git-repo>/build$ mpirun ./MM_cblas_mpi [OPTIONAL:<matrix dimension of A & B>]
<git-repo>/build$ ./MM_cblas_openmp [OPTIONAL:<matrix dimension of A & B>]
<git-repo>/build$ ./MM_cblas_seq [OPTIONAL:<matrix dimension of A & B>]
<git-repo>/build$ mpirun ./MM_RS_mpi [OPTIONAL:<matrix dimension of A & B>]
<git-repo>/build$ ./MM_RS_openmp [OPTIONAL:<matrix dimension of A & B>]
<git-repo>/build$ ./MM_RS_seq [OPTIONAL:<matrix dimension of A & B>]
```
默认矩阵维度为 $N=1000$，即 $A(1000, 500) \cdot B(500, 2000)$。

#### 综合性展示程序
同时项目还具有一个综合性的可执行文件 `comprehensive_demo`，用以展示不同算法的性能对比。由于 OpenMP 并行计算原理与实现方法较为简单，该可执行文件仅包含 MPI 并行计算的性能对比。根据宏定义的实现细节，`comprehensive_demo` 有多个编译可选项（可在 `CMakeLists.txt` 的 `target_compile_options()` 中自定义配置宏进行调整）：
- NON_BLOCKING_COMMUNICATION——主节点是否采用无阻塞通信：非串行分发/接收信息，而是一次性开启多个线程并行分发/接收信息
- MPI_COMPUTATION_USE_BLAS——工作节点计算的方法：原始手写串行或调用 `cblas_dgemm`
- ADD_RSSEQ_COMPARE——是否加入原始手写串行对比：原始手写串行很慢，如果设置为加入对比注意调整矩阵维度以免耗费过多时间
- REPORT_DEBUG_INFO——MPI通信过程展示，用以调试和理解MPI通信过程

调用方法：
```bash
<git-repo>$ mkdir build
<git-repo>/build$ cd build
<git-repo>/build$ cmake -G Ninja ..
<git-repo>/build$ ninja
<git-repo>/build$ mpirun ./comprehensive_demo [OPTIONAL:<matrix dimension of A & B>]
```
默认矩阵维度为 $N=1000$，即 $A(1000, 500) \cdot B(500, 2000)$。

### 性能测试——不同算法在不同矩阵规模下的性能对比

本节固定测试六个程序：MM_cblas_mpi、MM_cblas_openmp、MM_cblas_seq、MM_RS_mpi、MM_RS_openmp、MM_RS_seq。为简化不同矩阵大小的描述，令A与B矩阵维度均为$N \times N$（注意：矩阵下标使用int类型标记，$N^2$若超过int类型最大值则需要调整）。

测试设置与统计口径如下：

1. 矩阵规模取 $N=10,100,500,1000,5000,10000$。
2. 每个程序在每个 $N$ 下重复运行 5 次，取平均运行时间。
3. 运行时间从程序标准输出中的 `[Time taken]<module_name>` 行提取，`module_name` 用于标识算法名称。
4. 原始数据文件为 report/data/perf_results_6_programs.csv。

下图给出了六个可执行文件在不同矩阵规模下的平均运行时间。为了同时观察小规模与大规模矩阵的变化趋势，横轴和纵轴都采用了对数坐标。

图1：六个程序总体性能对比
![六个程序总体性能对比](figures/perf/perf_overview_loglog.png)

图2：CBLAS系列算法性能对比
![CBLAS 系列性能对比](figures/perf/perf_cblas_loglog.png)

图3：Rough Simple系列算法性能对比
![Rough Simple 系列性能对比](figures/perf/perf_rs_loglog.png)

从总体结果看，性能差异主要由两部分决定：一是每个节点/线程内部采用的是 CBLAS 还是手写三重循环，二是外层并行方式是 MPI 还是 OpenMP。六个程序的结果总结分析如下：

#### CBLAS 系列算法

CBLAS 系列算法在所有规模上都明显快于 Rough Simple 系列，说明底层 BLAS 实现对缓存、向量化（SIMD，单指令多数据）和块划分的优化非常有效。

CBLAS 系列内部对比时，串行版本始终最快，MPI 和 OpenMP 的外层并行反而引入了通信、调度和线程管理开销；在 $N=10000$ 时，CBLAS sequential 约为 11.48 s，CBLAS MPI 约为 15.98 s，CBLAS OpenMP 约为 23.09 s。本质上外层并行任务是通过充分利用计算资源，避免单线程读取内存数据时本可以进行的无关本段数据的计算被阻塞。当任务的浮点运算已经占据绝大部分时间，并且所有的核心都被用于运算，外层并行的收益就不明显了，反而通信和调度开销会影响运算性能。

#### Rough Simple (手写串行) 系列算法

小规模矩阵下，MPI 和 OpenMP 的额外开销更容易抵消并行收益，因此并行版本不一定比串行版本快；随着 $N$ 增大，计算量快速增长，并行收益才逐渐体现出来。

对于 Rough Simple 系列计算大型矩阵，OpenMP 和 MPI 都能明显优于纯串行实现，尤其在大规模矩阵下效果更明显；在 $N=10000$ 时，Rough Simple sequential 约为 6922.71 s，MPI 约为 1502.47 s，OpenMP 约为 1002.42 s。以下是更加细致的调查分析：
   
图4：手写串行算法运行时CPU负载的可视化
![手写串行算法运行时CPU负载的可视化](figures/monitoring/MM_RS_seq_cpu_cores.png)
图4所示是通过btop对手写串行算法运行时CPU负载的可视化，可以看到手写串行算法运行时仅仅使用一个核心，而其他核心处于空闲状态；同时还可以看到进程在不同核心之间反复切换，当矩阵很大时，这样在核心之间切换上下文的开销是很大的。

图5：MPI并行算法运行时CPU负载的可视化
![MPI并行算法运行时CPU负载的可视化](figures/monitoring/MM_RS_mpi_cpu_cores.png)
图6：MPI并行算法运行时CPU频率的可视化
![MPI并行算法运行全过程的CPU频率波动图](figures/monitoring/MM_RS_mpi_cpu_cores_2.png)

图5所示是通过btop对MPI并行算法运行时CPU负载的可视化，可以看到MPI并行算法运行时多个核心都被充分利用。

但是也可以从图6所示的CPU频率波动图中看到，MPI并行算法随着运行时间增加，CPU总负载逐渐降低，大概率是手动分配的计算任务不均匀：
1. 对于N行矩阵，m个进程，当N不能被m整除时，每个进程分配到的行数有两种批次，一种是N/m行，另一种是N/m+1行；
2. 这导致CPU频率显著分为前后两段，前一段所有的进程都在计算，后一段已经算完N/m行的进程已经完成任务，而主进程仍然需要等待N/m+1行的进程完成任务。
3. 当N很大时，这种不均匀分配带来的性能问题就会逐渐显现出来。
4. 在实现MPI算法时为减少代码复杂度，没有给主节点安排计算任务，或许给主节点安排一部分计算任务可以有性能提升。

图7：OpenMP并行算法运行时CPU负载的可视化
![OpenMP并行算法运行时CPU负载的可视化](figures/monitoring/MM_RS_openmp_cpu_cores.png)
图8：OpenMP并行算法运行时CPU频率的可视化
![OpenMP并行算法运行全过程的CPU频率波动图](figures/monitoring/MM_RS_openmp_cpu_cores_2.png)

对比OpenMP并行算法的CPU频率波动图，相比于硬性对矩阵分块计算的MPI，OpenMP并行算法能够更好地平衡每个硬件核心的负载。
1. 推测OpenMP在并行运算过程中会动态调整每个线程的计算任务，避免了类似MPI由于手动分配任务不均匀带来的性能问题；线程之间共享内存，不存在互相不可见的数据，可以随时调整运算任务，这是OpenMP可以实现这点的原因。
2. 并且OpenMP并行算法不需要像MPI那样依赖主节点进行数据分发和收集，不存在节点通信带来的性能开销，能够更好地利用每个核心的运算资源。

3. CBLAS 与 Rough Simple 的差距远大于 MPI 与 OpenMP 的差距，这说明算法/库级优化对矩阵乘法（尤其是大型矩阵）性能的影响比外层并行调度更显著。

#### 综合分析
综上所述，若使用高性能 BLAS 计算核心，继续叠加外层并行未必带来收益；而对于手写三重循环，MPI/OpenMP 能显著缓解计算瓶颈，但仍难以追上 BLAS 级别的优化。

### 性能测试——不同节点数量下MPI(OpenMP)的运行速度

本节固定矩阵规模为 $N=2000$，分别测试 Rough Simple MPI 与 Rough Simple OpenMP 在不同并行度下的运行时间。每个并行度重复运行 5 次，取平均值。对应原始数据见 report/data/perf_results_node_scaling.csv。

图9展示了不同进程/线程数量下的平均运行时间变化趋势。

图9：不同节点数量（进程/线程数）下的平均运行时间
![不同节点数量下的平均运行时间](figures/scaling/perf_node_scaling_time.png)

为了更直观比较扩展效率，图10给出相对加速比，定义为
$$
S(p)=\frac{T_{\text{base}=4}}{T_p}
$$
其中 $T_p$ 表示并行度为 $p$ 时的平均运行时间，基准取 $p=4$。

图10：相对加速比（以4进程/4线程为基准）
![不同节点数量下的相对加速比](figures/scaling/perf_node_scaling_speedup.png)

根据测试结果可得到以下结论：

1. OpenMP 在本组测试中整体优于 MPI。以并行度 14 为例，OpenMP 平均时间约 4.736 s，而 MPI 约 6.321 s，OpenMP 约快 1.33 倍。
2. OpenMP 曲线随线程数增加总体单调下降（4→14：9.429 s→4.736 s），相对加速比从 1.00 提升到 1.99，扩展趋势较稳定。
3. MPI 曲线存在明显波动，尤其在 8 进程处出现回退（6 进程约 7.941 s，8 进程约 10.217 s），说明并行度提升并不总能直接转化为性能收益。
4. MPI 在 14 进程时达到本组最优（6.321 s），相对 4 进程加速比为 1.83，但仍低于 OpenMP 的 1.99。

结合前文对 CPU 负载与频率波动的观察，这种差异可以由以下因素解释：

1. MPI 存在显式通信与同步开销，并且当前实现中主进程不参与计算，整体资源利用率受限。
2. 当行分块不能完全均匀时，MPI 末尾阶段会出现“快进程等待慢进程”的现象，导致部分并行度下出现性能抖动。
3. OpenMP 线程共享内存、调度更灵活且成本相对低，在本机多核场景下更容易保持负载平衡，扩展性表现更平滑。
4. 但是，在更大规模矩阵或分布式环境下，MPI 的优势可能会逐渐显现；而在当前测试条件下，OpenMP 的性能表现更优。

## 附录程序说明
源码已经上传至 [GitHub](https://github.com/blouwern/CompPhy-course-assignment-SYSU-PHY308) 仓库，源码详情请参见 `src` 和 `include` 目录下的源代码文件。

使用方法参见此节：[程序实例使用说明](#程序实例使用说明) 或者 GitHub 仓库中的 [`README.md`](https://github.com/blouwern/CompPhy-course-assignment-SYSU-PHY308/blob/master/README.md)。