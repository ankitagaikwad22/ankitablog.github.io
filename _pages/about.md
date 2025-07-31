---
permalink: /
title: "Separable Physics-Informed Neural Networks"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
<style>
p {
    text-align: justify;
}
</style>



Welcome to my blog post describing and explaining the paper "Separable Physics-Informed Neural Networks" by Cho et al., which was presented at NeurIPS 2023. This research describes a unique and scalable method for solving PDEs that restructures how neural networks model multidimensional input. As someone who is very interested in the convergence of deep learning and computational physics, I found SPINNs to be a novel and interesting approach to addressing the bottlenecks of previous PINN structures. It demonstrates how traditional concepts such as variable separation and tensor decomposition may be combined with neural networks to provide faster, more efficient scientific computing models.

This blog article is based on my reading of the research, my seminar presentation on SPINNs, and comparisons with other comparable methods such as PINNs, Causal PINNs, and low-rank neural PDE solvers. I've endeavored to make the concepts comprehensible even if you're not deeply into numerical methods or theoretical machine learning.

In this blog post, we examine Separable Physics-Informed Neural Networks (SPINNs), a sophisticated method for applying machine learning to the solution of high-dimensional partial differential equations (PDEs). The exponential cost of sampling and differentiation in high dimensions affects the scalability of traditional Physics-Informed Neural Networks (PINNs). By segmenting the network according to input dimensions and utilizing forward-mode automatic differentiation, SPINNs provide a straightforward yet effective redesign. When combined, these two concepts and result in significant increases in speed and memory effectiveness. With an emphasis on the physics-inspired motivations and little mathematical overhead, this post seeks to make the concepts underlying SPINNs understandable and approachable. Let’s decompose.


<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body" style="text-align: center;">
      <div style="max-width: 700px; margin: 0 auto;">
        <video id="teaser" autoplay controls muted loop playsinline style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
          <source src="images/Blog_videos/Navier_Stokes.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  </div>
</section>


<div style="text-align: center;">
  <p style="font-size: 0.95em; color: #555; text-align: center; margin-top: 0.5em;">Figure 1: PINN solves multi-dimensional PDEs 60× faster than conventional PINN. [1]</p>
</div>


# Introduction


Partial differential equations (PDEs) are a fundamental problem in computational science and engineering. PDEs are used in a variety of applications, including fluid flow and electromagnetic field simulations, as well as modeling quantum systems and weather patterns.  
Traditional methods to PDE solutions are Finite Element Methods or Finite Difference Methods which require lots of computation and it struggle with high dimensions. SO it was so challenging for In real-world scenarios, especially in high dimensions or complex geometries, classical solvers become too expensive or impractical.

In recent years, Physics-Informed Neural Networks (PINNs) have gained popularity as a mesh-free, data-efficient alternative. PINNs learn PDE solutions by explicitly encoding physical rules into a neural network's loss function. They operate exceptionally well for many situations, particularly in low dimensions, and require no labeled data. . Despite its potential, PINNs encounter significant challenges when used to high-dimensional PDEs. The number of points required to adequately record the physics explodes, and computing gradients using reverse-mode automated differentiation becomes prohibitively expensive. 

This is where the study "Separable Physics-Informed Neural Networks" comes in. SPINNs address the fundamental constraints of PINNs by making two innovative modifications: 
1. A separable architecture uses small MLPs for each input dimension and a low-rank tensor product to assemble the full solution. 
2. Forward-mode automatic differentiation is ideal for computing multiple output derivatives with few inputs, a common scenario in physics problems.
These changes may appear modest, but their influence is profound: SPINNs can achieve up to 62× speedups, reduce memory utilization by 29×, and handle PDEs in up to (3+1)D utilizing commodity GPUs, all while maintaining or enhancing accuracy.



# What Are PINNs and Where Do They Struggle? 


A PINN is a neural network that learns a function u(x) satisfying a PDE by minimizing the residuals of the PDE and initial/boundary conditions using automatic differentiation (AD).
PINNs approximate the solution u(x) to a PDE by minimizing 
- Residual loss using differential equation. 
- Losses due to initial and boundary conditions. 
These are enforced using automatic differentiation (often reverse-mode) and trained with standard gradient descent. These models are mesh-free, data-efficient (unsupervised), and can solve both forward and inverse problems. 

However, the cost of analyzing the network and computing gradients rises considerably as dimensionality increases. To solve PDEs with finer grids or higher dimensions, PINNs require 
- a large number of collocation points. 
- Cannot handle large training sets on single gpu
- they face high computational and memory costs 
- they suffer from slow convergence.
These scalability limits are well-documented [1, 3]. SPINNs confront things head on. 


# Introducing SPINNS


So to overcome all the problems researchers comes with a solution i.e SPINNS- which stands for Separable Physics-Informed Neural Networks.
It's a new way to structure and train PINNs that:
-	Handles multi-dimensional PDEs more efficiently
-	Allows using more collocation points (>10 million!) even on a single GPU
-	Is much faster and more accurate than traditional PINNs

To understand why SPINN is so quick, we must first consider how derivatives are produced in deep learning – especially, by automatic differentiation. There are two major modes:

## Forward vs Reverse Mode AD




| **Mode**        | **Best For**                          | **Use Case Example**                |
|-----------------|---------------------------------------|-------------------------------------|
| Reverse-mode    | Many inputs → single/scalar output    | Backpropagation in neural networks |
| Forward-mode    | Few inputs → many outputs             | Computing PDE residuals            |

<div style="text-align: center;">
   <p style="font-size: 0.95em; color: #555; text-align: center; margin-top: 0.5em;">Table 1: Comparison of Automatic Differentiation Modes</p>
</div>


**Reverse-Mode AD:**

Reverse-Mode AD (used in backpropagation) is ideal for functions with multiple inputs and a single output (e.g., loss functions). It operates by first performing a forward pass to compute outputs, followed by a backward pass to calculate gradients with respect to inputs. This is useful for training neural networks that require the gradient of a scalar loss with respect to all model parameters.

**Forward-Mode AD:**

Forward-Mode AD works best for functions with few inputs and many outputs, such as computing derivatives of solutions with respect to input coordinates in PDEs. Forward-mode AD computes these derivatives immediately during the forward pass, making it more efficient when assessing PDE residuals.

In SPINN, the number of input variables (coordinates) is minimal (for example, time and space), but the number of outputs (solution evaluations at various points) is huge, making forward-mode AD the best fit.


# SPINN Architecture


<div style="text-align: center;">
  <img src="images/Blog_images/Architecture.png" alt="SPINN Architecture" style="width: 75%; margin: 0 auto; display: block;" />
  <p style="font-size: 0.95em; color: #555; text-align: center; margin-top: 0.5em;">
    Figure 2: Architecture (Single Input) [1]
  </p>
</div>



SPINNs solve d-dimensional PDEs using a separated neural network design in which each input dimension is processed by a distinct MLP. Each of these networks converts a scalar into a feature vector. The final scalar output is then calculated by performing an element-wise product on these vectors and summing over the shared rank dimension. 
Mathematically, the output of the model is:


$$
\hat{u}(x) = \sum_{j=1}^{r} f_j^{(\theta^{(1)})}(x_1) f_j^{(\theta^{(2)})}(x_2) \cdots f_j^{(\theta^{(d)})}(x_d)
$$

 
The model produces a CP-decomposition (CANDECOMP/PARAFAC) of a tensor that represents the solution. Each function is a small axis-specific MLP, and the network learns a solution with a low rank. 
The architecture enables rapid evaluation and training on structured input grids. 
The process involves sampling 1D points on each axis, translating them to a -dimensional vector using MLPs, and generating an element-wise product and summation over shared rank as output. 
- The authors use tanh activations for all MLPs. 
This structure is very parallelizable and efficient as the outputs of each MLP can be reused across several axis inputs. 
- Applying forward-mode AD to 1D MLPs reduces Jacobian computation costs dramatically. 
The whole discretized solution tensor can be evaluated in batches utilizing structured Cartesian product sampling of coordinates, making SPINNs appropriate for large-scale problems with low overhead.

# Detailed Experiments and Results



The authors of the SPINN paper performed rigorous experiments across a wide spectrum of PDEs, ranging from simple linear systems like diffusion to complex nonlinear equations such as Navier-Stokes. Each experiment highlights different advantages of SPINN — computational speed, memory efficiency, and accuracy scaling with resolution. Below is a breakdown of the major experiments and their findings:

## Hardware & Setup

All experiments were conducted on a single NVIDIA RTX 3090 GPU with 24GB of memory. The maximum resolution for standard PINNs was limited by GPU memory, while SPINN could scale up to significantly more collocation points thanks to its low memory footprint.
Each model was evaluated using:
-	Relative error
-	Wall-clock training time
-	GPU memory usage
-	Rank sensitivity (for SPINNs)




<div style="text-align: center; margin-top: 2em; margin-bottom: 1em;">
  <img src="images/Blog_images/Results.png" alt="SPINN PDE benchmark results" style="width: 90%; max-width: 1000px; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 0 8px rgba(0,0,0,0.1);" />
  <p style="font-size: 0.95em; color: #555; text-align: center; margin-top: 0.5em;">
    <strong>Figure:</strong> Figure 3: SPINN’s performance across various PDE benchmarks, including Diffusion, Helmholtz, and Klein-Gordon equations. [1]
  </p>
</div>


### Diffusion Equation

The Diffusion equation is a foundational PDE in physics, used to model heat conduction and particle diffusion.

1. In this experiment, SPINN-mod consistently achieves the best accuracy.
2. At a resolution of 128³ collocation points, SPINN-mod is:
-    52× faster than PINN-mod
-    29× more memory-efficient
3. Even when using the same number of points, SPINN delivers more accurate results with far less computational cost.

### Helmholtz Equation

Helmholtz equations appear in problems involving wave propagation and vibrations, and are notoriously hard to solve due to high-frequency components.

1. Traditional PINNs struggle here due to stiffness in gradient flow.
2. SPINN-mod, on the other hand, handles the equation gracefully:
-    Achieving a 62× speedup
-    Using 29× less GPU memory

### Klein-Gordon Equation (2+1D and 3+1D)

The Klein-Gordon equation, important in quantum field theory, is tested in two forms:

***(2+1)D Klein-Gordon***

1. SPINN-mod outperforms all baselines, with:
-    Lowest relative L₂ error
-    62× faster runtime than PINN-mod
-    29× memory savings
2. As the number of collocation points increases, SPINN scales better than PINNs both computationally and numerically, enabling larger, more stable models.


***(3+1)D Klein-Gordon***

To really push the limits, the team tested SPINN in 4D by adding an extra spatial axis:

1. PINNs (even with a modified MLP) hit memory limits at just 18⁴ points.
2. SPINN, however, processes up to 64⁴ collocation points — that's a 160× improvement in capacity!
3. This shows SPINN’s incredible scalability for high-dimensional scientific computing.

### Navier-Stokes Equation (2+1)D

Navier-Stokes governs fluid flow, and solving it accurately is central to computational physics.

1. SPINN was tested against Causal PINNs and PINN-mod, aiming to predict velocity fields and compute vorticity using forward-mode AD.
2. Results show that SPINN:
-    Is 60× faster
-    Uses ~25% of the memory
-    Achieves comparable or better accuracy

3. Crucially, SPINN achieved this without even using a causal loss function — which Causal PINNs depend on.


<div style="text-align: center; margin-top: 2em; margin-bottom: 1em;">
  <img src="images/Blog_images/3D_Navierstokes.png" alt="3D Navier-Stokes" style="width: 70%; max-width: 900px; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 0 8px rgba(0,0,0,0.1);" />
  <p style="font-size: 0.95em; color: #555; margin-top: 0.8em;">
    <strong>Figure 4:</strong> 3D Navier-Stokes results comparing SPINN to PINN variants in terms of error, runtime, and memory.
  </p>
</div>


### Navier-Stokes Equation (3+1)D

To top it off, SPINN was extended to the extremely complex (3+1)D Navier-Stokes, involving 33 derivative terms!

Even with this complexity:

-    SPINN trained in under 30 minutes
-    Reached a relative error of 1.9e-3
-    Used less than 3GB of GPU memory at 32⁴ collocation points


<div style="text-align: center; margin-top: 2em; margin-bottom: 1em;">
  <img src="images/Blog_images/4D_Navierstokes.png" alt="4D Navier-Stokes Results" style="width: 55%; max-width: 900px; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 0 8px rgba(0,0,0,0.1);" />
  <p style="font-size: 0.95em; color: #555; margin-top: 0.8em;">
    <strong>Figure 5:</strong> 4D Navier-Stokes results showing SPINN’s scalability and performance across different collocation resolutions.
  </p>
</div>



This cements SPINN as a highly scalable, fast, and accurate framework for chaotic and nonlinear PDEs in very high dimensions



<div style="text-align: center;">
  <img src="images/Blog_images/Navier_training.png" style="width: 75%; margin: 0 auto; display: block;" />
  <p style="font-size: 0.95em; color: #555; text-align: center; margin-top: 0.5em;">
    **Figure 1:** Training speed (w/ a single GPU) of our model compared to the causal PINN [4] in (2+1)-d Navier-Stokes equation of time interval [0, 0.1].

  </p>
</div>

## Experimental Summary – SPINN vs PINNS

| PDE Problem              | Best Accuracy Achieved By     | Speed      | Memory Usage       |
|--------------------------|-------------------------------|------------|--------------------|
| Diffusion     | SPINN-mod                     | 52× faster | 29× less           |
| Helmholtz       | SPINN-mod                     | 62× faster | 29× less           |
| Klein-Gordon (2+1)D | SPINN-mod                     | 62× faster | 29× less           |  
| Klein-Gordon (3+1D)      | SPINN                         | 160× more points | Not feasible for PINN           |
| Navier-Stokes (2+1D)     | SPINN                         | 60× faster |  ~75% less usage  |
| Navier-Stokes (3+1D)     | SPINN (Relative error 1.9e-3) | <30 min    | <3 GB (at 32⁴ pts) |



## Conclusion and Future Work:


SPINNs are a significant achievement in the field of physics-based machine learning.  By integrating the principles of variable separation and low-rank tensor decomposition with forward-mode automatic differentiation, SPINNs provide an elegant, scalable, and extremely efficient framework for solving high-dimensional PDEs.
 Key takeaways are:

1. **Efficiency:** SPINNs offer significant gains in performance and memory usage, making them suitable for large-scale scientific simulations.
2. **Scalability:** SPINNs can solve 3D+time PDEs on a single GPU, surpassing typical PINNs that struggle with memory limits.
3. **Accuracy:** SPINNs maintain or increase solution accuracy across many PDE types, despite their reduced computational complexity.

Looking forward, several exciting directions are possible:
- **Adaptive Rank Learning:** Future models can dynamically learn the ideal tensor rank during training, eliminating the need for human configuration.
- **Integration with Neural Operators:** Combining SPINNs with operator learning methods like DeepONets or FNOs can improve generalization across PDE families.
- **Geometric Generalization:** Extending SPINNs to irregular domains or manifolds could expand their use in engineering and physics. 
- **Robustness to Noise or Data:** Incorporating measurement data or noisy limitations can improve simulation's robustness to real-world observations.
 

Finally, SPINNs provide a principled but practical technique to solving large-scale PDEs.  They not only overcome the limits of ordinary PINNs, but also pave the way for neural network-based solutions to hitherto intractable scientific challenges.


## References:


1. Cho, J., Nam, S., Yang, H., Hong, Y., Yun, S.-B., & Park, E. (2023). *Separable Physics-Informed Neural Networks*. In *Advances in Neural Information Processing Systems* (NeurIPS 2023), Spotlight presentation.  
[https://jwcho5576.github.io/spinn.github.io/](https://jwcho5576.github.io/spinn.github.io/)

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. *Journal of Computational Physics*, 378, 686–707. 

3. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). *DeepXDE: A deep learning library for solving differential equations*. *SIAM Review*, 63(1), 208–228. 

4. Sifan Wang, Shyam Sankaran, and Paris Perdikaris. Respecting causality is all you need for training physics-informed neural networks. arXiv preprint arXiv:2203.07404, 2022.