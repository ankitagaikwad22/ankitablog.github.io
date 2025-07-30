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
• Residual loss using differential equation. 
• Losses due to initial and boundary conditions. 
These are enforced using automatic differentiation (often reverse-mode) and trained with standard gradient descent. These models are mesh-free, data-efficient (unsupervised), and can solve both forward and inverse problems. 

However, the cost of analyzing the network and computing gradients rises considerably as dimensionality increases. To solve PDEs with finer grids or higher dimensions, PINNs require 
• a large number of collocation points. 
• Cannot handle large training sets on single gpu
• they face high computational and memory costs 
• they suffer from slow convergence.
These scalability limits are well-documented [1, 3]. SPINNs confront things head on. 


# Introducing SPINNS


So to overcome all the problems researchers comes with a solution i.e SPINNS- which stands for Separable Physics-Informed Neural Networks.
It's a new way to structure and train PINNs that:
•	Handles multi-dimensional PDEs more efficiently
•	Allows using more collocation points (>10 million!) even on a single GPU
•	Is much faster and more accurate than traditional PINNs

To understand why SPINN is so quick, we must first consider how derivatives are produced in deep learning – especially, by automatic differentiation. There are two major modes:

Reverse-Mode AD
Reverse-Mode AD (used in backpropagation) is ideal for functions with multiple inputs and a single output (e.g., loss functions). It operates by first performing a forward pass to compute outputs, followed by a backward pass to calculate gradients with respect to inputs. This is useful for training neural networks that require the gradient of a scalar loss with respect to all model parameters.
Forward-Mode AD:
Forward-Mode AD works best for functions with few inputs and many outputs, such as computing derivatives of solutions with respect to input coordinates in PDEs. Forward-mode AD computes these derivatives immediately during the forward pass, making it more efficient when assessing PDE residuals.

In SPINN, the number of input variables (coordinates) is minimal (for example, time and space), but the number of outputs (solution evaluations at various points) is huge, making forward-mode AD the best fit.



# SPINN Architecture




SPINNs solve d-dimensional PDEs using a separated neural network design in which each input dimension is processed by a distinct MLP. Each of these networks converts a scalar into a feature vector. The final scalar output is then calculated by performing an element-wise product on these vectors and summing over the shared rank dimension. 
Mathematically, the output of the model is:
$$
\hat{u}(x) = \sum_{j=1}^{r} f_j^{(\theta^{(1)})}(x_1) f_j^{(\theta^{(2)})}(x_2) \cdots f_j^{(\theta^{(d)})}(x_d)
$$

 
The model produces a CP-decomposition (CANDECOMP/PARAFAC) of a tensor that represents the solution. Each function is a small axis-specific MLP, and the network learns a solution with a low rank. 
The architecture enables rapid evaluation and training on structured input grids. 
The process involves sampling 1D points on each axis, translating them to a -dimensional vector using MLPs, and generating an element-wise product and summation over shared rank as output. 
• The authors use tanh activations for all MLPs. 
This structure is very parallelizable and efficient as the outputs of each MLP can be reused across several axis inputs. 
• Applying forward-mode AD to 1D MLPs reduces Jacobian computation costs dramatically. 
The whole discretized solution tensor can be evaluated in batches utilizing structured Cartesian product sampling of coordinates, making SPINNs appropriate for large-scale problems with low overhead.

# Detailed Experiments and Results



The authors of the SPINN paper performed rigorous experiments across a wide spectrum of PDEs, ranging from simple linear systems like diffusion to complex nonlinear equations such as Navier-Stokes. Each experiment highlights different advantages of SPINN — computational speed, memory efficiency, and accuracy scaling with resolution. Below is a breakdown of the major experiments and their findings:

## Hardware & Setup

All experiments were conducted on a single NVIDIA RTX 3090 GPU with 24GB of memory. The maximum resolution for standard PINNs was limited by GPU memory, while SPINN could scale up to significantly more collocation points thanks to its low memory footprint.
Each model was evaluated using:
•	Relative error
•	Wall-clock training time
•	GPU memory usage
•	Rank sensitivity (for SPINNs)

## Visual Summary of Results

The figure below summarizes SPINN’s performance across several PDE benchmarks:
![SPINN Results](D:\BLOG\ankitablog.github.io\images/Blog_images/Results.png)

### Diffusion Equation
The Diffusion equation is a foundational PDE in physics, used to model heat conduction and particle diffusion.

•   In this experiment, SPINN-mod consistently achieves the best accuracy.
•   At a resolution of 128³ collocation points, SPINN-mod is:
- •   52× faster than PINN-mod
- •   29× more memory-efficient
•   Even when using the same number of points, SPINN delivers more accurate results with far less computational cost.
### Helmholtz Equation
Helmholtz equations appear in problems involving wave propagation and vibrations, and are notoriously hard to solve due to high-frequency components.

•   Traditional PINNs struggle here due to stiffness in gradient flow.
•   SPINN-mod, on the other hand, handles the equation gracefully:
- •   Achieving a 62× speedup
- •   Using 29× less GPU memory

### Klein-Gordon Equation (2+1D and 3+1D)
The Klein-Gordon equation, important in quantum field theory, is tested in two forms:

**(2+1)D Klein-Gordon**
•   SPINN-mod outperforms all baselines, with:
- •   Lowest relative L₂ error
- •   62× faster runtime than PINN-mod
- •   29× memory savings
•   As the number of collocation points increases, SPINN scales better than PINNs both computationally and numerically, enabling larger, more stable models.

**(3+1)D Klein-Gordon**
To really push the limits, the team tested SPINN in 4D by adding an extra spatial axis:

•   PINNs (even with a modified MLP) hit memory limits at just 18⁴ points.
•   SPINN, however, processes up to 64⁴ collocation points — that's a 160× improvement in capacity!
•   This shows SPINN’s incredible scalability for high-dimensional scientific computing.

### Navier-Stokes Equation (2+1)D

Navier-Stokes governs fluid flow, and solving it accurately is central to computational physics.

•   SPINN was tested against Causal PINNs and PINN-mod, aiming to predict velocity fields and compute vorticity using forward-mode AD.
•   Results show that SPINN:
- •   Is 60× faster
- •   Uses ~25% of the memory
- •   Achieves comparable or better accuracy

•   Crucially, SPINN achieved this without even using a causal loss function — which Causal PINNs depend on.

### Navier-Stokes Equation (3+1)D
To top it off, SPINN was extended to the extremely complex (3+1)D Navier-Stokes, involving 33 derivative terms!

Even with this complexity:

- •   SPINN trained in under 30 minutes
- •   Reached a relative error of 1.9e-3
- •   Used less than 3GB of GPU memory at 32⁴ collocation points

This cements SPINN as a highly scalable, fast, and accurate framework for chaotic and nonlinear PDEs in very high dimensions
