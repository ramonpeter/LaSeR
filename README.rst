========================
Latent Space Refinement 
========================

This repo contains the code for the **L**\ atente **S**\ pace **R**\ efinement (LaSeR) protocol to improve
any generative model by modifying the latent space distribution. Using the DCTR and DDLS idea, we can
derive latent space weights and thus define an optimized latent space distribution.
In order to sample from this distribution we sugggest the following methods

1. Hamilton MCMC: This is a more advanced MCMC method (compared to pure langevin dynamics) and guarantees sampling from the correct prior while keeping the autocorrelation small.
2. Weighted Flow: We train a Flow on the weighted latent space to produce the corresponding unweighted but deformed latent space distribution.
3. Weighted GAN: The same can in principle also be done using a GAN. We will however show that in most cases, Flow networks will perform better.

Moreover, we will give the necessary mathematical requirements in order to make this refinement work.
This will also explain why Flow networks fail in many situations and why GANs do better in reproducing
complex topologies.

Background
~~~~~~~~~~~

Deep neural networks can be used to reweight high-dimensional phase spaces by repurposing classifiers as reweighting functions. 
This was demonstrated in:

- **D**\ eep neural networks using **C**\ lassification for **T**\ uning and **R**\ eweighting (DCTR, pronounced “doctor”): https://arxiv.org/abs/1907.08209)

- DCTRGAN: https://arxiv.org/abs/2009.03796. 
  
It further has already been shown in:

- DDLS-GAN: https://arxiv.org/abs/2003.06060

that these weights can be backpropagated into the latent space of a GAN to modify the proposal function of the latent space.
In order to sample from this modified prior a simple MCMC method (Langevin dynamics) was used.
