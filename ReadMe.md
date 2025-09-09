Hebbian RNN with Neuromodulation

This repository contains a basic recurrent neural network (RNN) model with Hebbian plasticity and neuromodulation, inspired by the framework described in:
Miconi, T. (2015). Reinforcement Learning Through Modulation of Spike-Timing–Dependent Synaptic Plasticity. Frontiers in Neural Circuits. link: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full 

The model serves as a base implementation for exploring how biologically inspired synaptic plasticity rules can be used in neural networks without relying on backpropagation.

What the Model Does: 
At its core, this is a neural network where the weights update themselves during activity, instead of being fixed after training. Learning happens in real time, guided by:
Hebbian Plasticity — “neurons that fire together, wire together.”

If two neurons are active at the same time, their connection strengthens.
If not, the connection weakens.

Layer-Wide Updates — The Hebbian rule applies across the whole hidden layer, not just one pair of neurons.
Temporal Memory — The model accumulates past interactions into a Hebbian trace, giving it memory of previous activations.
Bounded Growth — A decay mechanism prevents runaway growth of synaptic weights.
Neuromodulation — A global signal (like dopamine in the brain) scales plasticity.
Stronger when reward/feedback is present.
Weaker when there’s no important signal.
Together, these ingredients allow the RNN to adapt trial-by-trial, similar to how animals learn from experience.

Why This Is Biologically Inspired
Mimics synaptic plasticity observed in real neurons.
Learning happens locally and continuously, not through global backpropagation.
Uses neuromodulatory signals (reward/punishment) to influence how strongly connections change.
Provides a simple framework for credit assignment in recurrent circuits.

Mathematical Framework
The learning rule builds on Hebb’s postulate and extends it to recurrent networks with modulation:
Basic Hebbian Update
The Full Mathematical Framework is given in the Folder: Mathematical Framework 

Code Overview
HebbianRNN — Core PyTorch module implementing an RNN with Hebbian plasticity and neuromodulation.
Training Loop — Demonstrates toy sequence prediction task, using synthetic inputs and a modulatory reward signal.
Visualization — Tracks mean-squared error (MSE) loss over training epochs.

 How to Run

Clone the repo:
git clone https://github.com/Ria-Chaudhry/Hebbian-RNN-Model-.git
cd Hebbian-RNN-Model-

Run the demo training script:
python hebbian_rnn.py
You’ll see:
Loss curve over epochs
Sample input/target/predicted sequences printed after training

 Next Steps
This is a base model for exploration. Potential extensions include:
More realistic spiking neuron dynamics.
Task-specific modulation signals (beyond synthetic reward).
Applying the model to reinforcement learning benchmarks.
Comparing performance with standard backprop-based RNNs.

 Disclaimer
This code is for educational and exploratory purposes.
The mathematical framework is adapted from Miconi (2015) and related works.
Parameters and implementation are simplified.
This is not intended as a fully validated biological simulation.