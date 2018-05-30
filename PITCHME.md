# GPU Programming 101

### Doing parallel programming is really easy, as long as you don't need it to run fast

---

## What is a GPU?

>A graphics processor unit (GPU), is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display. GPUs are used in embedded systems, mobile phones, personal computers, workstations, and game consoles.

--- 

## What is a GPU?

>Modern GPUs are very efficient at manipulating computer graphics and image processing, and their highly parallel structure makes them more efficient than general-purpose CPUs for algorithms where the processing of large blocks of data is done in parallel.

---

## What is a GPU?
>In a personal computer, a GPU can be present on a video card, or it can be embedded on the motherboard or—in certain CPUs—on the CPU die.

---

// images

---

## What is a CPU?
@ul
- small # of complex, independent cores
- hyper-threading
- cache hierarchy
- may have an _integrated_ GPU
- branch prediction & speculative execution
- SIMD-ification
- ILP
- OoO execution & register renaming
@ulend

---

## What is a CPU?
> Not only are CPUs good at running code, but they are good at running _bad_ code.

---

## What is a CPU?
@ul
- Moore's law - it's about transistor count, not speed
- a "wall" is hit around 2005
- limited memory bandwidth
- power consumption & complexity
@ulend

---

## What is GPGPU?

- using GPUs for general-purpose programming
- gets mainstream after 2008
- different than GPU programming for games (even the architecture shows)

---

![But why?](assets/why.webp)

---

![GFLOPs](assets/flops.png)

---

![Comparison](assets/arch-comp1.png)

---

- GPUs are well-suited for data-parallel computations with high arithmetic intensity
- (this is the ratio of arithmetic operations to memory operations)
- => lower requirement for sophisticated flow control
- => memory access latency can be hidden with calculations instead of big data caches
- optimised for simple tasks & throughput

--- 

## Some terminology
@ul
- latency: how fast can a car go on this road
- bandwidth: how many cars at time
- throughput: how many cars can per hour
@ulend
- what are CPUs optimised for?

---

### Performance = Parallelism
### Efficiency = Locality

---

### The GPU is a huge SIMD machine for embarassingly parallel tasks - like CG/ML/RT

---

## Programming model
- Cache hierarchy & explicit memory labeling
- Each SIMD lane is a different thread (~SIMT)
- Write the program as if only **one** SIMD lane will execute it
- Run it on **thousands** of threads simultaneously
- Have each thread operate on a different piece of data
- Synchronization is limited => don't have to worry about it
- Tasks that require synchronization are harder (or impossible)

---

## GPGPU APIs

---


