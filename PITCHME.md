# GPU Programming 101

### Doing parallel programming is really easy, as long as you don't need it to run fast

---

## What is a GPU?

> A graphics processor unit (GPU), is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display. GPUs are used in embedded systems, mobile phones, personal computers, workstations, and game consoles.

--- 

## What is a GPU?

> Modern GPUs are very efficient at manipulating computer graphics and image processing, and their highly parallel structure makes them more efficient than general-purpose CPUs for algorithms where the processing of large blocks of data is done in parallel. In a personal computer, a GPU can be present on a video card, or it can be embedded on the motherboard or—in certain CPUs—on the CPU die.

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
