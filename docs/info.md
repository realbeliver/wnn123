## How it works

This project implements a single-neuron hardware accelerator for a Weightless Neural Network (WNN). It calculates a localized radial basis function commonly used in specific machine learning models. 

Mathematically, the core computes the following continuous function:
$$y = w \cdot z \cdot \exp(-0.5 \cdot z^2)$$
where $$z = \frac{x - t}{d}$$

To balance hardware efficiency with dynamic range, the accelerator features a unique internal data path:
* **External Interface:** All inputs and outputs (x, w, t, d, and y) use a **16-bit Q8.8** signed fixed-point format.
* **Internal Processing:** Data is converted into a custom **SEM20** floating-point format (1 sign bit, 6 exponent bits, 13 mantissa bits). 
* **Math Pipeline:** The design features a fully pipelined architecture containing a 6-cycle multiplier, a 16-cycle restoring divider, and a hybrid 4th-order Taylor series / Look-Up Table (LUT) engine to calculate the complex exponential function.
* **Latency:** The full pipeline takes **117 cycles** (16 cycles to shift in, 100 cycles for the core neuron calculation, and 1 cycle to latch).

## How to test

Because Tiny Tapeout has a limited pin count, data is passed into the neuron serially. 

**1. Configuration Phase:**
You must first configure the neuron's parameters: weight ($w$), threshold ($t$), and divisor ($d$).
* Shift 16 bits of Q8.8 data (LSB first) into **`ui_in[0]`** (`cfg_serial`), setting **`ui_in[1]`** (`cfg_valid`) high for each bit.
* Select the target parameter using **`ui_in[4:3]`** (`cfg_param`): **00** for $w$, **01** for $t$, **10** for $d$.
* Pulse **`ui_in[2]`** (`cfg_load`) high for one cycle to latch the shifted data into the chosen register.

**2. Inference Phase:**
* Monitor **`uo_out[2]`** (`ready`). When it goes high, the pipeline is idle.
* Shift a 16-bit Q8.8 input value ($x$) into **`ui_in[5]`** (`x_serial`), setting **`ui_in[6]`** (`x_valid`) high for each bit.

**3. Output Phase:**
* Monitor **`uo_out[1]`** (`sum_valid`). When it goes high, the calculation is complete.
* Read the 16-bit Q8.8 output result serially from **`uo_out[0]`** (`sum_serial`) over the next 16 clock cycles (LSB first).

## External hardware

No specialized external hardware is strictly required. However, because the interface relies on precise serial bit-banging and data format conversion (standard floats to Q8.8 format), connecting the Tiny Tapeout board to a microcontroller (such as a Raspberry Pi Pico or Arduino) or an FPGA is highly recommended to drive the test vectors and capture the results.
