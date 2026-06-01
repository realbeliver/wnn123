import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import math
import random

def real_to_q8p8(r):
    """Converts a real float value into a signed 16-bit Q8.8 integer."""
    t = int(r * 256.0)
    if t > 32767: 
        return 0x7FFF
    if t < -32768: 
        return 0x8000
    return t & 0xFFFF

def q8p8_to_real(v):
    """Converts a signed 16-bit Q8.8 integer back into a real float value."""
    if v & 0x8000:
        v -= 0x10000
    return v / 256.0

def neuron_golden(x, w, t, d):
    """Golden reference calculation matching the hardware's target function."""
    if abs(d) < 1e-9:
        return 0.0
    z = (x - t) / d
    ev = math.exp(-0.5 * z * z)
    val = w * z * ev
    
    # Saturation bounds for Q8.8 range
    if val > 127.99609375: 
        return 127.99609375
    if val < -128.0: 
        return -128.0
    return val

def set_ui_in(cfg_serial=0, cfg_valid=0, cfg_load=0, cfg_param=0, x_serial=0, x_valid=0):
    """Helper to assemble individual signals into the 8-bit ui_in bus."""
    return (cfg_serial & 1) | \
           ((cfg_valid & 1) << 1) | \
           ((cfg_load & 1) << 2) | \
           ((cfg_param & 3) << 3) | \
           ((x_serial & 1) << 5) | \
           ((x_valid & 1) << 6)

async def load_cfg_param(dut, param, value):
    """Serializes a 16-bit configuration word and pulses cfg_load."""
    # Send 16 bits LSB first
    for i in range(16):
        bit = (value >> i) & 1
        dut.ui_in.value = set_ui_in(cfg_serial=bit, cfg_valid=1, cfg_param=param)
        await ClockCycles(dut.clk, 1)
        
    # Deassert valid
    dut.ui_in.value = set_ui_in(cfg_valid=0, cfg_param=param)
    await ClockCycles(dut.clk, 1)
    
    # Pulse load high for 1 cycle
    dut.ui_in.value = set_ui_in(cfg_load=1, cfg_param=param)
    await ClockCycles(dut.clk, 1)
    dut.ui_in.value = set_ui_in(cfg_load=0, cfg_param=param)
    await ClockCycles(dut.clk, 1)

async def run_neuron_test(dut, x_real, w, t, d, grp_name, test_idx):
    """Executes an inference test cycle and monitors compliance with tolerance floors."""
    x_q88 = real_to_q8p8(x_real)
    gld = neuron_golden(x_real, w, t, d)
    
    # Match Verilog tolerance: 8% relative error + 0.15 absolute floor
    tol = abs(gld) * 0.08 + 0.15
    
    # Wait until the neuron core is ready to accept a new input vector
    while dut.uo_out[2].value == 0:
        await ClockCycles(dut.clk, 1)
        
    # Stream in the 16-bit x vector LSB-first
    for i in range(16):
        bit = (x_q88 >> i) & 1
        dut.ui_in.value = set_ui_in(x_serial=bit, x_valid=1)
        await ClockCycles(dut.clk, 1)
    dut.ui_in.value = set_ui_in(x_valid=0)
    
    # Wait for the design to finish pipeline processing and raise sum_valid
    while dut.uo_out[1].value == 0:
        await ClockCycles(dut.clk, 1)
        
    # Sample bit 0 immediately on the rising edge of sum_valid
    hw_raw = 0
    hw_raw |= (dut.uo_out[0].value & 1)
    
    # Capture the remaining 15 bits of the computed response serially
    for i in range(1, 16):
        await ClockCycles(dut.clk, 1)
        hw_raw |= ((dut.uo_out[0].value & 1) << i)
        
    hw_real = q8p8_to_real(hw_raw)
    diff = hw_real - gld
    
    verdict = "PASS" if abs(diff) <= tol else "FAIL"
    dut._log.info(
        f"[TEST {test_idx:2d}/75][GRP {grp_name}] x_in={x_real:11.5f} (0x{x_q88:04h}) | "
        f"golden={gld:9.4f} | hw={hw_real:9.4f} | diff={diff:8.4f} | tol={tol:7.4f} | {verdict}"
    )
    
    assert abs(diff) <= tol, f"Math mismatch at test {test_idx}! Diff is {diff}, allowed tolerance is {tol}."

@cocotb.test()
async def test_wnn_q8p8(dut):
    dut._log.info("================================================================")
    dut._log.info("  WNN Q8.8 Single-Neuron Top - Cocotb Testbench Conversion")
    dut._log.info("  75 test vectors | 1 neuron | pipeline ~24 cyc")
    dut._log.info("================================================================")
    
    # Drive clock system
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    
    # Drive initial reset and interface assignments
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    
    await ClockCycles(dut.clk, 12)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 3)
    
    test_idx = 1
    
    # =====================================================================
    # PHASE 1 - CONFIGURATION (w=1.0, t=0.0, d=0.80)
    # =====================================================================
    w, t, d = 1.0, 0.0, 0.80
    dut._log.info(f"\n--- PHASE 1: Configuring neuron (w={w}, t={t}, d={d}) ---")
    await load_cfg_param(dut, 0, real_to_q8p8(w))
    await load_cfg_param(dut, 1, real_to_q8p8(t))
    await load_cfg_param(dut, 2, real_to_q8p8(d))
    dut._log.info("  -> Parameter stack loaded successfully.\n")
    
    # GROUP A - Standard functional values
    dut._log.info("--- GROUP A: Standard functional values ---")
    group_a = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 50.0, -50.0, 75.0, -75.0, 0.25]
    for val in group_a:
        await run_neuron_test(dut, val, w, t, d, "A", test_idx)
        test_idx += 1
        
    # GROUP B - Exact Q8.8 boundary values
    dut._log.info("\n--- GROUP B: Q8.8 format boundary values ---")
    group_b = [127.99609375, -128.0, 0.00390625, -0.00390625, 1.0, -1.0, 0.5, -0.5]
    for val in group_b:
        await run_neuron_test(dut, val, w, t, d, "B", test_idx)
        test_idx += 1
        
    # GROUP C - Near-zero / tiny inputs
    dut._log.info("\n--- GROUP C: Near-zero inputs ---")
    group_c = [0.001, -0.001, 0.01, -0.01, 0.0078125, -0.0078125]
    for val in group_c:
        await run_neuron_test(dut, val, w, t, d, "C", test_idx)
        test_idx += 1
        
    # GROUP D - Saturation boundary
    dut._log.info("\n--- GROUP D: Saturation boundary ---")
    group_d = [100.0, -100.0, 120.0, -120.0, 126.5, -126.5]
    for val in group_d:
        await run_neuron_test(dut, val, w, t, d, "D", test_idx)
        test_idx += 1
        
    # GROUP E - Mathematical constants
    dut._log.info("\n--- GROUP E: Mathematical constants ---")
    group_e = [3.14159265, -3.14159265, 2.71828182, -2.71828182, 1.41421356, -1.41421356]
    for val in group_e:
        await run_neuron_test(dut, val, w, t, d, "E", test_idx)
        test_idx += 1
        
    # GROUP F - Deterministic Random narrow [-5.0, 5.0] (seed = 42)
    dut._log.info("\n--- GROUP F: Random narrow [-5, +5] (seed=42) ---")
    random.seed(42)
    for _ in range(12):
        val = random.uniform(-5.0, 5.0)
        await run_neuron_test(dut, val, w, t, d, "F", test_idx)
        test_idx += 1
        
    # GROUP G - Deterministic Random wide [-50.0, 50.0] (seed = 137)
    dut._log.info("\n--- GROUP G: Random wide [-50, +50] (seed=137) ---")
    random.seed(137)
    for _ in range(12):
        val = random.uniform(-50.0, 50.0)
        await run_neuron_test(dut, val, w, t, d, "G", test_idx)
        test_idx += 1
        
    # =====================================================================
    # PHASE 2 - RECONFIGURATION (w=1.0, t=0.0, d=1.0)
    # =====================================================================
    w, t, d = 1.0, 0.0, 1.0
    dut._log.info(f"\n--- PHASE 2: Reconfiguring neuron 0 to w={w}, t={t}, d={d} ---")
    await load_cfg_param(dut, 0, real_to_q8p8(w))
    await load_cfg_param(dut, 1, real_to_q8p8(t))
    await load_cfg_param(dut, 2, real_to_q8p8(d))
    dut._log.info("  -> Reconfiguration complete.\n")
    
    # GROUP H - Post-reconfiguration verification
    dut._log.info("--- GROUP H: Post-reconfiguration (w=1, t=0, d=1) ---")
    group_h = [0.0, 1.0, -1.0, 2.0, -2.0]
    for val in group_h:
        await run_neuron_test(dut, val, w, t, d, "H", test_idx)
        test_idx += 1
        
    dut._log.info("\n================================================================")
    dut._log.info("  STATUS   |  *** ALL 75 PRODUCTION TESTS PASSED ***")
    dut._log.info("================================================================")
