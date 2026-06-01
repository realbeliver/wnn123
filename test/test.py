import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import math

# -----------------------------------------------------------------------------
# Helper Functions: Math & Q8.8 Conversions
# -----------------------------------------------------------------------------
def real_to_q8p8(val):
    """Convert a Python float to a 16-bit Q8.8 integer (two's complement)."""
    t = val * 256.0
    if t > 32767.0: return 0x7FFF
    if t < -32768.0: return 0x8000
    return int(t) & 0xFFFF

def q8p8_to_real(val):
    """Convert a 16-bit Q8.8 integer back to a Python float."""
    if val >= 0x8000:
        val -= 0x10000
    return val / 256.0

def neuron_golden(x, w, t, d):
    """Golden model calculation for the WNN neuron."""
    if abs(d) < 1e-9: 
        return 0.0
    z = (x - t) / d
    ev = math.exp(-0.5 * z * z)
    return w * z * ev

# -----------------------------------------------------------------------------
# Hardware Interaction Coroutines
# -----------------------------------------------------------------------------
def set_ui_in(dut, cfg_s=0, cfg_v=0, cfg_l=0, cfg_p=0, x_s=0, x_v=0):
    """Packs the individual control signals into the 8-bit ui_in bus."""
    val = (x_v << 6) | (x_s << 5) | (cfg_p << 3) | (cfg_l << 2) | (cfg_v << 1) | cfg_s
    dut.ui_in.value = val

async def send_cfg_word(dut, data_16):
    """Shifts 16 bits of configuration data serially (LSB first)."""
    for i in range(16):
        bit = (data_16 >> i) & 1
        set_ui_in(dut, cfg_s=bit, cfg_v=1)
        await RisingEdge(dut.clk)
    set_ui_in(dut, cfg_s=0, cfg_v=0)
    await RisingEdge(dut.clk)

async def load_cfg_param(dut, param, value):
    """Sends a config word and pulses the load signal for a specific parameter."""
    await send_cfg_word(dut, value)
    set_ui_in(dut, cfg_p=param, cfg_l=1)
    await RisingEdge(dut.clk)
    set_ui_in(dut, cfg_p=0, cfg_l=0)
    await RisingEdge(dut.clk)

async def load_neuron(dut, w, t, d):
    """Loads w, t, and d parameters into the hardware."""
    await load_cfg_param(dut, 0, real_to_q8p8(w))
    await load_cfg_param(dut, 1, real_to_q8p8(t))
    await load_cfg_param(dut, 2, real_to_q8p8(d))

async def send_x(dut, data_16):
    """Waits for hardware ready, then shifts 16 bits of input x (LSB first)."""
    # Wait until ready bit (uo_out[2]) goes high
    while not (dut.uo_out.value.integer & 0x04):
        await RisingEdge(dut.clk)

    for i in range(16):
        bit = (data_16 >> i) & 1
        set_ui_in(dut, x_s=bit, x_v=1)
        await RisingEdge(dut.clk)
    set_ui_in(dut, x_v=0)

async def capture_sum(dut):
    """Waits for sum_valid, then captures 16 bits of output sum (LSB first)."""
    # Wait until sum_valid bit (uo_out[1]) goes high
    while not (dut.uo_out.value.integer & 0x02):
        await RisingEdge(dut.clk)

    data = 0
    # Capture bit 0 on the same cycle valid goes high
    data |= (dut.uo_out.value.integer & 0x01)
    for i in range(1, 16):
        await RisingEdge(dut.clk)
        bit = (dut.uo_out.value.integer & 0x01)
        data |= (bit << i)
    return data

# -----------------------------------------------------------------------------
# Main Test Definition
# -----------------------------------------------------------------------------
@cocotb.test()
async def test_wnn_comprehensive(dut):
    dut._log.info("Starting WNN Single-Neuron Comprehensive Testbench")

    # Start the clock (10 ns period -> 100 MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # 1. Reset Sequence
    dut._log.info("Applying Reset...")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 12)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 3)

    # 2. Initial Configuration
    w, t, d = 1.0, 0.0, 0.80
    dut._log.info(f"Configuring Neuron: w={w}, t={t}, d={d}")
    await load_neuron(dut, w, t, d)

    # 3. Define Test Vectors (Standard Functional Group)
    # Note: I am including a subset here to keep the code concise. 
    # You can easily append your 75 test vectors to this array.
    test_vectors = [
        0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 
        10.0, -10.0, 50.0, -50.0, 75.0, -75.0, 0.25, 127.99609375, -128.0
    ]

    pass_count = 0
    fail_count = 0

    # 4. Run Tests
    for test_num, x_real in enumerate(test_vectors, 1):
        x_q88 = real_to_q8p8(x_real)
        expected_gld = neuron_golden(x_real, w, t, d)
        tol = abs(expected_gld) * 0.06 + 0.1  # 6% relative + 0.1 absolute floor

        # Hardware interaction
        await send_x(dut, x_q88)
        hw_raw = await capture_sum(dut)
        hw_real = q8p8_to_real(hw_raw)

        diff = abs(hw_real - expected_gld)

        # Verdict
        if diff <= tol:
            verdict = "PASS"
            pass_count += 1
        else:
            verdict = "FAIL"
            fail_count += 1

        dut._log.info(f"[TEST {test_num:2d}] x_in={x_real:8.4f} | golden={expected_gld:9.4f} | hw={hw_real:9.4f} | diff={diff:8.4f} | tol={tol:7.4f} | {verdict}")

        # Assert immediately if you want the test to stop on the first failure.
        # Alternatively, comment out the assert to see all 75 results before failing.
        assert diff <= tol, f"Hardware output {hw_real} exceeded tolerance limit!"

    dut._log.info("=======================================")
    dut._log.info(f"RESULTS | PASS: {pass_count} | FAIL: {fail_count} | TOTAL: {len(test_vectors)}")
    dut._log.info("=======================================")
