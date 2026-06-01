`default_nettype none
// -----------------------------------------------------------------------------
// tb_wnn_q8p8.v  (Verilog testbench for tt_um_wnn_q8p8)
// Comprehensive testbench: 75 test vectors, structured output, assertions.
//
// Key differences from tt_um_wnn (SEM20) reference testbench:
//   - DUT is tt_um_wnn_q8p8 (pure Q8.8, no SEM20 encoding)
//   - Pipeline latency: 24 cycles (1+16+1+1+3+1+1)
//   - exp modelled via 64-entry LUT + linear interpolation (Q8.8)
//   - Tolerance: 8% relative + 0.15 absolute floor (LUT quantisation noise)
//   - busy_counter is 5-bit; assertions updated accordingly
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb;

    // -------------------------------------------------------------------------
    // DUT connections  (Tiny Tapeout standard interface)
    // -------------------------------------------------------------------------
    reg        clk, rst_n, ena;
    reg  [7:0] ui_in;
    wire [7:0] uo_out;
    reg  [7:0] uio_in;
    wire [7:0] uio_out;
    wire [7:0] uio_oe;

    // Convenience aliases onto the TT byte buses
    reg        cfg_serial, cfg_valid, cfg_load;
    reg  [1:0] cfg_param;
    reg        x_serial, x_valid;
    wire       sum_serial, sum_valid, ready;

    always @* begin
        ui_in       = 8'b0;
        ui_in[0]    = cfg_serial;
        ui_in[1]    = cfg_valid;
        ui_in[2]    = cfg_load;
        ui_in[4:3]  = cfg_param;
        ui_in[5]    = x_serial;
        ui_in[6]    = x_valid;
    end

    assign sum_serial = uo_out[0];
    assign sum_valid  = uo_out[1];
    assign ready      = uo_out[2];

    always @* begin
        uio_in = 8'b0;
        ena    = 1'b1;
    end

    tt_um_wnn_q8p8 dut (
        .ui_in   (ui_in),
        .uo_out  (uo_out),
        .uio_in  (uio_in),
        .uio_out (uio_out),
        .uio_oe  (uio_oe),
        .ena     (ena),
        .clk     (clk),
        .rst_n   (rst_n)
    );

    // -------------------------------------------------------------------------
    // Clock: 10 ns period
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Golden-model storage (mirrors the DUT config register file)
    // -------------------------------------------------------------------------
    real w_arr[0:0], t_arr[0:0], d_arr[0:0];

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------
    function automatic real q8p8_to_real;
        input signed [15:0] v;
        begin
            q8p8_to_real = $itor(v) / 256.0;
        end
    endfunction

    function automatic signed [15:0] real_to_q8p8;
        input real r;
        real t;
        begin
            t = r * 256.0;
            if      (t >  32767.0) real_to_q8p8 = 16'h7FFF;
            else if (t < -32768.0) real_to_q8p8 = 16'h8000;
            else                   real_to_q8p8 = $rtoi(t);
        end
    endfunction

    function automatic real abs_real;
        input real v;
        begin
            abs_real = (v >= 0.0) ? v : -v;
        end
    endfunction

    // Golden WNN neuron (Q8.8 hardware uses LUT exp; use $exp here for reference)
    // w * z * exp(-0.5*z^2),  z = (x - t) / d
    function automatic real neuron_golden;
        input real x, w, t, d;
        real z, ev;
        begin
            if (abs_real(d) < 1e-9)
                neuron_golden = 0.0;
            else begin
                z  = (x - t) / d;
                ev = $exp(-0.5 * z * z);
                neuron_golden = w * z * ev;
            end
        end
    endfunction

    // Saturate a real golden value to the Q8.8 representable range
    function automatic real saturate_q8p8;
        input real v;
        begin
            if      (v >  127.99609375) saturate_q8p8 =  127.99609375;
            else if (v < -128.0)        saturate_q8p8 = -128.0;
            else                        saturate_q8p8 =  v;
        end
    endfunction

    function automatic real sum_golden;
        input real x;
        real s;
        integer k;
        begin
            s = 0.0;
            for (k = 0; k < 1; k = k + 1)
                s = s + neuron_golden(x, w_arr[k], t_arr[k], d_arr[k]);
            sum_golden = saturate_q8p8(s);
        end
    endfunction

    // -------------------------------------------------------------------------
    // Bus tasks
    // -------------------------------------------------------------------------
    task automatic send_cfg_word;
        input [15:0] data;
        integer i;
        begin
            for (i = 0; i < 16; i = i + 1) begin
                cfg_serial <= data[i];
                cfg_valid  <= 1'b1;
                @(posedge clk);
            end
            cfg_valid <= 1'b0;
            @(posedge clk);
        end
    endtask

    task automatic load_cfg_param;
        input [1:0]  param;
        input [15:0] value;
        begin
            send_cfg_word(value);
            cfg_param <= param;
            cfg_load  <= 1'b1;
            @(posedge clk);
            cfg_load  <= 1'b0;
            @(posedge clk);
        end
    endtask

    // Configure w, t, d for the single neuron and update the golden arrays
    task automatic load_neuron;
        input integer n;
        input real    w, t_val, d;
        begin
            load_cfg_param(2'b00, real_to_q8p8(w));
            load_cfg_param(2'b01, real_to_q8p8(t_val));
            load_cfg_param(2'b10, real_to_q8p8(d));
            w_arr[n] = w;
            t_arr[n] = t_val;
            d_arr[n] = d;
        end
    endtask

    // send_x: wait for ready (registered), then clock 16 bits LSB-first
    task automatic send_x;
        input [15:0] data;
        integer i;
        begin
            while (!ready) @(posedge clk);
            for (i = 0; i < 16; i = i + 1) begin
                x_serial <= data[i];
                x_valid  <= 1'b1;
                @(posedge clk);
            end
            x_valid <= 1'b0;
        end
    endtask

    // capture_sum: wait for sum_valid, read 16 bits LSB-first
    task automatic capture_sum;
        output [15:0] data;
        integer i;
        begin
            while (!sum_valid) @(posedge clk);
            data[0] = sum_serial;           // bit 0: sampled as sum_valid rises
            for (i = 1; i < 16; i = i + 1) begin
                @(posedge clk);
                data[i] = sum_serial;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    integer pass_count, fail_count, test_num;

    task automatic run_test;
        input real  x_real;
        input [7:0] grp;
        reg [15:0]  x_q88, hw_raw;
        real        hw_real, gld, diff, tol;
        reg [31:0]  verdict;
        begin
            x_q88 = real_to_q8p8(x_real);
            gld   = sum_golden(x_real);
            // 8% relative + 0.15 absolute floor
            // (LUT has 64 entries / linear interpolation adds ~0.5 LSB error;
            //  two Q8.8 multiplies add further rounding; d-divider is 14-bit)
            tol   = abs_real(gld) * 0.08 + 0.15;

            send_x(x_q88);
            capture_sum(hw_raw);

            hw_real = q8p8_to_real(hw_raw);
            diff    = hw_real - gld;

            if (abs_real(diff) <= tol) begin
                verdict    = "PASS";
                pass_count = pass_count + 1;
            end else begin
                verdict    = "FAIL";
                fail_count = fail_count + 1;
            end

            $display(
                "[TEST %2d/75][GRP %s] x_in=%11.5f (0x%04h) | golden=%9.4f | hw=%9.4f | diff=%8.4f | tol=%7.4f | %s",
                test_num, grp,
                x_real, x_q88,
                gld, hw_real,
                diff, tol,
                verdict
            );
            test_num = test_num + 1;
        end
    endtask

    // -------------------------------------------------------------------------
    // Main test body
    // -------------------------------------------------------------------------
    initial begin
        integer seed_a, seed_b;
        real    rnd_val;
        integer r;

        // --- reset & default drive values ------------------------------------
        rst_n      = 0;
        cfg_serial = 0; cfg_valid = 0; cfg_load = 0;
        cfg_param  = 0;
        x_serial   = 0;
        x_valid    = 0;
        pass_count = 0; fail_count = 0; test_num = 1;

        repeat(12) @(posedge clk);
        rst_n = 1;
        repeat(3)  @(posedge clk);

        // =====================================================================
        // PHASE 1 - INITIAL CONFIGURATION  (1 neuron, index 0)
        // =====================================================================
        $display("\n================================================================");
        $display("  WNN Q8.8 Single-Neuron Top - Comprehensive Testbench");
        $display("  75 test vectors | 1 neuron | pipeline ~24 cyc | sim clk 10 ns");
        $display("================================================================");
        $display("\n--- PHASE 1: Configuring 1 neuron ---");

        load_neuron(0, 1.0, 0.0, 0.80);
        $display("  -> neuron 0 loaded (w=1.0, t=0.0, d=0.80).\n");

        // =====================================================================
        // GROUP A - Standard functional (20 tests)
        // =====================================================================
        $display("--- GROUP A: Standard functional values ---");
        run_test(   0.0,   "A");
        run_test(   0.5,   "A");
        run_test(  -0.5,   "A");
        run_test(   1.0,   "A");
        run_test(  -1.0,   "A");
        run_test(   2.0,   "A");
        run_test(  -2.0,   "A");
        run_test(   3.0,   "A");
        run_test(  -3.0,   "A");
        run_test(   5.0,   "A");
        run_test(  -5.0,   "A");
        run_test(  10.0,   "A");
        run_test( -10.0,   "A");
        run_test(  20.0,   "A");
        run_test( -20.0,   "A");
        run_test(  50.0,   "A");
        run_test( -50.0,   "A");
        run_test(  75.0,   "A");
        run_test( -75.0,   "A");
        run_test(   0.25,  "A");

        // =====================================================================
        // GROUP B - Exact Q8.8 boundary values (8 tests)
        // =====================================================================
        $display("\n--- GROUP B: Q8.8 format boundary values ---");
        run_test( 127.99609375, "B");
        run_test(-128.0,        "B");
        run_test(   0.00390625, "B");
        run_test(  -0.00390625, "B");
        run_test(   1.0,        "B");
        run_test(  -1.0,        "B");
        run_test(   0.5,        "B");
        run_test(  -0.5,        "B");

        // =====================================================================
        // GROUP C - Near-zero / tiny inputs (6 tests)
        // =====================================================================
        $display("\n--- GROUP C: Near-zero inputs ---");
        run_test(  0.001,      "C");
        run_test( -0.001,      "C");
        run_test(  0.01,       "C");
        run_test( -0.01,       "C");
        run_test(  0.0078125,  "C");
        run_test( -0.0078125,  "C");

        // =====================================================================
        // GROUP D - Saturation boundary (6 tests)
        // =====================================================================
        $display("\n--- GROUP D: Saturation boundary ---");
        run_test( 100.0,  "D");
        run_test(-100.0,  "D");
        run_test( 120.0,  "D");
        run_test(-120.0,  "D");
        run_test( 126.5,  "D");
        run_test(-126.5,  "D");

        // =====================================================================
        // GROUP E - Mathematical constants (6 tests)
        // =====================================================================
        $display("\n--- GROUP E: Mathematical constants ---");
        run_test(  3.14159265, "E");
        run_test( -3.14159265, "E");
        run_test(  2.71828182, "E");
        run_test( -2.71828182, "E");
        run_test(  1.41421356, "E");
        run_test( -1.41421356, "E");

        // =====================================================================
        // GROUP F - Random narrow [-5.0, 5.0] (12 tests, fixed seed = 42)
        // =====================================================================
        $display("\n--- GROUP F: Random narrow [-5, +5] (seed=42) ---");
        seed_a = 42;
        for (r = 0; r < 12; r = r + 1) begin
            rnd_val = ($random(seed_a) % 10001) / 1000.0 - 5.0;
            run_test(rnd_val, "F");
        end

        // =====================================================================
        // GROUP G - Random wide [-50.0, 50.0] (12 tests, fixed seed = 137)
        // =====================================================================
        $display("\n--- GROUP G: Random wide [-50, +50] (seed=137) ---");
        seed_b = 137;
        for (r = 0; r < 12; r = r + 1) begin
            rnd_val = ($random(seed_b) % 100001) / 1000.0 - 50.0;
            run_test(rnd_val, "G");
        end

        // =====================================================================
        // PHASE 2 - RECONFIGURATION
        // =====================================================================
        $display("\n--- PHASE 2: Reconfiguring neuron 0 to w=1, t=0, d=1 ---");
        load_neuron(0, 1.0, 0.0, 1.0);
        $display("  -> Reconfiguration complete.\n");

        // =====================================================================
        // GROUP H - Post-reconfiguration verification (5 tests)
        // =====================================================================
        $display("--- GROUP H: Post-reconfiguration (w=1, t=0, d=1) ---");
        run_test(  0.0,  "H");
        run_test(  1.0,  "H");
        run_test( -1.0,  "H");
        run_test(  2.0,  "H");
        run_test( -2.0,  "H");

        // =====================================================================
        // SUMMARY
        // =====================================================================
        $display("\n================================================================");
        $display("  RESULTS  |  PASS: %2d  |  FAIL: %2d  |  TOTAL: %2d", pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  STATUS   |  *** ALL TESTS PASSED ***");
        else
            $display("  STATUS   |  *** %0d FAILURE(S) - inspect diff/tol above ***", fail_count);
        $display("================================================================\n");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Simulation timeout
    // -------------------------------------------------------------------------
    initial begin
        #50_000_000;
        $display("TIMEOUT at %0t ns - simulation exceeded 50 ms budget.", $time);
        $finish;
    end

    // -------------------------------------------------------------------------
    // Concurrent assertion checks (6 properties, mirrored from reference TB)
    // -------------------------------------------------------------------------

    // A: ready and x_latch_valid must never overlap
    always @(posedge clk) begin
        if (rst_n) begin
            if (ready && dut.x_latch_valid)
                $display("Error: [ASSERT A] ready & x_latch_valid both high at %0t ns", $time);
        end
    end

    // B: sum_valid must equal shift_active
    always @(posedge clk) begin
        if (rst_n) begin
            if (sum_valid && !dut.shift_active)
                $display("Error: [ASSERT B] sum_valid high but shift_active low at %0t ns", $time);
        end
    end

    // C: ready must imply pipeline not busy (busy_counter is 5-bit in q8p8 design)
    always @(posedge clk) begin
        if (rst_n) begin
            if (ready && (dut.busy_counter != 5'd0))
                $display("Error: [ASSERT C] ready high but busy_counter is non-zero at %0t ns", $time);
        end
    end

    // D: ready must imply output serialiser idle
    always @(posedge clk) begin
        if (rst_n) begin
            if (ready && dut.shift_active)
                $display("Error: [ASSERT D] ready & shift_active both high at %0t ns", $time);
        end
    end

    // E: sum_bit_cnt stays in [0,15]
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.sum_bit_cnt > 4'd15)
                $display("Error: [ASSERT E] sum_bit_cnt > 15 at %0t ns", $time);
        end
    end

    // F: x_bit_cnt stays in [0,15]
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.x_bit_cnt > 4'd15)
                $display("Error: [ASSERT F] x_bit_cnt > 15 at %0t ns", $time);
        end
    end

endmodule
