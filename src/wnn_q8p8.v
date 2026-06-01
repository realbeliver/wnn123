/*
 * wnn_q8p8.v
 * Single-neuron WNN accelerator — pure Q8.8 fixed-point, no SEM20.
 *
 * Arithmetic format: Q8.8 signed, 16-bit
 *   Represents range [-128, +127.99609375] with resolution 1/256 = 0.00390625
 *   Bit 15    : sign
 *   Bits 14:8 : integer part (7 bits + sign)
 *   Bits  7:0 : fractional part
 *
 * Neuron function: output = w * z * exp(-0.5 * z^2),  z = (x - t) / d
 *
 * Pipeline stages:
 *   Stage 1 : sub    x - t          (1 cycle,  Q8.8 subtraction)
 *   Stage 2 : div    (x-t) / d      (16 cycles, restoring fixed-point divider)
 *   Stage 3 : z2     z * z          (1 cycle,  Q8.8 multiply)
 *   Stage 4 : half   0.5 * z^2      (1 cycle,  right-shift)
 *   Stage 5 : exp    exp(-0.5*z^2)  (8 cycles, LUT + linear interpolation, Q8.8 result in [0,1])
 *   Stage 6 : shape  z * exp(...)   (1 cycle,  Q8.8 multiply)
 *   Stage 7 : final  w * shape      (1 cycle,  Q8.8 multiply)
 *   Total latency: ~29 cycles from in_valid to out_valid
 *
 * Why cfg registers exist (answered at bottom of file).
 *
 * Top-level I/O: same Tiny Tapeout serial interface as original.
 *   ui_in[0]    cfg_serial   — config shift-register data (LSB first)
 *   ui_in[1]    cfg_valid    — clock enable for config shift register
 *   ui_in[2]    cfg_load     — latch shifted word into selected register
 *   ui_in[4:3]  cfg_param    — register select: 00=w, 01=t, 10=d
 *   ui_in[5]    x_serial     — inference input data (LSB first)
 *   ui_in[6]    x_valid      — clock enable for input deserializer
 *   uo_out[0]   sum_serial   — inference output data (LSB first)
 *   uo_out[1]   sum_valid    — output shift register active
 *   uo_out[2]   ready        — design ready for next input
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none
`timescale 1ns / 1ps


// =============================================================================
// q8p8_mul — signed Q8.8 * Q8.8 → Q8.8 (1 cycle)
// Full product is 32 bits (Q16.16); we keep bits [23:8] as Q8.8 with saturation.
// =============================================================================
module q8p8_mul (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output reg  signed [15:0] product,
    output reg                out_valid
);
    wire signed [31:0] full = a * b;             // Q16.16
    wire signed [15:0] result_raw = full[23:8];  // keep Q8.8 slice

    // Saturate: if high bits [31:24] are not all-same as sign of [23], saturate
    wire overflow_pos = (~full[31]) & (|full[30:24]);  // positive overflow
    wire overflow_neg =   full[31]  & (~(&full[30:24])); // negative overflow

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product   <= 16'sd0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid;
            if (overflow_pos)      product <= 16'h7FFF;
            else if (overflow_neg) product <= 16'sh8000;
            else                   product <= result_raw;
        end
    end
endmodule


// =============================================================================
// q8p8_div — signed Q8.8 / Q8.8 → Q8.8  (16-cycle restoring divider)
// Computes a/b in Q8.8.  If b==0, output is max positive/negative (saturate).
// Strategy: convert both to integer Q16 by scaling, do 16-bit restoring
// division, then re-scale result back to Q8.8.
// =============================================================================
module q8p8_div (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output reg  signed [15:0] result,
    output reg                out_valid
);
    // Pipeline stage 0: sign extraction and abs
    reg        s0_valid, s0_sign, s0_bzero;
    reg [15:0] s0_num, s0_den;   // unsigned magnitudes

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0; s0_sign <= 1'b0; s0_bzero <= 1'b0;
            s0_num <= 16'd0; s0_den <= 16'd0;
        end else begin
            s0_valid <= in_valid;
            s0_sign  <= a[15] ^ b[15];
            s0_bzero <= (b == 16'sd0);
            // abs: handle -32768 specially
            s0_num <= a[15] ? (a == 16'sh8000 ? 16'h8000 : ~a + 16'd1) : a;
            s0_den <= b[15] ? (b == 16'sh8000 ? 16'h8000 : ~b + 16'd1) : b;
        end
    end

    // Pipeline stage 1: shift numerator left by 8 to pre-scale for Q8.8 result
    // dividend = num << 8, divisor = den
    // raw_quotient is in the same scale as Q8.8
    reg        s1_valid, s1_sign, s1_bzero;
    reg [23:0] s1_dividend;
    reg [15:0] s1_divisor;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_sign <= 1'b0; s1_bzero <= 1'b0;
            s1_dividend <= 24'd0; s1_divisor <= 16'd0;
        end else begin
            s1_valid    <= s0_valid;
            s1_sign     <= s0_sign;
            s1_bzero    <= s0_bzero;
            s1_dividend <= {s0_num, 8'b0};  // Q8.8 numerator scaled by 2^8
            s1_divisor  <= s0_den;
        end
    end

    // 14-stage restoring divider (generates 14 quotient bits → Q6.8 range)
    // Each stage: R = 2*R - D; if >= 0 qbit=1 keep, else qbit=0 restore
    localparam NBITS = 14;

    reg        dv_valid [0:NBITS-1];
    reg        dv_sign  [0:NBITS-1];
    reg        dv_bzero [0:NBITS-1];
    reg [23:0] dv_R     [0:NBITS-1];
    reg [15:0] dv_D     [0:NBITS-1];
    reg [13:0] dv_Q     [0:NBITS-1];

    wire [24:0] dv0_r2   = {s1_dividend, 1'b0};           // 2*R (25 bits)
    wire [24:0] dv0_sub  = dv0_r2 - {9'b0, s1_divisor};
    wire        dv0_qbit = ~dv0_sub[24];
    wire [23:0] dv0_Rnxt = dv0_qbit ? dv0_sub[23:0] : dv0_r2[23:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[0] <= 1'b0; dv_sign[0] <= 1'b0; dv_bzero[0] <= 1'b0;
            dv_D[0] <= 16'd0; dv_R[0] <= 24'd0; dv_Q[0] <= 14'd0;
        end else begin
            dv_valid[0] <= s1_valid; dv_sign[0] <= s1_sign; dv_bzero[0] <= s1_bzero;
            dv_D[0] <= s1_divisor; dv_R[0] <= dv0_Rnxt;
            dv_Q[0] <= {13'd0, dv0_qbit};
        end
    end

    genvar gi;
    generate
        for (gi = 1; gi < NBITS; gi = gi + 1) begin : div_stages
            wire [24:0] r2   = {dv_R[gi-1], 1'b0};
            wire [24:0] sub  = r2 - {9'b0, dv_D[gi-1]};
            wire        qbit = ~sub[24];
            wire [23:0] Rnxt = qbit ? sub[23:0] : r2[23:0];

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    dv_valid[gi] <= 1'b0; dv_sign[gi] <= 1'b0; dv_bzero[gi] <= 1'b0;
                    dv_D[gi] <= 16'd0; dv_R[gi] <= 24'd0; dv_Q[gi] <= 14'd0;
                end else begin
                    dv_valid[gi] <= dv_valid[gi-1];
                    dv_sign[gi]  <= dv_sign[gi-1];
                    dv_bzero[gi] <= dv_bzero[gi-1];
                    dv_D[gi]     <= dv_D[gi-1];
                    dv_R[gi]     <= Rnxt;
                    dv_Q[gi]     <= {dv_Q[gi-1][12:0], qbit};
                end
            end
        end
    endgenerate

    // Output stage: apply sign, saturate, handle div-by-zero
    wire [13:0] raw_q   = dv_Q[NBITS-1];
    wire        sign_f  = dv_sign[NBITS-1];
    wire        bzero_f = dv_bzero[NBITS-1];

    // raw_q is a 14-bit unsigned quotient representing Q6.8 range
    // Saturate to Q8.8 (max 0x7FFF unsigned = 32767 → Q8.8 = 127.996)
    wire ovf = (raw_q[13:8] != 6'd0);  // integer part needs > 6 bits → overflow

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= 16'sd0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= dv_valid[NBITS-1];
            if (bzero_f) begin
                result <= sign_f ? 16'sh8000 : 16'h7FFF;
            end else if (ovf) begin
                result <= sign_f ? 16'sh8000 : 16'h7FFF;
            end else begin
                // raw_q[13:0] in Q6.8 fits in Q8.8; apply sign
                result <= sign_f ? -(16'sd0 + raw_q[15:0]) : {2'b00, raw_q};
            end
        end
    end
endmodule


// =============================================================================
// exp_lut_q8p8 — exp(-x) for x in Q8.8 unsigned [0, ~8.0]
// Uses 64-entry LUT sampled at x = 0.0, 0.125, 0.25, ..., 7.875
// Then does linear interpolation between adjacent entries.
// Result is Q8.8 unsigned in range (0, 1].
// Latency: 3 cycles.
// =============================================================================
module exp_lut_q8p8 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [15:0] x_abs,   // Q8.8 unsigned magnitude of input (already 0.5*z^2)
    output reg  [15:0] result,  // Q8.8 result exp(-x), unsigned
    output reg         out_valid
);
    // 64-entry LUT: exp(-k * 0.125) for k = 0..63, stored as Q8.8 unsigned
    // Q8.8 = round(exp(-k/8) * 256)
    // k=0: exp(0)=1.0 → 256=0x0100
    // k=8: exp(-1)≈0.3679 → 94=0x005E
    // etc.
    function [15:0] lut_val;
        input [5:0] k;
        begin
            case (k)
                6'd0:  lut_val = 16'd256;  // exp(-0.000) = 1.00000
                6'd1:  lut_val = 16'd234;  // exp(-0.125) = 0.88250
                6'd2:  lut_val = 16'd214;  // exp(-0.250) = 0.77880
                6'd3:  lut_val = 16'd195;  // exp(-0.375) = 0.68729
                6'd4:  lut_val = 16'd178;  // exp(-0.500) = 0.60653
                6'd5:  lut_val = 16'd162;  // exp(-0.625) = 0.53526
                6'd6:  lut_val = 16'd147;  // exp(-0.750) = 0.47237
                6'd7:  lut_val = 16'd134;  // exp(-0.875) = 0.41686
                6'd8:  lut_val = 16'd122;  // exp(-1.000) = 0.36788
                6'd9:  lut_val = 16'd111;  // exp(-1.125) = 0.32465
                6'd10: lut_val = 16'd101;  // exp(-1.250) = 0.28650
                6'd11: lut_val = 16'd92;   // exp(-1.375) = 0.25284
                6'd12: lut_val = 16'd84;   // exp(-1.500) = 0.22313
                6'd13: lut_val = 16'd76;   // exp(-1.625) = 0.19691
                6'd14: lut_val = 16'd69;   // exp(-1.750) = 0.17377
                6'd15: lut_val = 16'd63;   // exp(-1.875) = 0.15335
                6'd16: lut_val = 16'd57;   // exp(-2.000) = 0.13534
                6'd17: lut_val = 16'd52;   // exp(-2.125) = 0.11943
                6'd18: lut_val = 16'd47;   // exp(-2.250) = 0.10540
                6'd19: lut_val = 16'd43;   // exp(-2.375) = 0.09302
                6'd20: lut_val = 16'd39;   // exp(-2.500) = 0.08208
                6'd21: lut_val = 16'd35;   // exp(-2.625) = 0.07244
                6'd22: lut_val = 16'd32;   // exp(-2.750) = 0.06393
                6'd23: lut_val = 16'd29;   // exp(-2.875) = 0.05643
                6'd24: lut_val = 16'd26;   // exp(-3.000) = 0.04979
                6'd25: lut_val = 16'd24;   // exp(-3.125) = 0.04394
                6'd26: lut_val = 16'd22;   // exp(-3.250) = 0.03877
                6'd27: lut_val = 16'd20;   // exp(-3.375) = 0.03422
                6'd28: lut_val = 16'd18;   // exp(-3.500) = 0.03020
                6'd29: lut_val = 16'd16;   // exp(-3.625) = 0.02664
                6'd30: lut_val = 16'd15;   // exp(-3.750) = 0.02352
                6'd31: lut_val = 16'd13;   // exp(-3.875) = 0.02075
                6'd32: lut_val = 16'd12;   // exp(-4.000) = 0.01832
                6'd33: lut_val = 16'd11;   // exp(-4.125) = 0.01616
                6'd34: lut_val = 16'd10;   // exp(-4.250) = 0.01426
                6'd35: lut_val = 16'd9;    // exp(-4.375) = 0.01259
                6'd36: lut_val = 16'd8;    // exp(-4.500) = 0.01111
                6'd37: lut_val = 16'd7;    // exp(-4.625) = 0.00980
                6'd38: lut_val = 16'd6;    // exp(-4.750) = 0.00865
                6'd39: lut_val = 16'd6;    // exp(-4.875) = 0.00763
                6'd40: lut_val = 16'd5;    // exp(-5.000) = 0.00674
                6'd41: lut_val = 16'd5;    // exp(-5.125) = 0.00595
                6'd42: lut_val = 16'd4;    // exp(-5.250) = 0.00525
                6'd43: lut_val = 16'd4;    // exp(-5.375) = 0.00463
                6'd44: lut_val = 16'd3;    // exp(-5.500) = 0.00409
                6'd45: lut_val = 16'd3;    // exp(-5.625) = 0.00361
                6'd46: lut_val = 16'd3;    // exp(-5.750) = 0.00318
                6'd47: lut_val = 16'd3;    // exp(-5.875) = 0.00281
                6'd48: lut_val = 16'd2;    // exp(-6.000) = 0.00248
                6'd49: lut_val = 16'd2;    // exp(-6.125) = 0.00219
                6'd50: lut_val = 16'd2;    // exp(-6.250) = 0.00193
                6'd51: lut_val = 16'd2;    // exp(-6.375) = 0.00170
                6'd52: lut_val = 16'd1;    // exp(-6.500) = 0.00150
                6'd53: lut_val = 16'd1;    // exp(-6.625) = 0.00133
                6'd54: lut_val = 16'd1;    // exp(-6.750) = 0.00117
                6'd55: lut_val = 16'd1;    // exp(-6.875) = 0.00103
                default: lut_val = 16'd0;  // x >= 7.0: exp ≈ 0
            endcase
        end
    endfunction

    // Stage 1: extract LUT index (upper 6 bits = integer + upper 3 fractional)
    // x_abs is Q8.8: bits [15:8] = integer, bits [7:0] = frac
    // LUT index = x / 0.125 = x_abs >> (8-3) but we index by floor(x/0.125)
    // = x_abs[15:5] clamped to 6 bits (max index 63)
    // Fractional part for interpolation = x_abs[4:0] (lower 5 bits)

    reg        s1_valid;
    reg [15:0] s1_y0, s1_y1;
    reg [4:0]  s1_frac;  // 5-bit fraction within LUT interval
    reg        s1_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_y0 <= 16'd0; s1_y1 <= 16'd0;
            s1_frac <= 5'd0; s1_zero <= 1'b0;
        end else begin
            s1_valid <= in_valid;
            s1_frac  <= x_abs[4:0];
            s1_zero  <= (x_abs[15:11] != 5'd0); // x >= 8.0 → exp ≈ 0
            begin : lut_stage
                reg [5:0] idx, idx1;
                idx  = (x_abs[15:5] > 6'd63) ? 6'd63 : x_abs[15:5];
                idx1 = (idx == 6'd63) ? 6'd63 : idx + 6'd1;
                s1_y0 <= lut_val(idx);
                s1_y1 <= lut_val(idx1);
            end
        end
    end

    // Stage 2: linear interpolation y = y0 + (y1 - y0) * frac / 32
    // frac is 5-bit [0..31], interval step = 0.125 in Q8.8 = 32 LSBs at frac scale
    reg        s2_valid;
    reg [15:0] s2_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_result <= 16'd0;
        end else begin
            s2_valid <= s1_valid;
            if (s1_zero) begin
                s2_result <= 16'd0;
            end else begin : interp
                reg signed [16:0] diff;
                reg signed [21:0] interp_full;
                reg [15:0]        interp_val;
                diff        = $signed({1'b0, s1_y1}) - $signed({1'b0, s1_y0}); // can be neg
                interp_full = $signed(diff) * $signed({2'b0, s1_frac});         // scaled by 32
                interp_val  = s1_y0 + ($signed(interp_full) >>> 5);
                s2_result   <= interp_val;
            end
        end
    end

    // Stage 3: output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'd0; out_valid <= 1'b0;
        end else begin
            out_valid <= s2_valid;
            result    <= s2_result;
        end
    end
endmodule


// =============================================================================
// neuron_core — Q8.8 WNN neuron, no SEM20
// Computes: out = w * z * exp(-0.5 * z^2),  z = (x - t) / d
//
// Pipeline latency: 1 + 16 + 1 + 1 + 3 + 1 + 1 = 24 cycles
// =============================================================================
module neuron_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire signed [15:0] x_q8p8,
    input  wire signed [15:0] t_q8p8,
    input  wire signed [15:0] d_q8p8,
    input  wire signed [15:0] w_q8p8,
    output wire signed [15:0] out_q8p8,
    output wire               out_valid
);
    // -------------------------------------------------------------------------
    // Stage 1: sub = x - t  (1 cycle)
    // -------------------------------------------------------------------------
    reg signed [15:0] sub_out;
    reg               sub_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sub_out <= 16'sd0; sub_valid <= 1'b0;
        end else begin
            sub_valid <= in_valid;
            // Saturating subtraction
            begin : sub_block
                reg signed [16:0] diff;
                diff = $signed({x_q8p8[15], x_q8p8}) - $signed({t_q8p8[15], t_q8p8});
                if      (diff > 17'sh07FFF) sub_out <= 16'h7FFF;
                else if (diff < 17'sh18000) sub_out <= 16'sh8000;
                else                         sub_out <= diff[15:0];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: div = (x - t) / d  (16 cycles)
    // -------------------------------------------------------------------------
    wire signed [15:0] z_out;
    wire               z_valid;

    q8p8_div u_div (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (sub_valid),
        .a        (sub_out),
        .b        (d_q8p8),
        .result   (z_out),
        .out_valid(z_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 3: z2 = z * z  (1 cycle)
    // -------------------------------------------------------------------------
    wire signed [15:0] z2_out;
    wire               z2_valid;

    q8p8_mul u_z2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (z_valid),
        .a        (z_out),
        .b        (z_out),
        .product  (z2_out),
        .out_valid(z2_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 4: half_z2 = 0.5 * z^2  (1 cycle: arithmetic right shift by 1)
    // z^2 is always positive (product of same-sign operands), take as unsigned
    // -------------------------------------------------------------------------
    reg [15:0] half_z2;
    reg        half_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            half_z2 <= 16'd0; half_valid <= 1'b0;
        end else begin
            half_valid <= z2_valid;
            half_z2    <= z2_out[15:1];  // divide by 2, Q8.8 unsigned
        end
    end

    // Delay z to align with exp output (exp takes 3 cycles from half_z2)
    // We need z delayed by 3+1=4 more cycles after z2 stage
    // z_valid → z2_valid: 1 cycle
    // z2_valid → half_valid: 1 cycle
    // half_valid → exp_out_valid: 3 cycles
    // So z must be delayed 1+1+3 = 5 cycles from z_valid
    reg signed [15:0] z_delay [0:4];
    reg               zv_delay[0:4];

    integer di;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (di = 0; di < 5; di = di + 1) begin
                z_delay[di]  <= 16'sd0;
                zv_delay[di] <= 1'b0;
            end
        end else begin
            z_delay[0]  <= z_out;
            zv_delay[0] <= z_valid;
            for (di = 1; di < 5; di = di + 1) begin
                z_delay[di]  <= z_delay[di-1];
                zv_delay[di] <= zv_delay[di-1];
            end
        end
    end

    wire signed [15:0] z_aligned  = z_delay[4];
    wire               z_aligned_v = zv_delay[4];

    // -------------------------------------------------------------------------
    // Stage 5: exp_val = exp(-0.5 * z^2)  (3 cycles)
    // -------------------------------------------------------------------------
    wire [15:0] exp_out;
    wire        exp_valid;

    exp_lut_q8p8 u_exp (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (half_valid),
        .x_abs    (half_z2),
        .result   (exp_out),
        .out_valid(exp_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 6: shape = z * exp(...)  (1 cycle)
    // exp_out is Q8.8 unsigned in (0,1], treat as signed (always positive)
    // z_aligned is Q8.8 signed
    // -------------------------------------------------------------------------
    wire signed [15:0] shape_out;
    wire               shape_valid;

    q8p8_mul u_shape (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (exp_valid & z_aligned_v),
        .a        (z_aligned),
        .b        ($signed(exp_out)),
        .product  (shape_out),
        .out_valid(shape_valid)
    );

    // Delay w to align with shape_valid
    // z_valid → shape_valid = 5 (z delay) + 0 (sync) + 1 (mul) = 6 cycles from z_valid
    // w needs to be delayed 16 (div) + 6 = 22 cycles from in_valid
    // sub_valid is in_valid + 1 cycle, z_valid = sub_valid + 16 = in_valid + 17
    // shape_valid = z_valid + 5 + 1 = in_valid + 23
    // So delay w from in_valid by 23 cycles total (1 sub + 16 div + 6 more)
    reg signed [15:0] w_delay [0:22];
    reg               wv_delay[0:22];

    integer wi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < 23; wi = wi + 1) begin
                w_delay[wi]  <= 16'sd0;
                wv_delay[wi] <= 1'b0;
            end
        end else begin
            w_delay[0]  <= w_q8p8;
            wv_delay[0] <= in_valid;
            for (wi = 1; wi < 23; wi = wi + 1) begin
                w_delay[wi]  <= w_delay[wi-1];
                wv_delay[wi] <= wv_delay[wi-1];
            end
        end
    end

    wire signed [15:0] w_aligned   = w_delay[22];
    wire               w_aligned_v = wv_delay[22];

    // -------------------------------------------------------------------------
    // Stage 7: final = w * shape  (1 cycle)
    // -------------------------------------------------------------------------
    q8p8_mul u_final (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (shape_valid & w_aligned_v),
        .a        (w_aligned),
        .b        (shape_out),
        .product  (out_q8p8),
        .out_valid(out_valid)
    );

endmodule


// =============================================================================
// tt_um_wnn_q8p8 — Top-level Tiny Tapeout module (Q8.8 native, no SEM20)
//
// WHY ARE THERE CFG REGISTERS?
// -----------------------------
// The Tiny Tapeout platform gives only 8 input pins total. The neuron needs
// 3 parameters (w, t, d) each 16 bits = 48 bits of configuration that must
// be loaded before inference. There is no separate "config bus" on the chip.
//
// The cfg register file is a serial-load mechanism:
//   1. You shift a 16-bit word in over 16 clock cycles via cfg_serial/cfg_valid.
//   2. You pulse cfg_load with cfg_param selecting which register (w/t/d) to latch.
//   3. Repeat for each parameter.
//
// Without it you would need to hard-wire w, t, d at synthesis time — making
// the chip a fixed-function device. The cfg registers let one chip serve
// multiple wavelet shapes just by reprogramming over the serial interface,
// exactly like an SPI configuration register. It's the standard approach for
// pin-constrained ASICs like Tiny Tapeout.
// =============================================================================
module tt_um_wnn_q8p8 (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    wire       cfg_serial = ui_in[0];
    wire       cfg_valid  = ui_in[1];
    wire       cfg_load   = ui_in[2];
    wire [1:0] cfg_param  = ui_in[4:3];
    wire       x_serial   = ui_in[5];
    wire       x_valid    = ui_in[6];

    // -------------------------------------------------------------------------
    // Configuration Register File
    // Shift 16 bits in serially, then latch into w/t/d on cfg_load.
    // -------------------------------------------------------------------------
    reg signed [15:0] w_reg, t_reg, d_reg;
    reg [15:0]        cfg_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cfg_shift <= 16'd0;
        else if (cfg_valid)
            cfg_shift <= {cfg_serial, cfg_shift[15:1]};  // LSB first
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_reg <= 16'sd0; t_reg <= 16'sd0; d_reg <= 16'd256; // d default = 1.0 in Q8.8
        end else if (cfg_load) begin
            case (cfg_param)
                2'b00: w_reg <= cfg_shift;
                2'b01: t_reg <= cfg_shift;
                2'b10: d_reg <= cfg_shift;
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Input Deserializer — 16-bit serial → parallel
    // -------------------------------------------------------------------------
    reg [15:0] x_shift;
    reg [3:0]  x_bit_cnt;

    wire x_deser_done = (x_bit_cnt == 4'd15) && x_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_shift   <= 16'd0;
            x_bit_cnt <= 4'd0;
        end else if (x_valid) begin
            x_shift   <= {x_serial, x_shift[15:1]};
            x_bit_cnt <= (x_bit_cnt == 4'd15) ? 4'd0 : (x_bit_cnt + 1'b1);
        end
    end

    reg signed [15:0] x_q8p8_latched;
    reg               x_latch_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_q8p8_latched <= 16'sd0;
            x_latch_valid  <= 1'b0;
        end else if (x_deser_done) begin
            x_q8p8_latched <= {x_serial, x_shift[15:1]};
            x_latch_valid  <= 1'b1;
        end else begin
            x_latch_valid  <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Neuron instance
    // -------------------------------------------------------------------------
    wire signed [15:0] neuron_out;
    wire               neuron_out_valid;

    neuron_core u_neuron (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (x_latch_valid),
        .x_q8p8   (x_q8p8_latched),
        .t_q8p8   (t_reg),
        .d_q8p8   (d_reg),
        .w_q8p8   (w_reg),
        .out_q8p8 (neuron_out),
        .out_valid(neuron_out_valid)
    );

    // -------------------------------------------------------------------------
    // Output capture register
    // -------------------------------------------------------------------------
    reg signed [15:0] tree_sum;
    reg               tree_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tree_sum   <= 16'sd0;
            tree_valid <= 1'b0;
        end else begin
            tree_sum   <= neuron_out;
            tree_valid <= neuron_out_valid;
        end
    end

    // -------------------------------------------------------------------------
    // Output Serializer — 16-bit parallel → serial, LSB first
    // -------------------------------------------------------------------------
    reg [15:0] sum_shift;
    reg [3:0]  sum_bit_cnt;
    reg        shift_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_shift    <= 16'd0;
            sum_bit_cnt  <= 4'd0;
            shift_active <= 1'b0;
        end else if (tree_valid && !shift_active) begin
            sum_shift    <= tree_sum;
            sum_bit_cnt  <= 4'd0;
            shift_active <= 1'b1;
        end else if (shift_active) begin
            sum_shift   <= {1'b0, sum_shift[15:1]};
            sum_bit_cnt <= sum_bit_cnt + 1'b1;
            if (sum_bit_cnt == 4'd15)
                shift_active <= 1'b0;
        end
    end

    wire sum_serial_out = sum_shift[0];
    wire sum_valid_out  = shift_active;

    // -------------------------------------------------------------------------
    // Pipeline busy tracking (24 cycles)
    // -------------------------------------------------------------------------
    localparam integer PIPELINE_DEPTH = 25;  // 24 cycle latency + 1 margin
    reg [4:0] busy_counter;
    wire      pipeline_busy = (busy_counter != 5'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_counter <= 5'd0;
        end else if (x_latch_valid) begin
            busy_counter <= PIPELINE_DEPTH[4:0];
        end else if (busy_counter > 0) begin
            busy_counter <= busy_counter - 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Ready signal
    // -------------------------------------------------------------------------
    wire ready_comb = (x_bit_cnt == 4'd0) && !shift_active
                      && !pipeline_busy && !x_latch_valid;

    reg ready;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) ready <= 1'b0;
        else        ready <= ready_comb;
    end

    assign uo_out = {5'b00000, ready, sum_valid_out, sum_serial_out};

    wire _unused = &{ena, uio_in, 1'b0};

endmodule
