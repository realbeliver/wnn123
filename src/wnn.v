/*
 * tt_um_wnn.v  (converted from C2S0285_1n.sv)
 * Single-neuron WNN accelerator — Tiny Tapeout standard interface.
 *
 * SEM20 Floating-Point Format:
 *   Bit 19    : Sign (0 = positive, 1 = negative)
 *   Bits 18:13: Exponent (6 bits, bias = 31)
 *   Bits 12:0 : Mantissa (13 bits, implied leading 1 for normal numbers)
 *   Zero is represented by all bits zero (exponent = 0, mantissa = 0).
 *   Subnormals: exponent=0 with non-zero mantissa treated as zero.
 *   Rounding  : round to nearest, ties to even.
 *
 * Latency: 100 (neuron core) + 1 (output capture reg) + 16 (output serializer) = 117 cycles.
 *
 * I/O Mapping (Tiny Tapeout standard ports):
 *   ui_in[0]    cfg_serial   — config shift-register data (LSB first)
 *   ui_in[1]    cfg_valid    — clock enable for config shift register
 *   ui_in[2]    cfg_load     — latch shifted word into selected register
 *   ui_in[4:3]  cfg_param    — register select: 00=w, 01=t, 10=d
 *   ui_in[5]    x_serial     — inference input data (LSB first)
 *   ui_in[6]    x_valid      — clock enable for input deserializer
 *   ui_in[7]    (unused)
 *   uo_out[0]   sum_serial   — inference output data (LSB first)
 *   uo_out[1]   sum_valid    — output shift register active
 *   uo_out[2]   ready        — design ready for next input
 *   uo_out[7:3] tied 0
 *   uio_*       tied 0 / all-input  (unused)
 *   cfg_neuron  hardwired 0  — single-neuron design, no address needed
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none
`timescale 1ns / 1ps

// =============================================================================
// 3-to-2 Carry-Save Adder
// =============================================================================
module csa_3to2 #(parameter integer W = 44) (
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    input  wire [W-1:0] c,
    output wire [W-1:0] sum,
    output wire [W-1:0] carry
);
    assign sum   = a ^ b ^ c;
    assign carry = (a & b) | (b & c) | (a & c);
endmodule


// =============================================================================
// 16-bit Leading-Zero Detector
// =============================================================================
module lzd16 (
    input  wire [15:0] in,
    output reg  [3:0]  pos,
    output reg         valid
);
    always @(*) begin
        valid = 1'b1;
        casez (in)
            16'b1???_????_????_????: pos = 4'd15;
            16'b01??_????_????_????: pos = 4'd14;
            16'b001?_????_????_????: pos = 4'd13;
            16'b0001_????_????_????: pos = 4'd12;
            16'b0000_1???_????_????: pos = 4'd11;
            16'b0000_01??_????_????: pos = 4'd10;
            16'b0000_001?_????_????: pos = 4'd9;
            16'b0000_0001_????_????: pos = 4'd8;
            16'b0000_0000_1???_????: pos = 4'd7;
            16'b0000_0000_01??_????: pos = 4'd6;
            16'b0000_0000_001?_????: pos = 4'd5;
            16'b0000_0000_0001_????: pos = 4'd4;
            16'b0000_0000_0000_1???: pos = 4'd3;
            16'b0000_0000_0000_01??: pos = 4'd2;
            16'b0000_0000_0000_001?: pos = 4'd1;
            16'b0000_0000_0000_0001: pos = 4'd0;
            default: begin
                pos   = 4'd0;
                valid = 1'b0;
            end
        endcase
    end
endmodule


// =============================================================================
// SEM20 Multiplier - 6-cycle pipeline
// =============================================================================
module sem20_mul #(parameter integer W = 20) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    output reg  [W-1:0] product,
    output reg          valid_out
);
    localparam integer PW = 44;

    wire        s_a   = a[19];
    wire [5:0]  ea_w  = a[18:13];
    wire [13:0] ma_w  = (ea_w == 6'd0) ? 14'd0 : {1'b1, a[12:0]};
    wire        s_b   = b[19];
    wire [5:0]  eb_w  = b[18:13];
    wire [13:0] mb_w  = (eb_w == 6'd0) ? 14'd0 : {1'b1, b[12:0]};
    wire        za    = (ea_w == 6'd0);
    wire        zb    = (eb_w == 6'd0);
    wire [16:0] bpad  = {2'b00, mb_w, 1'b0};
    wire [2:0]  g0 = bpad[ 2: 0], g1 = bpad[ 4: 2], g2 = bpad[ 6: 4], g3 = bpad[ 8: 6];
    wire [2:0]  g4 = bpad[10: 8], g5 = bpad[12:10], g6 = bpad[14:12], g7 = bpad[16:14];

    // ---- Stage 1 -------------------------------------------------------
    reg        s1v, s1sgn, s1z;
    reg [7:0]  s1ea, s1eb;
    reg [13:0] s1ma;
    reg [2:0]  s1g0,s1g1,s1g2,s1g3,s1g4,s1g5,s1g6,s1g7;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1v <= 1'b0; s1sgn <= 1'b0; s1z <= 1'b0;
            s1ea <= 8'd0; s1eb <= 8'd0; s1ma <= 14'd0;
            s1g0 <= 3'd0; s1g1 <= 3'd0; s1g2 <= 3'd0; s1g3 <= 3'd0;
            s1g4 <= 3'd0; s1g5 <= 3'd0; s1g6 <= 3'd0; s1g7 <= 3'd0;
        end else begin
            s1v <= in_valid; s1sgn <= s_a ^ s_b; s1z <= za | zb;
            s1ea <= {2'b00, ea_w}; s1eb <= {2'b00, eb_w}; s1ma <= ma_w;
            s1g0 <= g0; s1g1 <= g1; s1g2 <= g2; s1g3 <= g3;
            s1g4 <= g4; s1g5 <= g5; s1g6 <= g6; s1g7 <= g7;
        end
    end

    // Booth partial-product decode function (implemented as task-like wires)
    function [16:0] bdec;
        input [2:0] grp;
        input [13:0] A;
        reg [15:0] pa, p2a, na, n2a, v;
        reg        c;
        begin
            pa  = {2'b00, A};
            p2a = {1'b0, A, 1'b0};
            na  = ~pa;
            n2a = ~p2a;
            case (grp)
                3'b000, 3'b111: begin v = 16'd0; c = 1'b0; end
                3'b001, 3'b010: begin v = pa;    c = 1'b0; end
                3'b011:         begin v = p2a;   c = 1'b0; end
                3'b100:         begin v = n2a;   c = 1'b1; end
                3'b101, 3'b110: begin v = na;    c = 1'b1; end
                default:        begin v = 16'd0; c = 1'b0; end
            endcase
            bdec = {c, v};
        end
    endfunction

    wire [16:0] r0=bdec(s1g0,s1ma), r1=bdec(s1g1,s1ma), r2=bdec(s1g2,s1ma), r3=bdec(s1g3,s1ma);
    wire [16:0] r4=bdec(s1g4,s1ma), r5=bdec(s1g5,s1ma), r6=bdec(s1g6,s1ma), r7=bdec(s1g7,s1ma);

    wire [PW-1:0] pp0 = {{(PW-16){r0[15]}}, r0[15:0]};
    wire [PW-1:0] pp1 = {{(PW-18){r1[15]}}, r1[15:0], 2'b00};
    wire [PW-1:0] pp2 = {{(PW-20){r2[15]}}, r2[15:0], 4'b0};
    wire [PW-1:0] pp3 = {{(PW-22){r3[15]}}, r3[15:0], 6'b0};
    wire [PW-1:0] pp4 = {{(PW-24){r4[15]}}, r4[15:0], 8'b0};
    wire [PW-1:0] pp5 = {{(PW-26){r5[15]}}, r5[15:0], 10'b0};
    wire [PW-1:0] pp6 = {{(PW-28){r6[15]}}, r6[15:0], 12'b0};
    wire [PW-1:0] pp7 = {{(PW-30){r7[15]}}, r7[15:0], 14'b0};

    wire [PW-1:0] bcorr = ( {{(PW-1){1'b0}}, r0[16]}        )
                        | ( {{(PW-3){1'b0}}, r1[16], 2'b0}  )
                        | ( {{(PW-5){1'b0}}, r2[16], 4'b0}  )
                        | ( {{(PW-7){1'b0}}, r3[16], 6'b0}  )
                        | ( {{(PW-9){1'b0}}, r4[16], 8'b0}  )
                        | ({{(PW-11){1'b0}}, r5[16],10'b0}  )
                        | ({{(PW-13){1'b0}}, r6[16],12'b0}  )
                        | ({{(PW-15){1'b0}}, r7[16],14'b0}  );

    wire [PW-1:0] l1s0,l1c0, l1s1,l1c1, l1s2,l1c2;
    csa_3to2 #(PW) c1a(.a(pp0),.b(pp1),.c(pp2),  .sum(l1s0),.carry(l1c0));
    csa_3to2 #(PW) c1b(.a(pp3),.b(pp4),.c(pp5),  .sum(l1s1),.carry(l1c1));
    csa_3to2 #(PW) c1c(.a(pp6),.b(pp7),.c(bcorr),.sum(l1s2),.carry(l1c2));

    wire [PW-1:0] l2s0,l2c0, l2s1,l2c1;
    csa_3to2 #(PW) c2a(.a(l1s0), .b({l1c0[PW-2:0],1'b0}), .c(l1s1), .sum(l2s0),.carry(l2c0));
    csa_3to2 #(PW) c2b(.a({l1c1[PW-2:0],1'b0}), .b(l1s2), .c({l1c2[PW-2:0],1'b0}), .sum(l2s1),.carry(l2c1));

    // ---- Stage 2 -------------------------------------------------------
    reg [PW-1:0] s2w0, s2w1, s2w2, s2w3;
    reg [7:0]    s2ea, s2eb;
    reg          s2v, s2sgn, s2z;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2v <= 1'b0; s2w0 <= {PW{1'b0}}; s2w1 <= {PW{1'b0}}; s2w2 <= {PW{1'b0}}; s2w3 <= {PW{1'b0}};
            s2ea <= 8'd0; s2eb <= 8'd0; s2sgn <= 1'b0; s2z <= 1'b0;
        end else begin
            s2v <= s1v;
            s2w0 <= l2s0; s2w1 <= {l2c0[PW-2:0],1'b0}; s2w2 <= l2s1; s2w3 <= {l2c1[PW-2:0],1'b0};
            s2ea <= s1ea; s2eb <= s1eb; s2sgn <= s1sgn; s2z <= s1z;
        end
    end

    // ---- CSA level 3 ---------------------------------------------------
    wire [PW-1:0] l3s, l3c;
    csa_3to2 #(PW) c3(.a(s2w0),.b(s2w1),.c(s2w2),.sum(l3s),.carry(l3c));

    // ---- CSA level 4 ---------------------------------------------------
    wire [PW-1:0] l4s, l4c;
    csa_3to2 #(PW) c4(.a(l3s),.b({l3c[PW-2:0],1'b0}),.c(s2w3),.sum(l4s),.carry(l4c));

    // ---- Stage 2b ------------------------------------------------------
    reg [PW-1:0] s2b_sum, s2b_carry;
    reg [7:0]    s2b_ea, s2b_eb;
    reg          s2b_v, s2b_sgn, s2b_z;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2b_v <= 1'b0; s2b_sum <= {PW{1'b0}}; s2b_carry <= {PW{1'b0}};
            s2b_ea <= 8'd0; s2b_eb <= 8'd0; s2b_sgn <= 1'b0; s2b_z <= 1'b0;
        end else begin
            s2b_v     <= s2v;
            s2b_sum   <= l4s;
            s2b_carry <= {l4c[PW-2:0], 1'b0};
            s2b_ea    <= s2ea; s2b_eb <= s2eb;
            s2b_sgn   <= s2sgn; s2b_z <= s2z;
        end
    end

    wire [27:0] cpa = s2b_sum[27:0] + s2b_carry[27:0];

    // ---- Stage 3 -------------------------------------------------------
    reg [27:0] s3prod;
    reg [7:0]  s3ea, s3eb;
    reg        s3v, s3sgn, s3z;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3v <= 1'b0; s3prod <= 28'd0; s3ea <= 8'd0; s3eb <= 8'd0; s3sgn <= 1'b0; s3z <= 1'b0;
        end else begin
            s3v <= s2b_v; s3prod <= cpa; s3ea <= s2b_ea; s3eb <= s2b_eb; s3sgn <= s2b_sgn; s3z <= s2b_z;
        end
    end

    wire       ovf4  = s3prod[27];
    wire [9:0] ebase = {2'b00, s3ea} + {2'b00, s3eb} - 10'd31;
    wire [9:0] enorm = ebase + {9'b0, ovf4};
    wire [12:0] mant4 = ovf4 ? s3prod[26:14] : s3prod[25:13];
    wire        grd4  = ovf4 ? s3prod[13]    : s3prod[12];
    wire        rnd4  = ovf4 ? s3prod[12]    : s3prod[11];
    wire        st4   = ovf4 ? |s3prod[11:0] : |s3prod[10:0];

    // ---- Stage 4 -------------------------------------------------------
    reg [12:0] s4mant;
    reg [9:0]  s4esum;
    reg        s4v, s4sgn, s4z;
    reg        s4g, s4r, s4st;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4v <= 1'b0; s4mant <= 13'd0; s4esum <= 10'd0; s4sgn <= 1'b0; s4z <= 1'b0;
            s4g <= 1'b0; s4r <= 1'b0; s4st <= 1'b0;
        end else begin
            s4v <= s3v; s4mant <= mant4; s4esum <= enorm; s4sgn <= s3sgn; s4z <= s3z;
            s4g <= grd4; s4r <= rnd4; s4st <= st4;
        end
    end

    wire        rndup  = s4g & (s4r | s4st | s4mant[0]);
    wire [13:0] mantr  = {1'b0, s4mant} + {13'b0, rndup};
    wire        mcarry = mantr[13];
    wire [9:0]  efin   = s4esum + {9'b0, mcarry};
    wire [12:0] mfin   = mcarry ? 13'd0 : mantr[12:0];
    wire        uflow  = s4z | efin[9] | (efin == 10'd0);
    wire        oflow  = (~efin[9]) & (efin >= 10'd63);

    // ---- Stage 5 (output FF) -------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0; product <= 20'd0;
        end else begin
            valid_out <= s4v;
            if (uflow)       product <= {s4sgn, 19'd0};
            else if (oflow)  product <= {s4sgn, 6'd62, 13'h1FFF};
            else             product <= {s4sgn, efin[5:0], mfin};
        end
    end
endmodule


// =============================================================================
// SEM20 Adder (6-cycle pipeline)
// =============================================================================
module sem20_add (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [19:0] a,
    input  wire [19:0] b,
    output reg         out_valid,
    output reg  [19:0] result
);
    reg        s0_valid, s0_sign_a, s0_sign_b;
    reg [5:0]  s0_exp_a,  s0_exp_b;
    reg [13:0] s0_mant_a, s0_mant_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0; s0_sign_a <= 1'b0; s0_sign_b <= 1'b0;
            s0_exp_a <= 6'h0; s0_exp_b <= 6'h0; s0_mant_a <= 14'h0; s0_mant_b <= 14'h0;
        end else begin
            s0_valid  <= in_valid;
            s0_sign_a <= a[19]; s0_sign_b <= b[19];
            s0_exp_a  <= a[18:13]; s0_exp_b  <= b[18:13];
            s0_mant_a <= (a[18:13] == 6'h0 && a[12:0] == 13'h0) ? 14'h0 : {1'b1, a[12:0]};
            s0_mant_b <= (b[18:13] == 6'h0 && b[12:0] == 13'h0) ? 14'h0 : {1'b1, b[12:0]};
        end
    end

    reg [5:0]  c1_exp_big;
    reg [13:0] c1_mant_big, c1_mant_sml_aligned;
    reg        c1_sign_big, c1_sign_sml;
    reg [5:0]  c1_shamt;
    reg [27:0] c1_extended, c1_shifted;
    reg        c1_sticky;

    always @(*) begin
        if (s0_exp_a >= s0_exp_b) begin
            c1_exp_big  = s0_exp_a; c1_mant_big = s0_mant_a; c1_sign_big = s0_sign_a;
            c1_sign_sml = s0_sign_b;
            c1_shamt = ((s0_exp_a - s0_exp_b) > 6'd14) ? 6'd14 : (s0_exp_a - s0_exp_b);
            c1_extended = {s0_mant_b, 14'h0};
        end else begin
            c1_exp_big  = s0_exp_b; c1_mant_big = s0_mant_b; c1_sign_big = s0_sign_b;
            c1_sign_sml = s0_sign_a;
            c1_shamt = ((s0_exp_b - s0_exp_a) > 6'd14) ? 6'd14 : (s0_exp_b - s0_exp_a);
            c1_extended = {s0_mant_a, 14'h0};
        end
        case (c1_shamt)
            6'd0:    c1_shifted = c1_extended;
            6'd1:    c1_shifted = c1_extended >> 1;
            6'd2:    c1_shifted = c1_extended >> 2;
            6'd3:    c1_shifted = c1_extended >> 3;
            6'd4:    c1_shifted = c1_extended >> 4;
            6'd5:    c1_shifted = c1_extended >> 5;
            6'd6:    c1_shifted = c1_extended >> 6;
            6'd7:    c1_shifted = c1_extended >> 7;
            6'd8:    c1_shifted = c1_extended >> 8;
            6'd9:    c1_shifted = c1_extended >> 9;
            6'd10:   c1_shifted = c1_extended >> 10;
            6'd11:   c1_shifted = c1_extended >> 11;
            6'd12:   c1_shifted = c1_extended >> 12;
            6'd13:   c1_shifted = c1_extended >> 13;
            default: c1_shifted = c1_extended >> 14;
        endcase
        c1_mant_sml_aligned = c1_shifted[27:14];
        c1_sticky           = |c1_shifted[13:0];
    end

    reg        s1_valid;
    reg [5:0]  s1_exp_res;
    reg [13:0] s1_mant_big, s1_mant_sml;
    reg        s1_sticky;
    reg        s1_sign_big, s1_sign_sml;
    reg        s1_sign_a, s1_sign_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_exp_res <= 6'h0; s1_mant_big <= 14'h0; s1_mant_sml <= 14'h0;
            s1_sticky <= 1'b0; s1_sign_big <= 1'b0; s1_sign_sml <= 1'b0; s1_sign_a <= 1'b0; s1_sign_b <= 1'b0;
        end else begin
            s1_valid <= s0_valid; s1_exp_res <= c1_exp_big; s1_mant_big <= c1_mant_big;
            s1_mant_sml <= c1_mant_sml_aligned; s1_sticky <= c1_sticky;
            s1_sign_big <= c1_sign_big; s1_sign_sml <= c1_sign_sml;
            s1_sign_a <= s0_sign_a; s1_sign_b <= s0_sign_b;
        end
    end

    reg        c2_same_sign;
    reg [14:0] c2_sum, c2_sub_big_sml, c2_sub_sml_big;
    reg        c2_big_ge_sml;
    reg [14:0] c2_mant_res;
    reg        c2_sign_res;

    always @(*) begin
        c2_same_sign    = (s1_sign_a == s1_sign_b);
        c2_sum          = {1'b0, s1_mant_big} + {1'b0, s1_mant_sml};
        c2_sub_big_sml  = {1'b0, s1_mant_big} - {1'b0, s1_mant_sml};
        c2_sub_sml_big  = {1'b0, s1_mant_sml} - {1'b0, s1_mant_big};
        c2_big_ge_sml   = (s1_mant_big >= s1_mant_sml);
        if (c2_same_sign) begin
            c2_mant_res = c2_sum; c2_sign_res = s1_sign_big;
        end else begin
            if (c2_big_ge_sml) begin
                c2_mant_res = c2_sub_big_sml; c2_sign_res = s1_sign_big;
            end else begin
                c2_mant_res = c2_sub_sml_big; c2_sign_res = s1_sign_sml;
            end
        end
    end

    reg        s2_valid, s2_sign_res;
    reg [5:0]  s2_exp_res;
    reg [14:0] s2_mant_res;
    reg        s2_sticky;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_sign_res <= 1'b0; s2_exp_res <= 6'h0; s2_mant_res <= 15'h0; s2_sticky <= 1'b0;
        end else begin
            s2_valid <= s1_valid; s2_sign_res <= c2_sign_res; s2_exp_res <= s1_exp_res;
            s2_mant_res <= c2_mant_res; s2_sticky <= s1_sticky;
        end
    end

    function [3:0] lzd15;
        input [14:0] x;
        begin
            casez (x)
                15'b1??????????????: lzd15 = 4'd0;
                15'b01?????????????: lzd15 = 4'd1;
                15'b001????????????: lzd15 = 4'd2;
                15'b0001???????????: lzd15 = 4'd3;
                15'b00001??????????: lzd15 = 4'd4;
                15'b000001?????????: lzd15 = 4'd5;
                15'b0000001????????: lzd15 = 4'd6;
                15'b00000001???????: lzd15 = 4'd7;
                15'b000000001??????: lzd15 = 4'd8;
                15'b0000000001?????: lzd15 = 4'd9;
                15'b00000000001????: lzd15 = 4'd10;
                15'b000000000001???: lzd15 = 4'd11;
                15'b0000000000001??: lzd15 = 4'd12;
                15'b00000000000001?: lzd15 = 4'd13;
                15'b000000000000001: lzd15 = 4'd14;
                default:             lzd15 = 4'd15;
            endcase
        end
    endfunction

    reg        c3_is_zero;
    reg [3:0]  c3_lz, c3_shift_left, c3_raw_shift;
    reg [6:0]  c3_exp_tmp;
    reg [28:0] c3_left_ext, c3_left_shifted;
    reg [13:0] c3_mant_norm;
    reg [6:0]  c3_exp_norm;
    reg        c3_guard, c3_round_bit, c3_sticky;

    always @(*) begin
        c3_is_zero   = (s2_mant_res == 15'h0);
        c3_lz        = lzd15(s2_mant_res);
        c3_exp_tmp   = {1'b0, s2_exp_res};
        c3_mant_norm = 14'h0; c3_exp_norm = 7'h0; c3_guard = 1'b0; c3_round_bit = 1'b0;
        c3_sticky    = s2_sticky; c3_shift_left = 4'h0; c3_raw_shift = 4'h0;
        c3_left_ext  = 29'h0; c3_left_shifted = 29'h0;

        if (c3_is_zero) begin
            c3_exp_norm = 7'h0; c3_mant_norm = 14'h0; c3_guard = 1'b0; c3_round_bit = 1'b0; c3_sticky = 1'b0;
        end else if (s2_mant_res[14]) begin
            c3_exp_norm = c3_exp_tmp + 7'h1; c3_mant_norm = s2_mant_res[14:1];
            c3_guard = s2_mant_res[0]; c3_round_bit = 1'b0; c3_sticky = s2_sticky;
        end else begin
            c3_raw_shift = (c3_lz > 4'd0) ? (c3_lz - 4'd1) : 4'd0;
            if (c3_exp_tmp <= 7'h1) c3_shift_left = 4'd0;
            else if ({3'b0, c3_raw_shift} >= c3_exp_tmp)
                c3_shift_left = (c3_exp_tmp - 7'h1) > 7'hF ? 4'hF : c3_exp_tmp[3:0] - 4'd1;
            else c3_shift_left = c3_raw_shift;
            c3_left_ext = {s2_mant_res, 14'h0};
            case (c3_shift_left)
                4'd0:    c3_left_shifted = c3_left_ext;
                4'd1:    c3_left_shifted = c3_left_ext << 1;
                4'd2:    c3_left_shifted = c3_left_ext << 2;
                4'd3:    c3_left_shifted = c3_left_ext << 3;
                4'd4:    c3_left_shifted = c3_left_ext << 4;
                4'd5:    c3_left_shifted = c3_left_ext << 5;
                4'd6:    c3_left_shifted = c3_left_ext << 6;
                4'd7:    c3_left_shifted = c3_left_ext << 7;
                4'd8:    c3_left_shifted = c3_left_ext << 8;
                4'd9:    c3_left_shifted = c3_left_ext << 9;
                4'd10:   c3_left_shifted = c3_left_ext << 10;
                4'd11:   c3_left_shifted = c3_left_ext << 11;
                4'd12:   c3_left_shifted = c3_left_ext << 12;
                4'd13:   c3_left_shifted = c3_left_ext << 13;
                default: c3_left_shifted = c3_left_ext << 14;
            endcase
            c3_mant_norm = c3_left_shifted[27:14];
            c3_guard     = c3_left_shifted[13];
            c3_round_bit = c3_left_shifted[12];
            c3_sticky    = |c3_left_shifted[11:0] | s2_sticky;
            c3_exp_norm  = c3_exp_tmp - {3'b0, c3_shift_left};
        end
    end

    reg        s3_valid, s3_sign_res;
    reg [6:0]  s3_exp_res;
    reg [13:0] s3_mant_norm;
    reg        s3_guard, s3_round_bit, s3_sticky;
    reg        s3_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0; s3_sign_res <= 1'b0; s3_exp_res <= 7'h0; s3_mant_norm <= 14'h0;
            s3_guard <= 1'b0; s3_round_bit <= 1'b0; s3_sticky <= 1'b0; s3_zero <= 1'b0;
        end else begin
            s3_valid <= s2_valid; s3_sign_res <= s2_sign_res; s3_exp_res <= c3_exp_norm;
            s3_mant_norm <= c3_mant_norm; s3_guard <= c3_guard; s3_round_bit <= c3_round_bit;
            s3_sticky <= c3_sticky; s3_zero <= c3_is_zero;
        end
    end

    reg        c4_lsb, c4_round_up;
    reg [14:0] c4_mant_inc;
    reg [6:0]  c4_exp_out;
    reg [13:0] c4_mant_out;

    always @(*) begin
        c4_lsb      = s3_mant_norm[0];
        c4_round_up = s3_guard & (s3_round_bit | s3_sticky | c4_lsb);
        c4_mant_inc = {1'b0, s3_mant_norm} + {14'h0, c4_round_up};
        if (c4_mant_inc[14]) begin
            c4_exp_out = s3_exp_res + 7'h1; c4_mant_out = 14'h2000;
        end else begin
            c4_exp_out = s3_exp_res; c4_mant_out = c4_mant_inc[13:0];
        end
    end

    reg        s4_valid, s4_sign_res;
    reg [6:0]  s4_exp_res;
    reg [13:0] s4_mant_rnd;
    reg        s4_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_valid <= 1'b0; s4_sign_res <= 1'b0; s4_exp_res <= 7'h0; s4_mant_rnd <= 14'h0; s4_zero <= 1'b0;
        end else begin
            s4_valid <= s3_valid; s4_sign_res <= s3_sign_res; s4_exp_res <= c4_exp_out;
            s4_mant_rnd <= c4_mant_out; s4_zero <= s3_zero;
        end
    end

    reg        c5_overflow, c5_underflow;
    reg [19:0] c5_result;

    always @(*) begin
        c5_overflow  = (s4_exp_res >= 7'd63) && !s4_zero;
        c5_underflow = (s4_exp_res == 7'h0) && !s4_zero;
        if (s4_zero)            c5_result = 20'h0;
        else if (c5_overflow)   c5_result = {s4_sign_res, 6'd62, 13'h1FFF};
        else if (c5_underflow)  c5_result = 20'h0;
        else                    c5_result = {s4_sign_res, s4_exp_res[5:0], s4_mant_rnd[12:0]};
    end

    reg        s5_valid;
    reg [19:0] s5_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s5_valid <= 1'b0; s5_result <= 20'h0;
        end else begin
            s5_valid <= s4_valid; s5_result <= c5_result;
        end
    end

    assign out_valid = s5_valid;
    assign result    = s5_result;
endmodule


// =============================================================================
// SEM20 Divider (16-cycle restoring pipeline)
// =============================================================================
module sem20_div (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [19:0] a,
    input  wire [19:0] b,
    output reg  [19:0] result,
    output reg         out_valid
);
    localparam integer NBITS = 13;

    reg        s0_valid, s0_sign, s0_za, s0_zb;
    reg [5:0]  s0_ea, s0_eb;
    reg [13:0] s0_ma, s0_mb;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0; s0_sign <= 1'b0; s0_za <= 1'b0; s0_zb <= 1'b0;
            s0_ea <= 6'd0; s0_eb <= 6'd0; s0_ma <= 14'd0; s0_mb <= 14'd0;
        end else begin
            s0_valid <= in_valid; s0_sign <= a[19] ^ b[19];
            s0_za <= (a[18:0] == 19'd0); s0_zb <= (b[18:0] == 19'd0);
            s0_ea <= a[18:13]; s0_eb <= b[18:13];
            s0_ma <= (a[18:0] == 19'd0) ? 14'd0 : {1'b1, a[12:0]};
            s0_mb <= (b[18:0] == 19'd0) ? 14'd0 : {1'b1, b[12:0]};
        end
    end

    reg        s1_valid, s1_sign, s1_za, s1_zb;
    reg [7:0]  s1_exp;
    reg [13:0] s1_D, s1_R;

    wire [14:0] c1_two_a         = {s0_ma, 1'b0};
    wire [14:0] c1_r_two_a_minus_b = c1_two_a - {1'b0, s0_mb};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_sign <= 1'b0; s1_za <= 1'b0; s1_zb <= 1'b0;
            s1_exp <= 8'd0; s1_D <= 14'd0; s1_R <= 14'd0;
        end else begin
            s1_valid <= s0_valid; s1_sign <= s0_sign; s1_za <= s0_za; s1_zb <= s0_zb; s1_D <= s0_mb;
            if (s0_za || s0_zb) begin
                s1_R <= 14'd0; s1_exp <= 8'd0;
            end else if (s0_ma >= s0_mb) begin
                s1_R <= s0_ma - s0_mb;
                s1_exp <= {2'b00, s0_ea} - {2'b00, s0_eb} + 8'd95;
            end else begin
                s1_R <= c1_r_two_a_minus_b[13:0];
                s1_exp <= {2'b00, s0_ea} - {2'b00, s0_eb} + 8'd94;
            end
        end
    end

    // Restoring divider pipeline stages
    reg        dv_valid [0:NBITS-1];
    reg        dv_sign  [0:NBITS-1];
    reg [7:0]  dv_exp   [0:NBITS-1];
    reg [13:0] dv_D     [0:NBITS-1];
    reg [13:0] dv_R     [0:NBITS-1];
    reg [12:0] dv_Q     [0:NBITS-1];
    reg        dv_za    [0:NBITS-1];
    reg        dv_zb    [0:NBITS-1];

    // Stage 0 of divider pipeline
    wire [15:0] dv0_r16  = {1'b0, s1_R, 1'b0};
    wire [15:0] dv0_rsub = dv0_r16 - {2'b00, s1_D};
    wire        dv0_qbit = ~dv0_rsub[15];
    wire [13:0] dv0_Rnxt = dv0_qbit ? dv0_rsub[13:0] : dv0_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[0] <= 1'b0; dv_sign[0] <= 1'b0; dv_exp[0] <= 8'd0;
            dv_D[0] <= 14'd0; dv_R[0] <= 14'd0; dv_Q[0] <= 13'd0;
            dv_za[0] <= 1'b0; dv_zb[0] <= 1'b0;
        end else begin
            dv_valid[0] <= s1_valid; dv_sign[0] <= s1_sign; dv_exp[0] <= s1_exp;
            dv_D[0] <= s1_D; dv_R[0] <= dv0_Rnxt; dv_Q[0] <= {12'd0, dv0_qbit};
            dv_za[0] <= s1_za; dv_zb[0] <= s1_zb;
        end
    end

    // Stage 1
    wire [15:0] dv1_r16  = {1'b0, dv_R[0], 1'b0};
    wire [15:0] dv1_rsub = dv1_r16 - {2'b00, dv_D[0]};
    wire        dv1_qbit = ~dv1_rsub[15];
    wire [13:0] dv1_Rnxt = dv1_qbit ? dv1_rsub[13:0] : dv1_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[1] <= 1'b0; dv_sign[1] <= 1'b0; dv_exp[1] <= 8'd0;
            dv_D[1] <= 14'd0; dv_R[1] <= 14'd0; dv_Q[1] <= 13'd0;
            dv_za[1] <= 1'b0; dv_zb[1] <= 1'b0;
        end else begin
            dv_valid[1] <= dv_valid[0]; dv_sign[1] <= dv_sign[0]; dv_exp[1] <= dv_exp[0];
            dv_D[1] <= dv_D[0]; dv_R[1] <= dv1_Rnxt; dv_Q[1] <= {dv_Q[0][11:0], dv1_qbit};
            dv_za[1] <= dv_za[0]; dv_zb[1] <= dv_zb[0];
        end
    end

    // Stage 2
    wire [15:0] dv2_r16  = {1'b0, dv_R[1], 1'b0};
    wire [15:0] dv2_rsub = dv2_r16 - {2'b00, dv_D[1]};
    wire        dv2_qbit = ~dv2_rsub[15];
    wire [13:0] dv2_Rnxt = dv2_qbit ? dv2_rsub[13:0] : dv2_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[2] <= 1'b0; dv_sign[2] <= 1'b0; dv_exp[2] <= 8'd0;
            dv_D[2] <= 14'd0; dv_R[2] <= 14'd0; dv_Q[2] <= 13'd0;
            dv_za[2] <= 1'b0; dv_zb[2] <= 1'b0;
        end else begin
            dv_valid[2] <= dv_valid[1]; dv_sign[2] <= dv_sign[1]; dv_exp[2] <= dv_exp[1];
            dv_D[2] <= dv_D[1]; dv_R[2] <= dv2_Rnxt; dv_Q[2] <= {dv_Q[1][11:0], dv2_qbit};
            dv_za[2] <= dv_za[1]; dv_zb[2] <= dv_zb[1];
        end
    end

    // Stage 3
    wire [15:0] dv3_r16  = {1'b0, dv_R[2], 1'b0};
    wire [15:0] dv3_rsub = dv3_r16 - {2'b00, dv_D[2]};
    wire        dv3_qbit = ~dv3_rsub[15];
    wire [13:0] dv3_Rnxt = dv3_qbit ? dv3_rsub[13:0] : dv3_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[3] <= 1'b0; dv_sign[3] <= 1'b0; dv_exp[3] <= 8'd0;
            dv_D[3] <= 14'd0; dv_R[3] <= 14'd0; dv_Q[3] <= 13'd0;
            dv_za[3] <= 1'b0; dv_zb[3] <= 1'b0;
        end else begin
            dv_valid[3] <= dv_valid[2]; dv_sign[3] <= dv_sign[2]; dv_exp[3] <= dv_exp[2];
            dv_D[3] <= dv_D[2]; dv_R[3] <= dv3_Rnxt; dv_Q[3] <= {dv_Q[2][11:0], dv3_qbit};
            dv_za[3] <= dv_za[2]; dv_zb[3] <= dv_zb[2];
        end
    end

    // Stage 4
    wire [15:0] dv4_r16  = {1'b0, dv_R[3], 1'b0};
    wire [15:0] dv4_rsub = dv4_r16 - {2'b00, dv_D[3]};
    wire        dv4_qbit = ~dv4_rsub[15];
    wire [13:0] dv4_Rnxt = dv4_qbit ? dv4_rsub[13:0] : dv4_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[4] <= 1'b0; dv_sign[4] <= 1'b0; dv_exp[4] <= 8'd0;
            dv_D[4] <= 14'd0; dv_R[4] <= 14'd0; dv_Q[4] <= 13'd0;
            dv_za[4] <= 1'b0; dv_zb[4] <= 1'b0;
        end else begin
            dv_valid[4] <= dv_valid[3]; dv_sign[4] <= dv_sign[3]; dv_exp[4] <= dv_exp[3];
            dv_D[4] <= dv_D[3]; dv_R[4] <= dv4_Rnxt; dv_Q[4] <= {dv_Q[3][11:0], dv4_qbit};
            dv_za[4] <= dv_za[3]; dv_zb[4] <= dv_zb[3];
        end
    end

    // Stage 5
    wire [15:0] dv5_r16  = {1'b0, dv_R[4], 1'b0};
    wire [15:0] dv5_rsub = dv5_r16 - {2'b00, dv_D[4]};
    wire        dv5_qbit = ~dv5_rsub[15];
    wire [13:0] dv5_Rnxt = dv5_qbit ? dv5_rsub[13:0] : dv5_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[5] <= 1'b0; dv_sign[5] <= 1'b0; dv_exp[5] <= 8'd0;
            dv_D[5] <= 14'd0; dv_R[5] <= 14'd0; dv_Q[5] <= 13'd0;
            dv_za[5] <= 1'b0; dv_zb[5] <= 1'b0;
        end else begin
            dv_valid[5] <= dv_valid[4]; dv_sign[5] <= dv_sign[4]; dv_exp[5] <= dv_exp[4];
            dv_D[5] <= dv_D[4]; dv_R[5] <= dv5_Rnxt; dv_Q[5] <= {dv_Q[4][11:0], dv5_qbit};
            dv_za[5] <= dv_za[4]; dv_zb[5] <= dv_zb[4];
        end
    end

    // Stage 6
    wire [15:0] dv6_r16  = {1'b0, dv_R[5], 1'b0};
    wire [15:0] dv6_rsub = dv6_r16 - {2'b00, dv_D[5]};
    wire        dv6_qbit = ~dv6_rsub[15];
    wire [13:0] dv6_Rnxt = dv6_qbit ? dv6_rsub[13:0] : dv6_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[6] <= 1'b0; dv_sign[6] <= 1'b0; dv_exp[6] <= 8'd0;
            dv_D[6] <= 14'd0; dv_R[6] <= 14'd0; dv_Q[6] <= 13'd0;
            dv_za[6] <= 1'b0; dv_zb[6] <= 1'b0;
        end else begin
            dv_valid[6] <= dv_valid[5]; dv_sign[6] <= dv_sign[5]; dv_exp[6] <= dv_exp[5];
            dv_D[6] <= dv_D[5]; dv_R[6] <= dv6_Rnxt; dv_Q[6] <= {dv_Q[5][11:0], dv6_qbit};
            dv_za[6] <= dv_za[5]; dv_zb[6] <= dv_zb[5];
        end
    end

    // Stage 7
    wire [15:0] dv7_r16  = {1'b0, dv_R[6], 1'b0};
    wire [15:0] dv7_rsub = dv7_r16 - {2'b00, dv_D[6]};
    wire        dv7_qbit = ~dv7_rsub[15];
    wire [13:0] dv7_Rnxt = dv7_qbit ? dv7_rsub[13:0] : dv7_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[7] <= 1'b0; dv_sign[7] <= 1'b0; dv_exp[7] <= 8'd0;
            dv_D[7] <= 14'd0; dv_R[7] <= 14'd0; dv_Q[7] <= 13'd0;
            dv_za[7] <= 1'b0; dv_zb[7] <= 1'b0;
        end else begin
            dv_valid[7] <= dv_valid[6]; dv_sign[7] <= dv_sign[6]; dv_exp[7] <= dv_exp[6];
            dv_D[7] <= dv_D[6]; dv_R[7] <= dv7_Rnxt; dv_Q[7] <= {dv_Q[6][11:0], dv7_qbit};
            dv_za[7] <= dv_za[6]; dv_zb[7] <= dv_zb[6];
        end
    end

    // Stage 8
    wire [15:0] dv8_r16  = {1'b0, dv_R[7], 1'b0};
    wire [15:0] dv8_rsub = dv8_r16 - {2'b00, dv_D[7]};
    wire        dv8_qbit = ~dv8_rsub[15];
    wire [13:0] dv8_Rnxt = dv8_qbit ? dv8_rsub[13:0] : dv8_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[8] <= 1'b0; dv_sign[8] <= 1'b0; dv_exp[8] <= 8'd0;
            dv_D[8] <= 14'd0; dv_R[8] <= 14'd0; dv_Q[8] <= 13'd0;
            dv_za[8] <= 1'b0; dv_zb[8] <= 1'b0;
        end else begin
            dv_valid[8] <= dv_valid[7]; dv_sign[8] <= dv_sign[7]; dv_exp[8] <= dv_exp[7];
            dv_D[8] <= dv_D[7]; dv_R[8] <= dv8_Rnxt; dv_Q[8] <= {dv_Q[7][11:0], dv8_qbit};
            dv_za[8] <= dv_za[7]; dv_zb[8] <= dv_zb[7];
        end
    end

    // Stage 9
    wire [15:0] dv9_r16  = {1'b0, dv_R[8], 1'b0};
    wire [15:0] dv9_rsub = dv9_r16 - {2'b00, dv_D[8]};
    wire        dv9_qbit = ~dv9_rsub[15];
    wire [13:0] dv9_Rnxt = dv9_qbit ? dv9_rsub[13:0] : dv9_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[9] <= 1'b0; dv_sign[9] <= 1'b0; dv_exp[9] <= 8'd0;
            dv_D[9] <= 14'd0; dv_R[9] <= 14'd0; dv_Q[9] <= 13'd0;
            dv_za[9] <= 1'b0; dv_zb[9] <= 1'b0;
        end else begin
            dv_valid[9] <= dv_valid[8]; dv_sign[9] <= dv_sign[8]; dv_exp[9] <= dv_exp[8];
            dv_D[9] <= dv_D[8]; dv_R[9] <= dv9_Rnxt; dv_Q[9] <= {dv_Q[8][11:0], dv9_qbit};
            dv_za[9] <= dv_za[8]; dv_zb[9] <= dv_zb[8];
        end
    end

    // Stage 10
    wire [15:0] dv10_r16  = {1'b0, dv_R[9], 1'b0};
    wire [15:0] dv10_rsub = dv10_r16 - {2'b00, dv_D[9]};
    wire        dv10_qbit = ~dv10_rsub[15];
    wire [13:0] dv10_Rnxt = dv10_qbit ? dv10_rsub[13:0] : dv10_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[10] <= 1'b0; dv_sign[10] <= 1'b0; dv_exp[10] <= 8'd0;
            dv_D[10] <= 14'd0; dv_R[10] <= 14'd0; dv_Q[10] <= 13'd0;
            dv_za[10] <= 1'b0; dv_zb[10] <= 1'b0;
        end else begin
            dv_valid[10] <= dv_valid[9]; dv_sign[10] <= dv_sign[9]; dv_exp[10] <= dv_exp[9];
            dv_D[10] <= dv_D[9]; dv_R[10] <= dv10_Rnxt; dv_Q[10] <= {dv_Q[9][11:0], dv10_qbit};
            dv_za[10] <= dv_za[9]; dv_zb[10] <= dv_zb[9];
        end
    end

    // Stage 11
    wire [15:0] dv11_r16  = {1'b0, dv_R[10], 1'b0};
    wire [15:0] dv11_rsub = dv11_r16 - {2'b00, dv_D[10]};
    wire        dv11_qbit = ~dv11_rsub[15];
    wire [13:0] dv11_Rnxt = dv11_qbit ? dv11_rsub[13:0] : dv11_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[11] <= 1'b0; dv_sign[11] <= 1'b0; dv_exp[11] <= 8'd0;
            dv_D[11] <= 14'd0; dv_R[11] <= 14'd0; dv_Q[11] <= 13'd0;
            dv_za[11] <= 1'b0; dv_zb[11] <= 1'b0;
        end else begin
            dv_valid[11] <= dv_valid[10]; dv_sign[11] <= dv_sign[10]; dv_exp[11] <= dv_exp[10];
            dv_D[11] <= dv_D[10]; dv_R[11] <= dv11_Rnxt; dv_Q[11] <= {dv_Q[10][11:0], dv11_qbit};
            dv_za[11] <= dv_za[10]; dv_zb[11] <= dv_zb[10];
        end
    end

    // Stage 12 (NBITS-1 = 12)
    wire [15:0] dv12_r16  = {1'b0, dv_R[11], 1'b0};
    wire [15:0] dv12_rsub = dv12_r16 - {2'b00, dv_D[11]};
    wire        dv12_qbit = ~dv12_rsub[15];
    wire [13:0] dv12_Rnxt = dv12_qbit ? dv12_rsub[13:0] : dv12_r16[13:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dv_valid[12] <= 1'b0; dv_sign[12] <= 1'b0; dv_exp[12] <= 8'd0;
            dv_D[12] <= 14'd0; dv_R[12] <= 14'd0; dv_Q[12] <= 13'd0;
            dv_za[12] <= 1'b0; dv_zb[12] <= 1'b0;
        end else begin
            dv_valid[12] <= dv_valid[11]; dv_sign[12] <= dv_sign[11]; dv_exp[12] <= dv_exp[11];
            dv_D[12] <= dv_D[11]; dv_R[12] <= dv12_Rnxt; dv_Q[12] <= {dv_Q[11][11:0], dv12_qbit};
            dv_za[12] <= dv_za[11]; dv_zb[12] <= dv_zb[11];
        end
    end

    wire [7:0] c_exp_out = dv_exp[NBITS-1] - 8'd64;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; result <= 20'd0;
        end else begin
            out_valid <= dv_valid[NBITS-1];
            if (dv_zb[NBITS-1])
                result <= {dv_sign[NBITS-1], 6'd62, 13'h1FFF};
            else if (dv_za[NBITS-1])
                result <= 20'd0;
            else if (dv_exp[NBITS-1] > 8'd126)
                result <= {dv_sign[NBITS-1], 6'd62, 13'h1FFF};
            else if (dv_exp[NBITS-1] < 8'd65)
                result <= 20'd0;
            else
                result <= {dv_sign[NBITS-1], c_exp_out[5:0], dv_Q[NBITS-1]};
        end
    end
endmodule


// =============================================================================
// delay_line - Parameterized shift register
// =============================================================================
module delay_line #(
    parameter integer WIDTH = 20,
    parameter integer DEPTH = 1
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH-1:0] q,
    input  wire             v_in,
    output wire             v_out
);
    reg [WIDTH-1:0] data_pipe  [0:DEPTH-1];
    reg             valid_pipe [0:DEPTH-1];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                data_pipe[i]  <= {WIDTH{1'b0}};
                valid_pipe[i] <= 1'b0;
            end
        end else begin
            data_pipe[0]  <= d;
            valid_pipe[0] <= v_in;
            for (i = 1; i < DEPTH; i = i + 1) begin
                data_pipe[i]  <= data_pipe[i-1];
                valid_pipe[i] <= valid_pipe[i-1];
            end
        end
    end

    assign q     = data_pipe[DEPTH-1];
    assign v_out = valid_pipe[DEPTH-1];
endmodule


// =============================================================================
// poly_engine_top - 4th-order Taylor series for exp(x)
// Latency: 48 cycles
// =============================================================================
module poly_engine_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [19:0] x,
    output wire [19:0] y,
    output wire        out_valid
);
    localparam [19:0] C_1_0  = 20'h3E000;
    localparam [19:0] C_0_5  = 20'h3C000;
    localparam [19:0] C_1_6  = 20'h38AAB;
    localparam [19:0] C_1_24 = 20'h34AAB;

    reg [19:0] x_pipe [0:35];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 36; i = i + 1) x_pipe[i] <= 20'h0;
        end else begin
            x_pipe[0] <= x;
            for (i = 1; i < 36; i = i + 1) x_pipe[i] <= x_pipe[i-1];
        end
    end

    wire [19:0] mul4_out, add4_out;   wire v_mul4, v_add4;
    wire [19:0] mul3_out, add3_out;   wire v_mul3, v_add3;
    wire [19:0] mul2_out, add2_out;   wire v_mul2, v_add2;
    wire [19:0] mul1_out, add1_out;   wire v_mul1, v_add1;

    sem20_mul mul4 (.clk(clk), .rst_n(rst_n), .in_valid(in_valid),  .a(x),          .b(C_1_24), .product(mul4_out), .valid_out(v_mul4));
    sem20_add add4 (.clk(clk), .rst_n(rst_n), .in_valid(v_mul4),    .a(C_1_6),      .b(mul4_out), .result(add4_out), .out_valid(v_add4));

    sem20_mul mul3 (.clk(clk), .rst_n(rst_n), .in_valid(v_add4),    .a(x_pipe[11]), .b(add4_out), .product(mul3_out), .valid_out(v_mul3));
    sem20_add add3 (.clk(clk), .rst_n(rst_n), .in_valid(v_mul3),    .a(C_0_5),      .b(mul3_out), .result(add3_out), .out_valid(v_add3));

    sem20_mul mul2 (.clk(clk), .rst_n(rst_n), .in_valid(v_add3),    .a(x_pipe[23]), .b(add3_out), .product(mul2_out), .valid_out(v_mul2));
    sem20_add add2 (.clk(clk), .rst_n(rst_n), .in_valid(v_mul2),    .a(C_1_0),      .b(mul2_out), .result(add2_out), .out_valid(v_add2));

    sem20_mul mul1 (.clk(clk), .rst_n(rst_n), .in_valid(v_add2),    .a(x_pipe[35]), .b(add2_out), .product(mul1_out), .valid_out(v_mul1));
    sem20_add add1 (.clk(clk), .rst_n(rst_n), .in_valid(v_mul1),    .a(C_1_0),      .b(mul1_out), .result(add1_out), .out_valid(v_add1));

    assign y         = add1_out;
    assign out_valid = v_add1;
endmodule


// =============================================================================
// exp_neg_hybrid - exp(-x) with hybrid Taylor/LUT/zero path
// Latency: 48 cycles on all three paths.
// =============================================================================
module exp_neg_hybrid (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [19:0] x,
    output reg  [19:0] result,
    output reg         out_valid
);
    localparam [19:0] NEG_1_2   = 20'hBE666;
    localparam [19:0] C_63div28 = 20'h4450F;
    localparam [19:0] C_1_2     = 20'h3E666;
    localparam [19:0] C_8_0     = 20'h44000;

    reg le_1_2, gt_4;
    reg [1:0] sel_comb;

    always @(*) begin
        if (x[19] == 1'b1) le_1_2 = 1'b1;
        else if (x[18:13] < C_1_2[18:13]) le_1_2 = 1'b1;
        else if (x[18:13] > C_1_2[18:13]) le_1_2 = 1'b0;
        else le_1_2 = (x[12:0] <= C_1_2[12:0]);
    end
    always @(*) begin
        if (x[19] == 1'b1) gt_4 = 1'b0;
        else if (x[18:13] > C_8_0[18:13]) gt_4 = 1'b1;
        else if (x[18:13] < C_8_0[18:13]) gt_4 = 1'b0;
        else gt_4 = (x[12:0] > C_8_0[12:0]);
    end
    always @(*) begin
        if (le_1_2) sel_comb = 2'd0;
        else if (gt_4) sel_comb = 2'd2;
        else sel_comb = 2'd1;
    end

    reg [1:0] sel_pipe   [0:47];
    reg       valid_pipe [0:47];

    integer pi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pi = 0; pi < 48; pi = pi + 1) begin
                sel_pipe[pi] <= 2'd0; valid_pipe[pi] <= 1'b0;
            end
        end else begin
            sel_pipe[0] <= sel_comb; valid_pipe[0] <= in_valid;
            for (pi = 1; pi < 48; pi = pi + 1) begin
                sel_pipe[pi] <= sel_pipe[pi-1]; valid_pipe[pi] <= valid_pipe[pi-1];
            end
        end
    end

    wire [1:0] sel_delayed   = sel_pipe[47];
    wire       valid_delayed = valid_pipe[47];

    // --- Taylor path (sel==0) ---
    wire [19:0] x_neg = {~x[19], x[18:0]};

    wire [19:0] taylor_result;
    wire        taylor_valid;
    poly_engine_top u_taylor (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .x(x_neg), .y(taylor_result), .out_valid(taylor_valid));

    // --- LUT path (sel==1) ---
    wire [19:0] x_minus_12;
    wire        add_valid;
    sem20_add u_add (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .a(x), .b(NEG_1_2), .result(x_minus_12), .out_valid(add_valid));

    wire [19:0] t_product;
    wire        mul_valid;
    sem20_mul u_mul (.clk(clk), .rst_n(rst_n), .in_valid(add_valid), .a(x_minus_12), .b(C_63div28), .product(t_product), .valid_out(mul_valid));

    wire [5:0]  t_exp  = t_product[18:13];
    wire [12:0] t_man  = t_product[12:0];
    wire [13:0] mant_full = {1'b1, t_man};
    wire signed [6:0] shift = $signed({1'b0, t_exp}) - 7'sd31;

    reg [31:0] value_fixed;
    reg [5:0]  index;

    always @(*) begin
        if (shift >= 0) value_fixed = mant_full << shift;
        else            value_fixed = mant_full >> (-shift);
        begin
            reg [5:0] int_part, index_raw;
            reg       round_bit;
            int_part  = value_fixed[31:13] > 6'd63 ? 6'd63 : value_fixed[31:13];
            round_bit = value_fixed[12];
            index_raw = int_part + round_bit;
            index     = (index_raw > 6'd63) ? 6'd63 : index_raw;
        end
    end

    reg [19:0] lut_out_comb;
    always @(*) begin
        case (index)
            6'd00: lut_out_comb = 20'h3A68E; 6'd01: lut_out_comb = 20'h3A29C;
            6'd02: lut_out_comb = 20'h39E22; 6'd03: lut_out_comb = 20'h397C7;
            6'd04: lut_out_comb = 20'h39212; 6'd05: lut_out_comb = 20'h38CF3;
            6'd06: lut_out_comb = 20'h38859; 6'd07: lut_out_comb = 20'h38438;
            6'd08: lut_out_comb = 20'h38084; 6'd09: lut_out_comb = 20'h37A60;
            6'd10: lut_out_comb = 20'h37467; 6'd11: lut_out_comb = 20'h36F0B;
            6'd12: lut_out_comb = 20'h36A3A; 6'd13: lut_out_comb = 20'h365E8;
            6'd14: lut_out_comb = 20'h36207; 6'd15: lut_out_comb = 20'h35D18;
            6'd16: lut_out_comb = 20'h356D8; 6'd17: lut_out_comb = 20'h3513B;
            6'd18: lut_out_comb = 20'h34C32; 6'd19: lut_out_comb = 20'h347AC;
            6'd20: lut_out_comb = 20'h3439D; 6'd21: lut_out_comb = 20'h33FF1;
            6'd22: lut_out_comb = 20'h33966; 6'd23: lut_out_comb = 20'h33387;
            6'd24: lut_out_comb = 20'h32E41; 6'd25: lut_out_comb = 20'h32985;
            6'd26: lut_out_comb = 20'h32546; 6'd27: lut_out_comb = 20'h32176;
            6'd28: lut_out_comb = 20'h31C12; 6'd29: lut_out_comb = 20'h315ED;
            6'd30: lut_out_comb = 20'h31069; 6'd31: lut_out_comb = 20'h30B75;
            6'd32: lut_out_comb = 20'h30702; 6'd33: lut_out_comb = 20'h30305;
            6'd34: lut_out_comb = 20'h2FEDF; 6'd35: lut_out_comb = 20'h2F870;
            6'd36: lut_out_comb = 20'h2F2AA; 6'd37: lut_out_comb = 20'h2ED7B;
            6'd38: lut_out_comb = 20'h2E8D4; 6'd39: lut_out_comb = 20'h2E4A6;
            6'd40: lut_out_comb = 20'h2E0E6; 6'd41: lut_out_comb = 20'h2DB11;
            6'd42: lut_out_comb = 20'h2D506; 6'd43: lut_out_comb = 20'h2CF99;
            6'd44: lut_out_comb = 20'h2CABA; 6'd45: lut_out_comb = 20'h2C65B;
            6'd46: lut_out_comb = 20'h2C26F; 6'd47: lut_out_comb = 20'h2BDD1;
            6'd48: lut_out_comb = 20'h2B77E; 6'd49: lut_out_comb = 20'h2B1D1;
            6'd50: lut_out_comb = 20'h2ACB8; 6'd51: lut_out_comb = 20'h2A825;
            6'd52: lut_out_comb = 20'h2A409; 6'd53: lut_out_comb = 20'h2A059;
            6'd54: lut_out_comb = 20'h29A14; 6'd55: lut_out_comb = 20'h29423;
            6'd56: lut_out_comb = 20'h28ECD; 6'd57: lut_out_comb = 20'h28A03;
            6'd58: lut_out_comb = 20'h285B7; 6'd59: lut_out_comb = 20'h281DB;
            6'd60: lut_out_comb = 20'h27CC8; 6'd61: lut_out_comb = 20'h27690;
            6'd62: lut_out_comb = 20'h270FB; 6'd63: lut_out_comb = 20'h26BF8;
            default: lut_out_comb = 20'h00000;
        endcase
    end

    reg [19:0] lut_out_reg;
    reg        lut_valid_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_out_reg <= 20'd0; lut_valid_reg <= 1'b0;
        end else begin
            lut_out_reg <= lut_out_comb; lut_valid_reg <= mul_valid;
        end
    end

    reg [19:0] lut_pipe       [0:34];
    reg        lut_valid_pipe [0:34];
    integer li;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (li = 0; li < 35; li = li + 1) begin
                lut_pipe[li] <= 20'd0; lut_valid_pipe[li] <= 1'b0;
            end
        end else begin
            lut_pipe[0] <= lut_out_reg; lut_valid_pipe[0] <= lut_valid_reg;
            for (li = 1; li < 35; li = li + 1) begin
                lut_pipe[li] <= lut_pipe[li-1]; lut_valid_pipe[li] <= lut_valid_pipe[li-1];
            end
        end
    end

    wire [19:0] lut_result = lut_pipe[34];
    wire        lut_valid  = lut_valid_pipe[34];

    always @(*) begin
        result = 20'd0; out_valid = 1'b0;
        case (sel_delayed)
            2'd0: begin result = taylor_result; out_valid = taylor_valid; end
            2'd1: begin result = lut_result;    out_valid = lut_valid;    end
            2'd2: begin result = 20'd0;         out_valid = valid_delayed; end
            default: begin result = 20'd0;      out_valid = 1'b0;         end
        endcase
    end
endmodule


// =============================================================================
// exp_top - exp(-0.5 * z^2) wrapper
// Latency: 60 cycles
// =============================================================================
module exp_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [19:0] z,
    output wire [19:0] y,
    output wire        out_valid
);
    localparam [19:0] HALF = 20'h3C000;

    wire [19:0] z2;
    wire        z2_valid;
    sem20_mul u_mul_z2   (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .a(z),  .b(z),    .product(z2),  .valid_out(z2_valid));

    wire [19:0] x;
    wire        x_valid;
    sem20_mul u_mul_half (.clk(clk), .rst_n(rst_n), .in_valid(z2_valid), .a(z2), .b(HALF), .product(x),   .valid_out(x_valid));

    exp_neg_hybrid u_exp (.clk(clk), .rst_n(rst_n), .in_valid(x_valid),  .x(x),            .result(y),    .out_valid(out_valid));
endmodule


// =============================================================================
// q8p8_to_sem20 - Q8.8 to SEM20 conversion (3-cycle)
// =============================================================================
module q8p8_to_sem20 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire signed [15:0] in_q8p8,
    input  wire               in_valid,
    output wire [19:0]        out_sem20,
    output wire               out_valid
);
    localparam integer EXP_BITS = 6;
    localparam integer MAN_BITS = 13;

    reg        s0_sign;
    reg [15:0] s0_abs;
    reg        s0_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0; s0_sign <= 1'b0; s0_abs <= 16'h0;
        end else begin
            s0_valid <= in_valid; s0_sign <= in_q8p8[15];
            if (in_q8p8 == 16'sh8000) s0_abs <= 16'h8000;
            else s0_abs <= in_q8p8[15] ? (-in_q8p8) : in_q8p8;
        end
    end

    wire [3:0] lzd_msb_comb;
    wire       lzd_valid_comb;
    lzd16 lzd_inst (.in(s0_abs), .pos(lzd_msb_comb), .valid(lzd_valid_comb));

    reg        s1_sign;
    reg [15:0] s1_abs;
    reg        s1_valid;
    reg [3:0]  s1_msb;
    reg        s1_valid_flag;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_sign <= 1'b0; s1_abs <= 16'h0; s1_msb <= 4'h0; s1_valid_flag <= 1'b0;
        end else begin
            s1_valid <= s0_valid; s1_sign <= s0_sign; s1_abs <= s0_abs;
            s1_msb <= lzd_msb_comb; s1_valid_flag <= lzd_valid_comb;
        end
    end

    reg [EXP_BITS-1:0] c2_E;
    reg [16:0]         c2_norm_wide;
    reg [MAN_BITS-1:0] c2_mant;
    reg [19:0]         c2_sem_out;

    always @(*) begin
        c2_E = {2'b00, s1_msb} + 6'd23;
        if (s1_msb > 4'd13)
            c2_norm_wide = {1'b0, s1_abs} >> (s1_msb - 4'd13);
        else
            c2_norm_wide = {1'b0, s1_abs} << (4'd13 - s1_msb);
        c2_mant    = c2_norm_wide[MAN_BITS-1:0];
        c2_sem_out = {s1_sign, c2_E, c2_mant};
    end

    reg [19:0] s2_out;
    reg        s2_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_out <= 20'd0;
        end else begin
            s2_valid <= s1_valid;
            s2_out <= s1_valid_flag ? c2_sem_out : 20'd0;
        end
    end

    assign out_sem20 = s2_out;
    assign out_valid = s2_valid;
endmodule


// =============================================================================
// sem20_to_q8p8 - SEM20 to Q8.8 conversion (3-cycle)
// =============================================================================
module sem20_to_q8p8 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire [19:0]        sem_in,
    input  wire               in_valid,
    output wire signed [15:0] q88_out,
    output wire               out_valid
);
    reg               s1_valid, s1_sign, s1_is_zero;
    reg signed [31:0] s1_norm;
    reg signed [7:0]  s1_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_sign <= 1'b0; s1_is_zero <= 1'b0; s1_norm <= 32'sd0; s1_shift <= 8'sd0;
        end else begin
            s1_valid   <= in_valid;
            s1_sign    <= sem_in[19];
            s1_is_zero <= (sem_in[18:0] == 19'd0);
            s1_norm    <= (1 <<< 13) | {19'b0, sem_in[12:0]};
            s1_shift   <= $signed({1'b0, sem_in[18:13]}) - 8'sd36;
        end
    end

    reg       c2_dir;
    reg [5:0] c2_mag;
    reg [7:0] c2_abs;
    reg [47:0] c2_norm_ext;
    reg signed [47:0] c2_shifted, c2_result_comb;

    always @(*) begin
        if (s1_shift[7]) c2_abs = (~s1_shift) + 8'd1;
        else c2_abs = {1'b0, s1_shift[6:0]};
    end
    assign c2_norm_ext = {16'b0, s1_norm[31:0]};

    always @(*) begin
        c2_dir = ~s1_shift[7];
        if (c2_dir) c2_mag = (c2_abs > 8'd27) ? 6'd27 : c2_abs[5:0];
        else        c2_mag = (c2_abs > 8'd36) ? 6'd36 : c2_abs[5:0];
        c2_shifted = 48'd0;
        if (c2_dir) begin
            case (c2_mag)
                6'd0:  c2_shifted = c2_norm_ext;        6'd1:  c2_shifted = c2_norm_ext << 1;
                6'd2:  c2_shifted = c2_norm_ext << 2;   6'd3:  c2_shifted = c2_norm_ext << 3;
                6'd4:  c2_shifted = c2_norm_ext << 4;   6'd5:  c2_shifted = c2_norm_ext << 5;
                6'd6:  c2_shifted = c2_norm_ext << 6;   6'd7:  c2_shifted = c2_norm_ext << 7;
                6'd8:  c2_shifted = c2_norm_ext << 8;   6'd9:  c2_shifted = c2_norm_ext << 9;
                6'd10: c2_shifted = c2_norm_ext << 10;  6'd11: c2_shifted = c2_norm_ext << 11;
                6'd12: c2_shifted = c2_norm_ext << 12;  6'd13: c2_shifted = c2_norm_ext << 13;
                6'd14: c2_shifted = c2_norm_ext << 14;  6'd15: c2_shifted = c2_norm_ext << 15;
                6'd16: c2_shifted = c2_norm_ext << 16;  6'd17: c2_shifted = c2_norm_ext << 17;
                6'd18: c2_shifted = c2_norm_ext << 18;  6'd19: c2_shifted = c2_norm_ext << 19;
                6'd20: c2_shifted = c2_norm_ext << 20;  6'd21: c2_shifted = c2_norm_ext << 21;
                6'd22: c2_shifted = c2_norm_ext << 22;  6'd23: c2_shifted = c2_norm_ext << 23;
                6'd24: c2_shifted = c2_norm_ext << 24;  6'd25: c2_shifted = c2_norm_ext << 25;
                6'd26: c2_shifted = c2_norm_ext << 26;  6'd27: c2_shifted = c2_norm_ext << 27;
                default: c2_shifted = c2_norm_ext << 27;
            endcase
        end else begin
            case (c2_mag)
                6'd0:  c2_shifted = c2_norm_ext;        6'd1:  c2_shifted = c2_norm_ext >> 1;
                6'd2:  c2_shifted = c2_norm_ext >> 2;   6'd3:  c2_shifted = c2_norm_ext >> 3;
                6'd4:  c2_shifted = c2_norm_ext >> 4;   6'd5:  c2_shifted = c2_norm_ext >> 5;
                6'd6:  c2_shifted = c2_norm_ext >> 6;   6'd7:  c2_shifted = c2_norm_ext >> 7;
                6'd8:  c2_shifted = c2_norm_ext >> 8;   6'd9:  c2_shifted = c2_norm_ext >> 9;
                6'd10: c2_shifted = c2_norm_ext >> 10;  6'd11: c2_shifted = c2_norm_ext >> 11;
                6'd12: c2_shifted = c2_norm_ext >> 12;  6'd13: c2_shifted = c2_norm_ext >> 13;
                6'd14: c2_shifted = c2_norm_ext >> 14;  6'd15: c2_shifted = c2_norm_ext >> 15;
                6'd16: c2_shifted = c2_norm_ext >> 16;  6'd17: c2_shifted = c2_norm_ext >> 17;
                6'd18: c2_shifted = c2_norm_ext >> 18;  6'd19: c2_shifted = c2_norm_ext >> 19;
                6'd20: c2_shifted = c2_norm_ext >> 20;  6'd21: c2_shifted = c2_norm_ext >> 21;
                6'd22: c2_shifted = c2_norm_ext >> 22;  6'd23: c2_shifted = c2_norm_ext >> 23;
                6'd24: c2_shifted = c2_norm_ext >> 24;  6'd25: c2_shifted = c2_norm_ext >> 25;
                6'd26: c2_shifted = c2_norm_ext >> 26;  6'd27: c2_shifted = c2_norm_ext >> 27;
                6'd28: c2_shifted = c2_norm_ext >> 28;  6'd29: c2_shifted = c2_norm_ext >> 29;
                6'd30: c2_shifted = c2_norm_ext >> 30;  6'd31: c2_shifted = c2_norm_ext >> 31;
                6'd32: c2_shifted = c2_norm_ext >> 32;  6'd33: c2_shifted = c2_norm_ext >> 33;
                6'd34: c2_shifted = c2_norm_ext >> 34;  6'd35: c2_shifted = c2_norm_ext >> 35;
                6'd36: c2_shifted = c2_norm_ext >> 36;
                default: c2_shifted = 48'd0;
            endcase
        end
        c2_result_comb = s1_sign ? -c2_shifted : c2_shifted;
    end

    reg               s2_valid, s2_is_zero;
    reg signed [47:0] s2_result;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_is_zero <= 1'b0; s2_result <= 48'sd0;
        end else begin
            s2_valid  <= s1_valid; s2_is_zero <= s1_is_zero;
            s2_result <= s1_valid ? c2_result_comb : 48'sd0;
        end
    end

    reg               s3_valid;
    reg signed [15:0] s3_out;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0; s3_out <= 16'sd0;
        end else begin
            s3_valid <= s2_valid;
            if (s2_is_zero) s3_out <= 16'sd0;
            else if (s2_result > 48'sd32767) s3_out <= 16'h7FFF;
            else if (s2_result < -48'sd32768) s3_out <= 16'sh8000;
            else s3_out <= s2_result[15:0];
        end
    end

    assign out_valid = s3_valid;
    assign q88_out   = s3_out;
endmodule


// =============================================================================
// C2S0285_neuron_top - Core neuron pipeline
// Full pipeline latency: 100 cycles from in_valid to out_valid
// =============================================================================
module C2S0285_neuron_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [15:0] x_q8p8,
    input  wire [15:0] t_q8p8,
    input  wire [15:0] d_q8p8,
    input  wire [15:0] w_q8p8,
    output wire [15:0] out_q8p8,
    output wire        out_valid
);
    wire [19:0] x_s, t_s, d_s, w_s;
    wire        v_conv;

    q8p8_to_sem20 conv_x (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_q8p8(x_q8p8), .out_sem20(x_s), .out_valid(v_conv));
    q8p8_to_sem20 conv_t (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_q8p8(t_q8p8), .out_sem20(t_s), .out_valid(/* unused */));
    q8p8_to_sem20 conv_d (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_q8p8(d_q8p8), .out_sem20(d_s), .out_valid(/* unused */));
    q8p8_to_sem20 conv_w (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_q8p8(w_q8p8), .out_sem20(w_s), .out_valid(/* unused */));

    wire [19:0] t_neg = {~t_s[19], t_s[18:0]};

    // sub: x - t  [6 cycles]
    wire [19:0] sub_out;
    wire        sub_valid;
    sem20_add u_sub (.clk(clk), .rst_n(rst_n), .in_valid(v_conv), .a(x_s), .b(t_neg), .result(sub_out), .out_valid(sub_valid));

    // div: (x-t)/d  [16 cycles]
    wire [19:0] div_out;
    wire        div_valid;
    sem20_div u_div (.clk(clk), .rst_n(rst_n), .in_valid(sub_valid), .a(sub_out), .b(d_s), .result(div_out), .out_valid(div_valid));

    // path1 = z delayed 6 cycles
    wire [19:0] path1_out;
    wire        path1_valid;
    delay_line #(.WIDTH(20), .DEPTH(6)) d_z6 (
        .clk(clk), .rst_n(rst_n), .d(div_out), .v_in(div_valid), .q(path1_out), .v_out(path1_valid)
    );

    // exp: exp(-0.5 * z^2)  [60 cycles from div_valid]
    wire [19:0] exp_out;
    wire        exp_valid;
    exp_top u_hybrid_exp (.clk(clk), .rst_n(rst_n), .in_valid(div_valid), .z(div_out), .y(exp_out), .out_valid(exp_valid));

    // Align path1 with exp: delay by 54 cycles
    wire [19:0] path1_aligned;
    wire        path1_aligned_v;
    delay_line #(.WIDTH(20), .DEPTH(54)) d_p1 (
        .clk(clk), .rst_n(rst_n), .d(path1_out), .v_in(path1_valid), .q(path1_aligned), .v_out(path1_aligned_v)
    );

    // shape: path1 * exp  [6 cycles]
    wire [19:0] shape_out;
    wire        shape_valid;
    sem20_mul u_mul_shape (.clk(clk), .rst_n(rst_n), .in_valid(path1_aligned_v & exp_valid),
                           .a(path1_aligned), .b(exp_out), .product(shape_out), .valid_out(shape_valid));

    // Align w_s with shape_valid: 88 cycles from v_conv
    wire [19:0] w_aligned;
    wire        w_aligned_v;
    delay_line #(.WIDTH(20), .DEPTH(88)) d_w (
        .clk(clk), .rst_n(rst_n), .d(w_s), .v_in(v_conv), .q(w_aligned), .v_out(w_aligned_v)
    );

    // final: shape * w  [6 cycles]
    wire [19:0] final_out;
    wire        final_valid;
    sem20_mul u_mul_final (.clk(clk), .rst_n(rst_n), .in_valid(shape_valid & w_aligned_v),
                           .a(shape_out), .b(w_aligned), .product(final_out), .valid_out(final_valid));

    // output conversion: SEM20 -> Q8.8  [3 cycles]
    sem20_to_q8p8 u_conv_out (.clk(clk), .rst_n(rst_n), .in_valid(final_valid), .sem_in(final_out),
                               .q88_out(out_q8p8), .out_valid(out_valid));
endmodule


// =============================================================================
// tt_um_wnn - Top-level Tiny Tapeout module
// =============================================================================
module tt_um_wnn (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 1=output, 0=input)
    input  wire       ena,      // always 1 when design is powered
    input  wire       clk,      // clock
    input  wire       rst_n     // reset, active low
);

    // Unused bidirectional bus
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // Internal signal mapping from TT ports
    wire       cfg_serial = ui_in[0];
    wire       cfg_valid  = ui_in[1];
    wire       cfg_load   = ui_in[2];
    wire [1:0] cfg_param  = ui_in[4:3];
    wire [5:0] cfg_neuron = 6'd0;     // single-neuron: address always 0
    wire       x_serial   = ui_in[5];
    wire       x_valid    = ui_in[6];

    // -------------------------------------------------------------------------
    // Configuration Register File
    // -------------------------------------------------------------------------
    localparam integer NUM_NEURONS = 1;

    reg [15:0] w_reg [0:NUM_NEURONS-1];
    reg [15:0] t_reg [0:NUM_NEURONS-1];
    reg [15:0] d_reg [0:NUM_NEURONS-1];

    reg [15:0] cfg_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_shift <= 16'd0;
        end else if (cfg_valid) begin
            cfg_shift <= {cfg_serial, cfg_shift[15:1]};  // LSB first
        end
    end

    always @(posedge clk) begin
        if (cfg_load) begin
            case (cfg_param)
                2'b00: w_reg[cfg_neuron] <= cfg_shift;
                2'b01: t_reg[cfg_neuron] <= cfg_shift;
                2'b10: d_reg[cfg_neuron] <= cfg_shift;
                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Input Deserializer
    // -------------------------------------------------------------------------
    reg [15:0] x_shift;
    reg [3:0]  x_bit_cnt;
    wire       x_deser_done = (x_bit_cnt == 4'd15) && x_valid;
    wire       x_ready      = (x_bit_cnt == 4'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_shift   <= 16'd0;
            x_bit_cnt <= 4'd0;
        end else if (x_valid) begin
            x_shift   <= {x_serial, x_shift[15:1]};
            x_bit_cnt <= (x_bit_cnt == 4'd15) ? 4'd0 : (x_bit_cnt + 1'b1);
        end
    end

    reg [15:0] x_q8p8_latched;
    reg        x_latch_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_q8p8_latched <= 16'd0;
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
    wire [15:0] neuron_out;
    wire        neuron_out_valid;

    C2S0285_neuron_top u_neuron (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (x_latch_valid),
        .x_q8p8    (x_q8p8_latched),
        .t_q8p8    (t_reg[0]),
        .d_q8p8    (d_reg[0]),
        .w_q8p8    (w_reg[0]),
        .out_q8p8  (neuron_out),
        .out_valid (neuron_out_valid)
    );

    // -------------------------------------------------------------------------
    // Single-Neuron Output Capture
    // -------------------------------------------------------------------------
    reg [15:0] tree_sum;
    reg        tree_valid;

    always @(posedge clk) begin
        tree_sum   <= neuron_out;
        tree_valid <= neuron_out_valid;
    end

    wire signed [15:0] sum_sat = $signed(tree_sum);

    // -------------------------------------------------------------------------
    // Output Serializer (16 cycles)
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
            sum_shift    <= sum_sat;
            sum_bit_cnt  <= 4'd0;
            shift_active <= 1'b1;
        end else if (shift_active) begin
            sum_shift    <= {1'b0, sum_shift[15:1]};
            sum_bit_cnt  <= sum_bit_cnt + 1'b1;
            if (sum_bit_cnt == 4'd15) begin
                shift_active <= 1'b0;
            end
        end
    end

    wire sum_serial = sum_shift[0];
    wire sum_valid  = shift_active;

    // -------------------------------------------------------------------------
    // Pipeline Busy Tracking
    // -------------------------------------------------------------------------
    localparam integer PIPELINE_DEPTH = 101;
    reg [6:0] busy_counter;
    wire      pipeline_busy = (busy_counter != 7'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_counter <= 7'd0;
        end else if (x_latch_valid) begin
            busy_counter <= PIPELINE_DEPTH[6:0];
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

    assign uo_out = {5'b00000, ready, sum_valid, sum_serial};

    // Suppress unused input warning
    wire _unused = &{ena, uio_in, 1'b0};

endmodule
