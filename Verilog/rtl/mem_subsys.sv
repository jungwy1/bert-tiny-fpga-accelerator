`timescale 1ns/1ps
// On-chip memory subsystem: all banks + operand mux + address routing + write decode.
//   Presents pmpu's operand-read interface (act/weight/bias) and hides the physical banks:
//     5× mem_act (resid,b0..b3) · 3× mem_weight URAM (QKVO,FFN0,FFN1) · mem_bias · mem_ln_param
//   Datapath ports are dedicated (no arbiter, §7): each bank read by <=1 operand at a time.
//   PS(host) touches only the residual bank, and only while idle (busy-mux).
module mem_subsys (
    input  logic         clk,
    input  logic         busy,                          // 1: datapath owns resid; 0: PS may access
    // -- pmpu operand read --
    input  logic [2:0]   act_sel,    input logic [12:0] act_addr,    output logic [127:0] act_rdata,
    input  logic [2:0]   weight_sel, input logic [12:0] weight_addr, output logic [127:0] weight_rdata,
    input  logic [12:0]  bias_addr,  output logic [63:0] bias_rdata,
    // -- VFU drain write (result -> act banks) --
    input  logic [2:0]   dest_sel,   input logic we, input logic [12:0] waddr, input logic [127:0] wdata,
    // -- VFU ln_param read --
    input  logic         ln_re,      input logic [8:0] ln_addr,      output logic [71:0] ln_rdata,
    // -- PS(host) load/readback (residual only) --
    input  logic         ps_we, ps_re, input logic [8:0] ps_addr,
    input  logic [127:0] ps_wdata,   output logic [127:0] ps_rdata
);

    localparam logic [2:0] A_RESID=3'd0, A_B0=3'd1, A_B1=3'd2, A_B2=3'd3, A_B3=3'd4, A_H=3'd5;
    localparam logic [2:0] W_QKVO=3'd0, W_FFN0=3'd1, W_FFN1=3'd2, W_B1=3'd3, W_B2=3'd4;

    // per-bank read data
    logic [127:0] rd_resid, rd_b0, rd_b1, rd_b2, rd_b3;
    logic [127:0] rd_qkvo, rd_ffn0, rd_ffn1;

    // ---- read address routing (each bank read by <=1 operand at a time) ----
    wire [8:0] a_act = act_addr[8:0];                    // H-mode: local = addr[8:0], bank = addr[10:9]
    wire [8:0] a_wgt = weight_addr[8:0];
    wire [1:0] a_hbank = act_addr[10:9];                 // H sub-bank select (FFN2 read)
    wire [8:0] resid_ra = busy ? a_act : ps_addr;        // PS reads resid only while idle
    // b0/b3: act only (direct or H) -> always act_addr.  b1/b2: weight (K/V) unless H picks them.
    wire [8:0] b1_ra = (act_sel == A_H && a_hbank == 2'd1) ? a_act : a_wgt;
    wire [8:0] b2_ra = (act_sel == A_H && a_hbank == 2'd2) ? a_act : a_wgt;

    // ---- write decode (VFU -> act banks; PS -> resid while idle) ----
    //   H mode (dest_sel=5): FFN1 writes H across b0..b3, bank = waddr[10:9], local = waddr[8:0].
    wire [8:0] vfu_wa = waddr[8:0];
    wire [1:0] w_hbank = waddr[10:9];
    wire         resid_we = busy ? (we & (dest_sel == A_RESID)) : ps_we;
    wire [8:0]   resid_wa = busy ? vfu_wa : ps_addr;
    wire [127:0] resid_wd = busy ? wdata  : ps_wdata;
    wire b0_we = we & ((dest_sel == A_B0) | (dest_sel == A_H && w_hbank == 2'd0));
    wire b1_we = we & ((dest_sel == A_B1) | (dest_sel == A_H && w_hbank == 2'd1));
    wire b2_we = we & ((dest_sel == A_B2) | (dest_sel == A_H && w_hbank == 2'd2));
    wire b3_we = we & ((dest_sel == A_B3) | (dest_sel == A_H && w_hbank == 2'd3));

    // ---- activation banks (512 x 128) ----
    mem_act u_resid (.clk, .we(resid_we), .waddr(resid_wa), .wdata(resid_wd),
                     .re(1'b1), .raddr(resid_ra), .rdata(rd_resid));
    mem_act u_b0    (.clk, .we(b0_we),    .waddr(vfu_wa),   .wdata(wdata),
                     .re(1'b1), .raddr(a_act),    .rdata(rd_b0));
    mem_act u_b1    (.clk, .we(b1_we),    .waddr(vfu_wa),   .wdata(wdata),
                     .re(1'b1), .raddr(b1_ra),    .rdata(rd_b1));
    mem_act u_b2    (.clk, .we(b2_we),    .waddr(vfu_wa),   .wdata(wdata),
                     .re(1'b1), .raddr(b2_ra),    .rdata(rd_b2));
    mem_act u_b3    (.clk, .we(b3_we),    .waddr(vfu_wa),   .wdata(wdata),
                     .re(1'b1), .raddr(a_act),    .rdata(rd_b3));

    // ---- weight URAM banks (4096 x 128, baked ROM: write port tied off) ----
    mem_weight #(.INIT_FILE("mem/weight_qkvo.txt"))   u_wq  (.clk, .we(1'b0), .waddr('0), .wdata('0),
                     .re(1'b1), .raddr(weight_addr[11:0]), .rdata(rd_qkvo));
    mem_weight #(.INIT_FILE("mem/weight_ffn_l0.txt")) u_wf0 (.clk, .we(1'b0), .waddr('0), .wdata('0),
                     .re(1'b1), .raddr(weight_addr[11:0]), .rdata(rd_ffn0));
    mem_weight #(.INIT_FILE("mem/weight_ffn_l1.txt")) u_wf1 (.clk, .we(1'b0), .waddr('0), .wdata('0),
                     .re(1'b1), .raddr(weight_addr[11:0]), .rdata(rd_ffn1));

    // ---- bias BRAM (1536 x 64, baked) ----
    mem_bias #(.INIT_FILE("mem/bias.txt")) u_bias (.clk, .we(1'b0), .waddr('0), .wdata('0),
                     .re(1'b1), .raddr(bias_addr[10:0]), .rdata(bias_rdata));

    // ---- ln_param BRAM (512 x 72, baked) ----
    mem_ln_param #(.INIT_FILE("mem/ln_param.txt")) u_ln (.clk, .we(1'b0), .waddr('0), .wdata('0),
                     .re(ln_re), .raddr(ln_addr), .rdata(ln_rdata));

    // ---- read-data mux (select delayed 1 cyc to align with RD_LAT=1 registered read) ----
    logic [2:0] asel_q, wsel_q;
    logic [1:0] hbank_q;                                 // delayed act_addr[10:9] for H mode
    always_ff @(posedge clk) begin
        asel_q  <= act_sel;
        wsel_q  <= weight_sel;
        hbank_q <= a_hbank;
    end

    logic [127:0] rd_hbank;                              // H sub-bank pick
    always_comb begin
        unique case (hbank_q)
            2'd0:    rd_hbank = rd_b0;
            2'd1:    rd_hbank = rd_b1;
            2'd2:    rd_hbank = rd_b2;
            default: rd_hbank = rd_b3;
        endcase
        unique case (asel_q)
            A_B0:    act_rdata = rd_b0;
            A_B1:    act_rdata = rd_b1;
            A_B2:    act_rdata = rd_b2;
            A_B3:    act_rdata = rd_b3;
            A_H:     act_rdata = rd_hbank;               // FFN2: H across b0..b3
            default: act_rdata = rd_resid;               // A_RESID
        endcase
        unique case (wsel_q)
            W_FFN0:  weight_rdata = rd_ffn0;
            W_FFN1:  weight_rdata = rd_ffn1;
            W_B1:    weight_rdata = rd_b1;
            W_B2:    weight_rdata = rd_b2;
            default: weight_rdata = rd_qkvo;             // W_QKVO
        endcase
    end

    assign ps_rdata = rd_resid;                          // PS reads residual only

endmodule
