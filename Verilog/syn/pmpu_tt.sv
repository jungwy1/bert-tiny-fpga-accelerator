`timescale 1ns/1ps
// ─────────────────────────────────────────────────────────────────────────
// Timing wrapper for pmpu (SYNTHESIS/OOC TIMING ONLY — not part of the design).
//
// Why: in out-of-context synth the pmpu outputs (acc, o_*) have no external
// load, so the tool dead-code-eliminates the whole datapath (256-DSP array,
// acc/bias_col registers) and reports a bogus 1-DSP timing.
//
// This wrapper registers every pmpu input and folds every pmpu output into a
// single registered observable (obs). Now the datapath sits between real FFs
// and is preserved → report_timing_summary gives true internal reg-to-reg Fmax
// including broadcast operand nets and the 256-fanout control signals.
// ─────────────────────────────────────────────────────────────────────────
module pmpu_tt (
    input  logic clk, rst,
    input  logic start_i, pack2_i, bias_en_i,
    input  logic [9:0]   M_i, N_i, K_i,
    input  logic [2:0]   act_sel_i, w_sel_i,
    input  logic [12:0]  baseA_i, baseW_i, baseB_i,
    input  logic [127:0] act_rdata_i, w_rdata_i,
    input  logic [63:0]  bias_rdata_i,
    output logic         obs
);
    // --- registered inputs (give the datapath real drivers) ---
    logic start, pack2, bias_en;
    logic [9:0]  M, N, K;
    logic [2:0]  act_sel, w_sel;
    logic [12:0] baseA, baseW, baseB;
    logic [127:0] act_rdata, w_rdata;
    logic [63:0]  bias_rdata;
    always_ff @(posedge clk) begin
        start<=start_i; pack2<=pack2_i; bias_en<=bias_en_i;
        M<=M_i; N<=N_i; K<=K_i; act_sel<=act_sel_i; w_sel<=w_sel_i;
        baseA<=baseA_i; baseW<=baseW_i; baseB<=baseB_i;
        act_rdata<=act_rdata_i; w_rdata<=w_rdata_i; bias_rdata<=bias_rdata_i;
    end

    // --- DUT ---
    logic busy, done, o_valid;
    logic [12:0] act_addr, w_addr, bias_addr;
    logic [2:0]  act_bank, w_bank;
    logic signed [31:0] acc [16];
    logic [3:0]  o_tr;
    logic [8:0]  o_feat;
    pmpu dut (
        .clk, .rst, .start, .M, .N, .K, .pack2, .bias_en, .act_sel, .w_sel,
        .baseA, .baseW, .baseB, .busy, .done,
        .act_addr, .act_bank, .act_rdata, .w_addr, .w_bank, .w_rdata,
        .bias_addr, .bias_rdata, .o_valid, .acc, .o_tr, .o_feat
    );

    // --- fold all outputs into one registered observable (keeps every output) ---
    logic [31:0] acc_x;
    always_comb begin
        acc_x = 32'b0;
        for (int i=0;i<16;i++) acc_x ^= acc[i];
    end
    always_ff @(posedge clk)
        obs <= ^{busy, done, o_valid, act_addr, w_addr, bias_addr,
                 act_bank, w_bank, acc_x, o_tr, o_feat};
endmodule
