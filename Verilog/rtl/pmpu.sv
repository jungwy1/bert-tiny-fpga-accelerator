`timescale 1ns/1ps
// Packed Matrix Processing Unit — INT32 matmul (+bias) only (requant/residual/act = VFU).
//   pe_array_16x16 + inline unpack + bias fabric-pack (C port) + tiling FSM + addr-gen.
//   One command = one matmul. Tiles swept tr-outer/tc-inner (VFU needs all col-tiles of a
//   token for LN/softmax), k streamed. Output: 16-lane INT32 stream to VFU, feature-major.
//   Operand memories external; pmpu drives addr+bank-sel, reads back RD_LAT cycles later.
module pmpu #(
    parameter int RD_LAT = 1   // operand memory read latency (registered output)
)(
    input  logic clk, rst,

    // -- command (top controller) --
    input  logic         start,
    input  logic [9:0]   M, N, K,        // output token / feature / hidden dim
    input  logic         pack2,          // 1=INT4 dual(proj/FFN), 0=INT8 single(scores/context)
    input  logic         bias_en,        // proj/FFN=1, scores/context=0
    input  logic [2:0]   act_sel, weight_sel, // operand sources bank ID
    input  logic [12:0]  baseA, baseW, baseB,
    output logic         busy, done,

    // -- operand read (-> memory subsystem) --
    output logic [12:0]  act_addr,   output logic [2:0] act_bank,
    input  logic [127:0] act_rdata,
    output logic [12:0]  weight_addr,     output logic [2:0] weight_bank,
    input  logic [127:0] weight_rdata,
    output logic [12:0]  bias_addr,
    input  logic [63:0]  bias_rdata,        // {bias_b, bias_a}

    // -- fifo state
    input  logic         fifo_full,

    // -- result out (-> VFU, 16-lane) --
    output logic         o_valid,
    output logic signed [31:0] acc [16],  // 16 token @ feature o_feat
    output logic [3:0]   o_tr,
    output logic [8:0]   o_feat
);

    // ----------------------------------- command latch 
    logic [9:0]  M_r, N_r, K_r;
    logic        pack2_r, bias_en_r;
    logic [12:0] baseA_r, baseW_r, baseB_r;

    // tile counts (assume dims divisible: M/16, N/32(pack2) or N/16)
    wire [3:0] MT = M_r[9:4];                          // M/16
    wire [6:0] NT = pack2_r ? {2'b0, N_r[9:5]} : {1'b0, N_r[9:4]}; // N/32 : N/16
    wire [5:0] BEATS = pack2_r ? 6'd32 : 6'd16;

    // ----------------------------------- FSM 
    typedef enum logic [2:0] {S_IDLE, S_BIAS, S_COMPUTE, S_FLUSH, S_DRAIN, S_NEXT, S_DONE} state_t;
    state_t state;

    logic [6:0] tc;          // col-tile
    logic [3:0] tr;          // row-tile
    logic [9:0] k_cnt;       // contraction counter (0..K)
    logic [4:0] bi;          // bias preload counter (0..16+RD_LAT)
    logic       flush_cnt;   // flush: P-settle wait (2 cyc)
    logic [5:0] beat;        // drain beat (0=warmup, 1..BEATS emit)

    assign busy = (state != S_IDLE);
    assign act_bank = act_sel;
    assign weight_bank   = weight_sel;

    // --------------------------------- address generation 
    // weight_addr/act_addr: registered accumulators (init per tile, += NT/MT per k) — no multiply.
    // both feature-major (k-major). bias combinational (shift-add).
    assign bias_addr = baseB_r + {tc, 4'b0} + bi;   // baseB + tc*16 + bi

    // --------------------------------- operand feed pipeline (RD_LAT=1) 
    wire issuing = (state == S_COMPUTE) && (k_cnt < K_r);
    logic feed_v, feed_first;                       // aligned with returned rdata
    always_ff @(posedge clk) begin
        if (rst) begin feed_v <= 1'b0; feed_first <= 1'b0; end
        else begin
            feed_v     <= issuing;
            feed_first <= issuing && (k_cnt == 0);
        end
    end

    // --------------------------------- array 
    logic signed [7:0]  act_row [16], weight_col [16];
    logic signed [47:0] bias_col [16];
    logic signed [47:0] p_drain [16];
    logic arr_init, arr_en, arr_drain;

    // drain shift-enable, paced so p_drain aligns with cur_col (shift lands 2 cyc after en).
    // pack2: shift every 2nd beat (2 feat/col);  single: every beat.
    wire drain_shift = pack2_r ? beat[0] : 1'b1;
    assign arr_init  = feed_first;
    // FLUSH needs no en: en_latch (CEP = en delayed 1) + MREG=0 land the last product.
    assign arr_en    = feed_v | ((state == S_DRAIN) & drain_shift);
    assign arr_drain = (state == S_DRAIN);

    always_comb begin
        for (int r = 0; r < 16; r++)
            act_row[r] = feed_v ? $signed(act_rdata[r*8 +: 8]) : 8'sd0;
        for (int c = 0; c < 16; c++)
            weight_col[c] = feed_v ? $signed(weight_rdata[c*8 +: 8]) : 8'sd0;
    end

    pe_array_16x16 u_arr (
        .clk, .rst, .init(arr_init), .en(arr_en), .drain(arr_drain), .pack2(pack2_r),
        .hold(fifo_full), .act_row, .w_col(weight_col), .bias_col, .p_drain
    );

    // --------------------------------- bias packing (preload → C port) 
    wire signed [47:0] b_packed =
          (48'($signed(bias_rdata[63:32])) <<< 20) + 48'($signed(bias_rdata[31:0]));

    // --------------------------------- unpack (raw P → acc0/acc1) 
    logic signed [31:0] acc0 [16], acc1 [16];
    always_comb begin
        for (int r = 0; r < 16; r++) begin
            if (pack2_r) begin
                acc0[r] = $signed({{12{p_drain[r][19]}}, p_drain[r][19:0]});
                acc1[r] = $signed({{4{p_drain[r][47]}}, p_drain[r][47:20]}) + p_drain[r][19];
            end else begin
                acc0[r] = p_drain[r][31:0];
                acc1[r] = 32'sd0;
            end
        end
    end

    // drain geometry (beat>=1; beat0=warmup): col 15->0. pack2 = 2 beats/col, single = 1.
    wire [3:0] cur_col   = pack2_r ? 4'd15 - ((beat - 1) >> 1)
                                   : 4'd15 - (beat - 1);
    wire       cur_phase = pack2_r & ~beat[0];       // pack2: 0=acc0(2c), 1=acc1(2c+1)

    // -------------------------------------- main sequence
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0; o_valid <= 1'b0;
            tc <= 0; tr <= 0; k_cnt <= 0; bi <= 0; flush_cnt <= 0; beat <= 0;
        end else begin
            done <= 1'b0; 
            if (~fifo_full) o_valid <= 1'b0;

            unique case (state)
            S_IDLE: if (start) begin
                M_r <= M; N_r <= N; K_r <= K; pack2_r <= pack2; bias_en_r <= bias_en;
                baseA_r <= baseA; baseW_r <= baseW; baseB_r <= baseB;
                tc <= 0; tr <= 0;
                bi <= 0; k_cnt <= 0;
                state <= bias_en ? S_BIAS : S_COMPUTE;
                if (!bias_en) begin
                    for (int i = 0; i < 16; i++) bias_col[i] <= 48'sd0;
                    weight_addr <= baseW; act_addr <=baseA;
                end
            end

            // preload tc's 16 bias words -> bias_col (fabric pack), then init addr accumulators
            S_BIAS: begin
                bi <= bi + 1;
                if (bi >= RD_LAT && bi <= RD_LAT + 15)   // data valid, index 0..15
                    bias_col[bi - RD_LAT] <= b_packed;
                if (bi == RD_LAT + 15) begin
                    weight_addr <= baseW_r + tc; act_addr <=baseA_r + tr;
                    bi <= 0; k_cnt <= 0; state <= S_COMPUTE;
                end
            end

            // stream k: feed operands, accumulate addr (+=NT/MT)
            S_COMPUTE: begin
                weight_addr <= weight_addr + NT;
                act_addr <= act_addr + MT;
                if (k_cnt < K_r) k_cnt <= k_cnt + 1;
                else begin flush_cnt <= 0; state <= S_FLUSH; end
            end

            // wait for last product to settle in P before drain
            S_FLUSH: begin
                flush_cnt <= ~flush_cnt;
                if (flush_cnt == 1'b1) begin beat <= 0; state <= S_DRAIN; end
            end

            // cascade drain -> 16-lane stream (beat0 warmup; pack2 2 beats/col, single 1/col)
            S_DRAIN: begin
                if (~fifo_full) begin
                    o_valid <= (beat != 0);
                    acc     <= cur_phase ? acc1 : acc0;
                    o_tr    <= tr;
                    o_feat  <= pack2_r ? ({tc, 5'b0} + {cur_col, 1'b0} + cur_phase)  // tc*32 + 2*col + phase
                                   : ({tc, 4'b0} + cur_col);                     // tc*16 + col
                    if (beat == BEATS) state <= S_NEXT;
                    else               beat <= beat + 1;
                end
            end

            // advance tiles (tc inner, tr outer); bias reloads per tile; re-init addr accum
            S_NEXT: begin
                if (tc + 1 < NT) begin
                    tc <= tc + 1; bi <= 0; k_cnt <=0;
                    weight_addr <= baseW_r + tc + 1; act_addr <=baseA_r + tr;
                    state <= bias_en_r ? S_BIAS : S_COMPUTE;
                end else begin
                    tc <= 0;
                    if (tr + 1 < MT) begin
                        tr <= tr + 1; bi <= 0; k_cnt <=0;
                        weight_addr <= baseW_r; act_addr <=baseA_r + tr + 1;
                        state <= bias_en_r ? S_BIAS : S_COMPUTE;
                    end else begin
                        state <= S_DONE;
                    end
                end
            end

            S_DONE: begin done <= 1'b1; state <= S_IDLE; end
            endcase
        end
    end

endmodule