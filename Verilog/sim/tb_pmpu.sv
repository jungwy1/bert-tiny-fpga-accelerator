`timescale 1ns/1ps
// Testbench for pmpu (INT32 matmul + bias) with FIFO backpressure.
//
// pmpu -> fifo (525b: acc[16]+o_tr+o_feat) -> "VFU" model (random reads). fifo.full
// drives pmpu.fifo_full: on full, pmpu global-freezes (array hold + beat/output hold),
// the held beat is written once a slot frees. Capture is on the FIFO READ side, so any
// loss or duplication from backpressure surfaces as a golden GEMM mismatch.
module tb_pmpu;

    logic clk = 0, rst;
    // command
    logic        start, pack2, bias_en;
    logic [9:0]  M, N, K;
    logic [2:0]  act_sel, weight_sel;
    logic [12:0] baseA, baseW, baseB;
    logic        busy, done;
    // operand read
    logic [12:0] act_addr, weight_addr, bias_addr;
    logic [2:0]  act_bank, weight_bank;
    logic [127:0] act_rdata, weight_rdata;
    logic [63:0]  bias_rdata;
    // fifo state + result
    logic        fifo_full;
    logic        o_valid;
    logic signed [31:0] acc [16];
    logic [3:0]  o_tr;
    logic [8:0]  o_feat;

    pmpu dut (.*);

    always #5 clk = ~clk;

    // ---- operand memories: 1-cycle registered read (RD_LAT=1) ----
    logic [127:0] act_mem  [0:1023];
    logic [127:0] weight_mem    [0:1023];
    logic [63:0]  bias_mem [0:1023];
    always_ff @(posedge clk) begin
        act_rdata  <= act_mem [act_addr];
        weight_rdata    <= weight_mem   [weight_addr];
        bias_rdata <= bias_mem[bias_addr];
    end

    // ---- output FIFO (525b) + random-rate "VFU" reader ----
    localparam int W = 525;                       // acc(512) + o_tr(4) + o_feat(9)
    logic [511:0] acc_flat;
    always_comb for (int r = 0; r < 16; r++) acc_flat[r*32 +: 32] = acc[r];
    wire  [W-1:0] wr_data = {acc_flat, o_tr, o_feat};
    logic [W-1:0] rd_data;
    logic         fifo_empty, vfu_rd;
    wire          fifo_pop = vfu_rd & ~fifo_empty;

    fifo #(.W(W), .DEPTH(8)) u_fifo (
        .clk, .rst, .wr_en(o_valid), .wr_data(wr_data),
        .full(fifo_full),
        .rd_en(fifo_pop), .rd_data(rd_data), .empty(fifo_empty)
    );

    // random VFU read rate (~50%) — slow enough to fill the FIFO during drain bursts
    always_ff @(posedge clk) vfu_rd <= ($urandom % 2);

    // ---- golden / capture (on FIFO read side) ----
    int  A      [32][16];
    int  Wt     [128][16];
    int  BA     [128];
    int  golden [32][128];
    int  cap    [32][128];
    logic cap_en;
    logic [8:0] cf; logic [3:0] ct;
    int  errors = 0;

    always_ff @(posedge clk)
        if (cap_en && fifo_pop) begin
            cf = rd_data[8:0]; ct = rd_data[12:9];
            for (int r = 0; r < 16; r++) cap[ct*16 + r][cf] = $signed(rd_data[13 + r*32 +: 32]);
        end

    task automatic do_reset;
        rst = 1; start = 0; cap_en = 0; vfu_rd = 0;
        M=0; N=0; K=0; pack2=0; bias_en=0; act_sel=0; weight_sel=0;
        baseA=0; baseW=0; baseB=0;
        @(posedge clk); @(posedge clk); #1; rst = 0; @(posedge clk); #1;
    endtask

    task automatic run(int m, int n, int kk, logic p2, logic ben, string name);
        int MT, NT, e_before, timeout;
        logic seen_done;
        e_before = errors;
        MT = m/16;
        NT = p2 ? n/32 : n/16;

        // --- random data + golden ---
        for (int t=0;t<m;t++) for (int k=0;k<kk;k++) A[t][k]  = $random % 128;
        for (int f=0;f<n;f++) for (int k=0;k<kk;k++) Wt[f][k] = p2 ? ($random % 8) : ($random % 128);
        for (int f=0;f<n;f++) BA[f] = ben ? ($urandom % 40) : 0;
        for (int t=0;t<m;t++) for (int f=0;f<n;f++) begin
            golden[t][f] = BA[f];
            for (int k=0;k<kk;k++) golden[t][f] += A[t][k]*Wt[f][k];
            cap[t][f] = 32'sh8000_0000;                 // sentinel
        end

        // --- fill operand memories (pmpu address formulas) ---
        for (int tr=0;tr<MT;tr++) for (int k=0;k<kk;k++) for (int r=0;r<16;r++)
            act_mem[k*MT + tr][r*8 +: 8] = A[tr*16 + r][k][7:0];
        for (int tc=0;tc<NT;tc++) for (int k=0;k<kk;k++) for (int col=0;col<16;col++)
            if (p2) weight_mem[k*NT + tc][col*8 +: 8] = {Wt[tc*32+2*col+1][k][3:0], Wt[tc*32+2*col][k][3:0]};
            else    weight_mem[k*NT + tc][col*8 +: 8] = Wt[tc*16 + col][k][7:0];
        if (ben) for (int tc=0;tc<NT;tc++) for (int col=0;col<16;col++) begin
            bias_mem[tc*16 + col][31:0]  = BA[tc*32 + 2*col];
            bias_mem[tc*16 + col][63:32] = BA[tc*32 + 2*col + 1];
        end

        // --- drive command ---
        M=m; N=n; K=kk; pack2=p2; bias_en=ben; baseA=0; baseW=0; baseB=0;
        act_sel=0; weight_sel=0;
        @(posedge clk); #1; cap_en = 1; start = 1;
        @(posedge clk); #1; start = 0;

        // --- wait until done AND FIFO fully read out (random reads drain it) ---
        seen_done = 0; timeout = 0;
        while (!(seen_done && fifo_empty) && timeout < 200000) begin
            @(posedge clk); #1;
            if (done) seen_done = 1;
            timeout++;
        end
        @(posedge clk); #1; cap_en = 0;
        if (timeout >= 200000) begin $display("  [TIMEOUT] %s", name); errors++; end

        // --- compare ---
        for (int t=0;t<m;t++) for (int f=0;f<n;f++)
            if (cap[t][f] !== golden[t][f]) begin
                errors++;
                if (errors - e_before <= 8)
                    $display("  [MISS] t%0d f%0d: got %0d exp %0d", t, f, cap[t][f], golden[t][f]);
            end

        $display("[%s] %-18s (M=%0d N=%0d K=%0d %s%s)  mismatches=%0d",
                 (errors==e_before) ? "PASS" : "FAIL", name, m, n, kk,
                 p2?"pack2":"single", ben?"+bias":"", errors - e_before);
    endtask

    initial begin
        #100;                                   // glbl GSR
        do_reset; run(32,  64,  8, 1'b1, 1'b1, "pack2 2x2");
        do_reset; run(16, 128, 16, 1'b1, 1'b1, "pack2 1x4");
        do_reset; run(32,  32,  8, 1'b0, 1'b0, "single 2x2");

        if (errors == 0) $display("\n== ALL PASS ==");
        else             $display("\n== %0d FAIL ==", errors);
        $finish;
    end

endmodule
