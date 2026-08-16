`timescale 1ns/1ps
// Testbench for pmpu (INT32 matmul + bias).
//
// Models the external operand memories (act / weight / bias) as 1-cycle registered
// reads (RD_LAT=1), filled to match pmpu's address formulas:
//   weight : addr = baseW + k*NT + tc   word = 16 col @ k  (pack2: 32 INT4 / single: 16 INT8)
//   act    : addr = baseA + k*MT + tr   word = 16 token @ k (INT8)
//   bias   : addr = baseB + tc*16 + col word = {bias_b, bias_a} column pair (INT32 x2)
// Drives one matmul command, captures the 16-lane INT32 output stream (o_valid/acc/
// o_tr/o_feat) into cap[token][feature], compares against a golden GEMM.
module tb_pmpu;

    logic clk = 0, rst;
    // command
    logic        start, pack2, bias_en;
    logic [9:0]  M, N, K;
    logic [2:0]  act_sel, w_sel;
    logic [12:0] baseA, baseW, baseB;
    logic        busy, done;
    // operand read
    logic [12:0] act_addr, w_addr, bias_addr;
    logic [2:0]  act_bank, w_bank;
    logic [127:0] act_rdata, w_rdata;
    logic [63:0]  bias_rdata;
    // result
    logic        o_valid;
    logic signed [31:0] acc [16];
    logic [3:0]  o_tr;
    logic [8:0]  o_feat;

    pmpu dut (.*);

    always #5 clk = ~clk;

    // ---- operand memories: 1-cycle registered read (RD_LAT=1) ----
    logic [127:0] act_mem  [0:4095];
    logic [127:0] w_mem    [0:4095];
    logic [63:0]  bias_mem [0:4095];
    always_ff @(posedge clk) begin
        act_rdata  <= act_mem [act_addr];
        w_rdata    <= w_mem   [w_addr];
        bias_rdata <= bias_mem[bias_addr];
    end

    // ---- golden / capture ----
    int  A      [64][512];     // activations [token][k]
    int  Wt     [512][512];    // weights [feature][k]
    int  BA     [512];         // bias [feature]
    int  golden [64][512];
    int  cap    [64][512];
    logic cap_en;
    int  errors = 0;

    // capture output stream: 16 tokens of tile o_tr at feature o_feat
    always_ff @(posedge clk)
        if (cap_en && o_valid)
            for (int r = 0; r < 16; r++) cap[o_tr*16 + r][o_feat] = acc[r];

    task automatic do_reset;
        rst = 1; start = 0; cap_en = 0;
        M=0; N=0; K=0; pack2=0; bias_en=0; act_sel=0; w_sel=0;
        baseA=0; baseW=0; baseB=0;
        @(posedge clk); @(posedge clk); #1; rst = 0; @(posedge clk); #1;
    endtask

    task automatic run(int m, int n, int kk, logic p2, logic ben, string name);
        int MT, NT, e_before, timeout;
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
            cap[t][f] = 32'sh8000_0000;                 // sentinel (detect no-write)
        end

        // --- fill act memory: word(k,tr) = 16 tokens @ k ---
        for (int tr=0;tr<MT;tr++)
            for (int k=0;k<kk;k++)
                for (int r=0;r<16;r++)
                    act_mem[k*MT + tr][r*8 +: 8] = A[tr*16 + r][k][7:0];

        // --- fill weight memory: word(k,tc) = 16 col @ k ---
        for (int tc=0;tc<NT;tc++)
            for (int k=0;k<kk;k++)
                for (int col=0;col<16;col++) begin
                    if (p2) w_mem[k*NT + tc][col*8 +: 8] =
                        {Wt[tc*32 + 2*col + 1][k][3:0], Wt[tc*32 + 2*col][k][3:0]};
                    else    w_mem[k*NT + tc][col*8 +: 8] = Wt[tc*16 + col][k][7:0];
                end

        // --- fill bias memory: word(tc,col) = {bias_b, bias_a} ---
        if (ben)
            for (int tc=0;tc<NT;tc++)
                for (int col=0;col<16;col++) begin
                    bias_mem[tc*16 + col][31:0]  = BA[tc*32 + 2*col];
                    bias_mem[tc*16 + col][63:32] = BA[tc*32 + 2*col + 1];
                end

        // --- drive command ---
        M=m; N=n; K=kk; pack2=p2; bias_en=ben; baseA=0; baseW=0; baseB=0;
        act_sel=0; w_sel=0;
        @(posedge clk); #1; cap_en = 1; start = 1;
        @(posedge clk); #1; start = 0;

        // --- wait done (with timeout) ---
        timeout = 0;
        while (!done && timeout < 100000) begin @(posedge clk); #1; timeout++; end
        @(posedge clk); #1; cap_en = 0;
        if (timeout >= 100000) begin $display("  [TIMEOUT] %s", name); errors++; end

        // --- compare ---
        for (int t=0;t<m;t++)
            for (int f=0;f<n;f++)
                if (cap[t][f] !== golden[t][f]) begin
                    errors++;
                    if (errors - e_before <= 8)
                        $display("  [MISS] t%0d f%0d: got %0d exp %0d", t, f, cap[t][f], golden[t][f]);
                end

        $display("[%s] %-20s (M=%0d N=%0d K=%0d %s%s)  mismatches=%0d",
                 (errors==e_before) ? "PASS" : "FAIL", name, m, n, kk,
                 p2?"pack2":"single", ben?"+bias":"", errors - e_before);
    endtask

    initial begin
        //$dumpfile("tb_pmpu.vcd");
        //$dumpvars(0, tb_pmpu);

        #100;                                   // glbl GSR
        do_reset; run(64, 128, 128, 1'b1, 1'b1, "QKVO proj");
        do_reset; run(64, 512, 128, 1'b1, 1'b1, "FFN1");
        do_reset; run(64, 128, 512, 1'b1, 1'b1, "FFN2");
        do_reset; run(64,  64,  64, 1'b0, 1'b0, "scores/context");

        if (errors == 0) $display("\n== ALL PASS ==");
        else             $display("\n== %0d FAIL ==", errors);
        $finish;
    end

endmodule
