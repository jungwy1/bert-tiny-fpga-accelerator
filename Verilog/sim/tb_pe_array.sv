`timescale 1ns/1ps
// Testbench for pe_array_16x16.
//
// Drives a K-deep GEMM tile: C[r][c] = sum_k A[r][k]*W[c][k] (+ bias_col[c]), where
// A[r][k]=act_row[r]@k and W[c][k]=w_col[c]@k. After K accumulate cycles it asserts
// drain for 16 cycles; the cascade shifts P left->right, so p_drain[] yields the
// rightmost column first: column 15, 14, ..., 0 (captured with reverse mapping).
// Each captured raw P is unpacked (borrow-corrected) and checked against the golden.
//   pack2=1 : acc0 = bias_a + sum wa*A,  acc1 = bias_b + sum wb*A
//   pack2=0 : acc0 = bias   + sum w*A,   acc1 = 0
module tb_pe_array;

    logic               clk = 0, rst, init, en, drain, pack2;
    logic signed [7:0]  act_row  [16];
    logic signed [7:0]  w_col    [16];
    logic signed [47:0] bias_col [16];
    logic signed [47:0] p_drain  [16];

    int errors = 0;

    pe_array_16x16 dut (.clk, .rst, .init, .en, .drain, .pack2,
                        .act_row, .w_col, .bias_col, .p_drain);

    always #5 clk = ~clk;

    // per-tile data / golden
    localparam int KMAX = 16;
    int A  [16][KMAX];           // activations (row, k)
    int WA [16][KMAX], WB [16][KMAX];  // weights (col, k): pack2 -> two INT4; single -> WA=INT8
    int BA [16], BB [16];        // per-column bias
    int g0 [16][16], g1 [16][16];      // golden acc0/acc1
    logic signed [47:0] cap [16][16];  // captured raw P per (row, col)

    // advance one cycle with given control; operands set by caller beforehand
    task automatic step(logic it, logic e, logic dr);
        init = it; en = e; drain = dr;
        @(posedge clk); #1;
    endtask

    task automatic do_reset;
        rst = 1; init = 0; en = 0; drain = 0; pack2 = 0;
        for (int i = 0; i < 16; i++) begin act_row[i]=0; w_col[i]=0; bias_col[i]=0; end
        @(posedge clk); @(posedge clk); #1;
        rst = 0; @(posedge clk); #1;
    endtask

    task automatic run_tile(int K, logic p2, string name);
        int e_before = errors;

        // --- random data + golden ---
        for (int c = 0; c < 16; c++) begin
            BA[c] = $urandom % 40; BB[c] = $urandom % 40;
            for (int k = 0; k < K; k++) begin
                WA[c][k] = p2 ? ($random % 8) : ($random % 128);
                WB[c][k] = p2 ? ($random % 8) : 0;
            end
        end
        for (int r = 0; r < 16; r++)
            for (int k = 0; k < K; k++) A[r][k] = $random % 128;
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++) begin
                g0[r][c] = BA[c];
                g1[r][c] = p2 ? BB[c] : 0;
                for (int k = 0; k < K; k++) begin
                    g0[r][c] += WA[c][k] * A[r][k];
                    if (p2) g1[r][c] += WB[c][k] * A[r][k];
                end
            end

        // --- drive accumulation ---
        pack2 = p2;
        for (int c = 0; c < 16; c++)
            bias_col[c] = p2 ? ((48'(BB[c]) <<< 20) + 48'(BA[c])) : 48'(BA[c]);
        for (int k = 0; k < K; k++) begin
            for (int r = 0; r < 16; r++) act_row[r] = A[r][k];
            for (int c = 0; c < 16; c++)
                w_col[c] = p2 ? {WB[c][k][3:0], WA[c][k][3:0]} : WA[c][k][7:0];
            step(k == 0, 1'b1, 1'b0);        // init on first, en=1, no drain
        end

        // --- flush (land last product) ---
        for (int r = 0; r < 16; r++) act_row[r] = 0;
        for (int c = 0; c < 16; c++) w_col[c] = 0;
        step(0, 1, 0); step(0, 1, 0);

        // --- drain 16 cycles: p_drain = col 15,14,...,0 ---
        for (int d = 0; d < 16; d++) begin
            step(0, 1, 1);
            for (int r = 0; r < 16; r++) cap[r][15 - d] = p_drain[r];
        end

        // --- unpack + compare ---
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++) begin
                int a0, a1;
                if (p2) begin
                    a0 = $signed({{12{cap[r][c][19]}}, cap[r][c][19:0]});
                    a1 = $signed({{4{cap[r][c][47]}}, cap[r][c][47:20]}) + cap[r][c][19];
                end else begin
                    a0 = $signed(cap[r][c][31:0]); a1 = 0;
                end
                if (a0 !== g0[r][c] || a1 !== g1[r][c]) begin
                    errors++;
                    if (errors - e_before <= 6)
                        $display("  [MISS] r%0d c%0d: a0=%0d(exp %0d) a1=%0d(exp %0d)",
                                 r, c, a0, g0[r][c], a1, g1[r][c]);
                end
            end

        $display("[%s] %-22s (K=%0d)  mismatches=%0d",
                 (errors == e_before) ? "PASS" : "FAIL", name, K, errors - e_before);
    endtask

    initial begin
        #100;                              // glbl GSR
        do_reset; run_tile(16, 1'b1, "pack2 tile");
        do_reset; run_tile(16, 1'b0, "single tile");
        do_reset; run_tile(8,  1'b1, "pack2 K=8");

        if (errors == 0) $display("\n== ALL PASS ==");
        else             $display("\n== %0d FAIL ==", errors);
        $finish;
    end

endmodule
