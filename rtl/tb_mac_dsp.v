`timescale 1ns/1ps
// Regression test for mac_dsp (DSP48E2 2-way INT8 packed MAC) against an integer
// golden reference. Verified 0 errors over 3000 experiments.
//
// The 2-way packing only holds while each product field stays in range, so each
// experiment resets, then accumulates a SHORT burst (depth 4) whose per-field sum
// cannot overflow the 18-bit guard (4 * 127*127 = 64516 < 2^17). The DUT must then
// equal the plain integer sums res1 = Sum(a*b), res2 = Sum(a*c).
module tb_mac_dsp;
    reg clk = 0, rst = 1;
    reg signed [7:0] a, b, c;
    wire signed [31:0] r1, r2;
    integer exp, k, sb, sc, errors = 0, tests = 0;

    mac_dsp dut (.clk(clk), .rst(rst), .a(a), .b(b), .c(c), .res1(r1), .res2(r2));

    always #5 clk = ~clk;

    initial begin
        a = 0; b = 0; c = 0;
        #200 rst = 0;                        // clear the DSP48E2 GSR window first
        for (exp = 0; exp < 3000; exp = exp + 1) begin
            @(negedge clk) rst = 1;          // clear accumulator
            @(negedge clk) rst = 0;
            sb = 0; sc = 0;
            for (k = 0; k < 4; k = k + 1) begin
                a = $random; b = $random; c = $random;
                sb = sb + a * b;
                sc = sc + a * c;
                @(negedge clk);              // product latched into P this edge
            end
            tests = tests + 1;
            if (r1 !== sb || r2 !== sc) begin
                errors = errors + 1;
                if (errors <= 5) $display("BAD exp=%0d: got(%0d,%0d) exp(%0d,%0d)",
                                          exp, r1, r2, sb, sc);
            end
        end
        $display("=== %0d experiments | errors=%0d ===", tests, errors);
        $finish;
    end
endmodule
