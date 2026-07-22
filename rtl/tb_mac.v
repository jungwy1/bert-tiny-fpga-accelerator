`timescale 1ns/1ps
// Regression test for mac (DSP = multiplier only, accumulation in fabric).
// Fabric accumulators are independent per product, so accumulation is DEEP:
// here depth 256 (far past mac_dsp's ~8 limit) must still match integer golden.
module tb_mac;
    reg clk = 0, rst = 1;
    reg signed [7:0] a, b, c;
    wire signed [31:0] r1, r2;
    integer exp, k, sb, sc, errors = 0, tests = 0;

    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .c(c), .res1(r1), .res2(r2));

    always #5 clk = ~clk;

    task step(input signed [7:0] ta, tb, tc);
        begin a = ta; b = tb; c = tc; @(negedge clk); end
    endtask

    initial begin
        a = 0; b = 0; c = 0;
        #40 rst = 0;
        for (exp = 0; exp < 2000; exp = exp + 1) begin
            @(negedge clk) rst = 1;          // clear fabric accumulators
            @(negedge clk) rst = 0;
            sb = 0; sc = 0;
            for (k = 0; k < 256; k = k + 1) begin   // DEEP accumulation
                a = $random; b = $random; c = $random;
                sb = sb + a * b;
                sc = sc + a * c;
                @(negedge clk);              // this cycle's product accumulates here
            end
            tests = tests + 1;
            if (r1 !== sb || r2 !== sc) begin
                errors = errors + 1;
                if (errors <= 5) $display("BAD exp=%0d depth256: got(%0d,%0d) exp(%0d,%0d)",
                                          exp, r1, r2, sb, sc);
            end
        end
        $display("=== %0d experiments (depth 256) | errors=%0d ===", tests, errors);
        $finish;
    end
endmodule
