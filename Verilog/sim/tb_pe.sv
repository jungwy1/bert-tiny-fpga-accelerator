`timescale 1ns/1ps
// Testbench for pe.sv (DSP48E2 MAC: pack2, in-DSP accumulate, C-port bias, cascade drain).

module tb_pe;

    logic               clk = 0, rst, drain, init, en, pack2, hold;
    logic signed [7:0]  act, w;
    logic signed [47:0] bias, pcin, pcout;

    // ===================================== unpacked signals for debug
    logic signed [3:0]  w_a, w_b;          
    logic signed [31:0] bias_a, bias_b;     
    logic signed [31:0] acc0, acc1;        

    assign w_a    = w[3:0];
    assign w_b    = w[7:4];
    assign bias_a = {{12{bias[19]}}, bias[19:0]};
    assign bias_b = {{4{bias[47]}}, bias[47:20]} + {31'b0, bias[19]};
    assign acc0   = pack2 ? {{12{pcout[19]}}, pcout[19:0]} : pcout[31:0];
    assign acc1   = pack2 ? ({{4{pcout[47]}}, pcout[47:20]} + {31'b0, pcout[19]}) : 32'sd0;
    // =================================================================

    int errors = 0;

    pe dut (.clk, .rst, .drain, .init, .en, .hold, .pack2, .act, .w, .bias, .pcin, .pcout);

    always #5 clk = ~clk;                       // 100 MHz

    task automatic drive(logic dr, logic it, logic e, logic p2,
                         logic signed [7:0] a, logic signed [7:0] wi,
                         logic signed [47:0] b, logic signed [47:0] pc);
        drain = dr; init = it; en = e; pack2 = p2;
        act = a; w = wi; bias = b; pcin = pc;
        @(posedge clk); #1;
    endtask

    task automatic flush(logic p2);
        drive(0, 0, 0, p2, 0, 0, 48'sd0, 48'sd0);
        drive(0, 0, 0, p2, 0, 0, 48'sd0, 48'sd0);
    endtask

    task automatic do_reset;
        rst = 1; drain = 0; init = 0; en = 0; pack2 = 0; hold = 0;
        act = 0; w = 0; bias = 0; pcin = 0;
        @(posedge clk); @(posedge clk); #1;
        rst = 0;
        @(posedge clk); #1;
    endtask

    // unpack pcout (gap-20 packed P) and compare
    task automatic check(int e0, int e1, logic p2, string name);
        int a0, a1;
        if (p2) begin
            a0 = $signed({{12{pcout[19]}}, pcout[19:0]});         // Sum(w_a*act)+bias_a
            a1 = $signed({{4{pcout[47]}}, pcout[47:20]}) + pcout[19];  // Sum(w_b*act)+bias_b
        end else begin
            a0 = $signed(pcout[31:0]);                           // Sum(w*act)+bias
            a1 = 0;
        end
        if (a0 === e0 && a1 === e1)
            $display("[PASS] %-26s a0=%0d a1=%0d", name, a0, a1);
        else begin
            $display("[FAIL] %-26s a0=%0d (exp %0d)  a1=%0d (exp %0d)", name, a0, e0, a1, e1);
            errors++;
        end
    endtask

    task automatic check_raw(logic signed [47:0] e, string name);
        if (pcout === e) $display("[PASS] %-26s pcout=%h", name, pcout);
        else begin $display("[FAIL] %-26s pcout=%h (exp %h)", name, pcout, e); errors++; end
    endtask

    // ------------------------------------------------------------- test tiles
    // pack2=1: K MACs of INT8 act x two INT4 weights, with packed C-port bias.
    task automatic run_pack2(int K, string name);
        logic signed [7:0] a; logic signed [3:0] wa, wb;
        int ba, bb, e0, e1; logic signed [47:0] bpk;
        ba = $urandom % 60; bb = $urandom % 60;          // small positive bias
        bpk = (48'(bb) <<< 20) + 48'(ba);                // packed bias -> C
        e0 = ba; e1 = bb;
        for (int i = 0; i < K; i++) begin
            a = $random % 128; wa = $random % 8; wb = $random % 8;
            drive(0, i == 0, 1, 1, a, {wb, wa}, bpk, 48'sd0);   // init on first
            e0 += wa * a; e1 += wb * a;
        end
        flush(1);
        check(e0, e1, 1'b1, name);
    endtask

    // pack2=0: K MACs of INT8 act x one INT8 weight, single C-port bias.
    task automatic run_single(int K, string name);
        logic signed [7:0] a, wi; int b, e0;
        b = $urandom % 60;
        e0 = b;
        for (int i = 0; i < K; i++) begin
            a = $random % 128; wi = $random % 128;
            drive(0, i == 0, 1, 0, a, wi, 48'(b), 48'sd0);
            e0 += wi * a;
        end
        flush(0);
        check(e0, 0, 1'b0, name);
    endtask

    // en=0 must skip exactly that cycle's product (pack2, no bias for clarity).
    task automatic test_hold;
        int e0, e1;
        e0 = 0; e1 = 0;
        drive(0, 1, 1, 1, 8'sd10, {4'sd2, 4'sd3}, 48'sd0, 48'sd0); e0+=3*10; e1+=2*10;
        drive(0, 0, 1, 1, 8'sd20, {4'sd1, 4'sd2}, 48'sd0, 48'sd0); e0+=2*20; e1+=1*20;
        drive(0, 0, 0, 1, 8'sd50, {4'sd7, 4'sd7}, 48'sd0, 48'sd0); // en=0 -> skip
        drive(0, 0, 1, 1, 8'sd5,  {4'sd1, 4'sd1}, 48'sd0, 48'sd0); e0+=1*5;  e1+=1*5;
        flush(1);
        check(e0, e1, 1'b1, "hold (en=0 skips)");
    endtask

    // drain: SHIFT mode loads pcin into P each cycle. Hold pcin=V and read while
    // still shifting (a stop cycle with pcin=0 would shift 0 in, since OPMODEREG
    // keeps SHIFT active one extra cycle).
    task automatic test_drain;
        logic signed [47:0] V;
        V = 48'h1234_5678_9ABC;
        repeat (6) drive(1, 0, 1, 0, 0, 0, 48'sd0, V);   // drain, en=1, pcin=V held
        check_raw(V, "drain (P <- pcin)");
    endtask

    // drain + random hold: freezing P mid-shift must be transparent — pcout with hold
    // cycles removed matches a clean (no-hold) run, and pcout stays frozen through each
    // hold (the in-flight shift resumes correctly on release).
    task automatic test_drain_hold;
        logic signed [47:0] seq [0:11];
        logic signed [47:0] clean [0:11];
        logic signed [47:0] frozen;
        int s, e_before;
        e_before = errors;
        for (int i = 0; i < 12; i++) seq[i] = {$random, $random};

        // clean reference: prime P->0, then shift seq, capturing pcout each cycle
        do_reset; hold = 0;
        repeat (4) drive(1, 0, 1, 0, 0, 0, 48'sd0, 48'sd0);
        for (s = 0; s < 12; s++) begin
            drive(1, 0, 1, 0, 0, 0, 48'sd0, seq[s]);
            clean[s] = pcout;
        end

        // held run: same prime + seq, with random hold cycles inserted
        do_reset; hold = 0;
        repeat (4) drive(1, 0, 1, 0, 0, 0, 48'sd0, 48'sd0);
        s = 0;
        while (s < 12) begin
            if (s > 0 && ($urandom % 2)) begin              // hold: freeze, garbage pcin
                frozen = pcout;
                hold = 1;
                drive(1, 0, 1, 0, 0, 0, 48'sd0, ~seq[s]);
                if (pcout !== frozen) begin
                    errors++;
                    $display("  [MISS] hold@%0d not frozen: %h != %h", s, pcout, frozen);
                end
            end else begin                                  // advance
                hold = 0;
                drive(1, 0, 1, 0, 0, 0, 48'sd0, seq[s]);
                if (pcout !== clean[s]) begin
                    errors++;
                    $display("  [MISS] step %0d: %h exp %h", s, pcout, clean[s]);
                end
                s++;
            end
        end
        hold = 0;
        $display("[%s] %-26s mismatches=%0d",
                 (errors == e_before) ? "PASS" : "FAIL", "drain + random hold", errors - e_before);
    endtask

    // ------------------------------------------------------------------- main
    initial begin
        $dumpfile("tb_pe.vcd");
        $dumpvars(0, tb_pe);

        #100;                                   // wait out glbl GSR
        do_reset;  run_pack2 (128, "pack2 K=128 (+bias)");
        do_reset;  run_pack2 (512, "pack2 K=512 (+bias)");
        do_reset;  run_single(128, "single K=128 (+bias)");
        do_reset;  run_single(512, "single K=512 (+bias)");
        do_reset;  test_hold;
        do_reset;  test_drain;
        do_reset;  test_drain_hold;

        if (errors == 0) $display("\n== ALL PASS ==");
        else             $display("\n== %0d FAIL ==", errors);
        $finish;
    end

endmodule
