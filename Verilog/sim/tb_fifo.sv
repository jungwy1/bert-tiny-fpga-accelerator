`timescale 1ns/1ps
// Testbench for fifo (sync, FWFT, almost_full).
// Scoreboard = SV queue mirroring the FIFO. Each cycle: check DUT flags/head against the
// model, apply the same do_wr/do_rd, update the model. Covers fill-to-full, overflow &
// underflow protection, FWFT data order, almost_full threshold, and random wr/rd mix.
module tb_fifo;
    localparam int W = 525, DEPTH = 8, AF = 2;

    logic clk = 0, rst;
    logic wr_en;  logic [W-1:0] wr_data;
    logic full, almost_full;
    logic rd_en;  logic [W-1:0] rd_data;  logic empty;

    fifo #(.W(W), .DEPTH(DEPTH), .AF(AF)) dut (.*);

    always #5 clk = ~clk;

    logic [W-1:0] model [$];
    int errors = 0;

    function automatic logic [W-1:0] rnd();
        logic [W+31:0] t;
        for (int i = 0; i <= W; i += 32) t[i +: 32] = $urandom;
        return t[W-1:0];
    endfunction

    task automatic err(string m);
        errors++;
        if (errors <= 10)
            $display("  [ERR] %-14s t=%0t size=%0d full=%b afull=%b empty=%b",
                     m, $time, model.size(), full, almost_full, empty);
    endtask

    // DUT flags/head vs model (state is stable between edges)
    task automatic check;
        if (full        !== (model.size() == DEPTH))        err("full");
        if (empty       !== (model.size() == 0))            err("empty");
        if (almost_full !== (model.size() >= DEPTH - AF))   err("almost_full");
        if (!empty && rd_data !== model[0])                 err("rd_data");
    endtask

    // one cycle: drive wr/rd, check pre-edge state, clock, mirror effect into model
    task automatic step(logic w, logic [W-1:0] d, logic r);
        logic dw, dr;
        wr_en = w; wr_data = d; rd_en = r;
        #1; check();
        dw = w & ~full;
        dr = r & ~empty;
        @(posedge clk);
        if (dr) void'(model.pop_front());
        if (dw) model.push_back(d);
        #1;
    endtask

    task automatic do_reset;
        rst = 1; wr_en = 0; rd_en = 0; wr_data = 0; model.delete();
        @(posedge clk); @(posedge clk); #1; rst = 0; #1;
    endtask

    initial begin
        do_reset;

        // 1) fill to full (check() validates almost_full onset each cycle)
        for (int i = 0; i < DEPTH; i++) step(1'b1, rnd(), 1'b0);
        if (!full) err("not full after DEPTH writes");

        // 2) overflow: write while full -> ignored
        step(1'b1, rnd(), 1'b0);
        if (model.size() != DEPTH) err("overflow changed size");

        // 3) drain to empty (FWFT order checked in step)
        for (int i = 0; i < DEPTH; i++) step(1'b0, '0, 1'b1);
        if (!empty) err("not empty after DEPTH reads");

        // 4) underflow: read while empty -> ignored
        step(1'b0, '0, 1'b1);
        if (model.size() != 0) err("underflow changed size");

        // 5) random wr/rd mix (incl. simultaneous)
        for (int i = 0; i < 5000; i++)
            step($urandom_range(0,1), rnd(), $urandom_range(0,1));

        if (errors == 0) $display("\n== ALL PASS ==");
        else             $display("\n== %0d FAIL ==", errors);
        $finish;
    end
endmodule
