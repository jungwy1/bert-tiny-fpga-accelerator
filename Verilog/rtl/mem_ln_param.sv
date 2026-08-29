`timescale 1ns/1ps

module mem_ln_param #(
    parameter int  DW        = 72,           // F_final(6) + M_gamma(18) + C_beta(48)
    parameter int  DEPTH     = 512,
    parameter      INIT_FILE = ""
)(
    input  logic                      clk,
    input  logic                      we,
    input  logic [$clog2(DEPTH)-1:0]  waddr,
    input  logic [DW-1:0]             wdata,
    input  logic                      re,
    input  logic [$clog2(DEPTH)-1:0]  raddr,
    output logic [DW-1:0]             rdata
);

    (* ram_style = "block" *) logic [DW-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata      <= mem[raddr];
    end

    initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);

endmodule
