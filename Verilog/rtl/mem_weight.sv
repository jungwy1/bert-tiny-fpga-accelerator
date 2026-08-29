`timescale 1ns/1ps

module mem_weight #(
    parameter int  DW        = 128,          // data width  (32 × INT4)
    parameter int  DEPTH     = 4096,         // words per bank
    parameter      INIT_FILE = ""            // sim hex preload; "" = uninitialised
)(
    input  logic                      clk,
    // write port  (init weight load)
    input  logic                      we,
    input  logic [$clog2(DEPTH)-1:0]  waddr,
    input  logic [DW-1:0]             wdata,
    // read port   (pmpu operand)  RD_LAT = 1
    input  logic                      re,
    input  logic [$clog2(DEPTH)-1:0]  raddr,
    output logic [DW-1:0]             rdata
);

    (* ram_style = "ultra" *) logic [DW-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;         // write port (init load only)
        if (re) rdata      <= mem[raddr];    // read port (registered output -> URAM)
    end

    initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);

endmodule
