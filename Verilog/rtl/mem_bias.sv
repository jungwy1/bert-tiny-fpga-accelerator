`timescale 1ns/1ps

module mem_bias #(
    parameter int  DW        = 64,           // data width  (2 × INT32)
    parameter int  DEPTH     = 1536,         // words (512×3-stack)
    parameter      INIT_FILE = ""            // sim hex preload; "" = uninitialised
)(
    input  logic                      clk,
    // write port  (init bias load)
    input  logic                      we,
    input  logic [$clog2(DEPTH)-1:0]  waddr,
    input  logic [DW-1:0]             wdata,
    // read port   (pmpu bias preload)  RD_LAT = 1
    input  logic                      re,
    input  logic [$clog2(DEPTH)-1:0]  raddr,
    output logic [DW-1:0]             rdata
);

    (* ram_style = "block" *) logic [DW-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;         // write port (init load only)
        if (re) rdata      <= mem[raddr];    // read port (registered output -> BRAM)
    end

    initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);

endmodule
