`timescale 1ns/1ps

module mem_act #(
    parameter int  DW        = 128,          // data width  (16 × INT8)
    parameter int  DEPTH     = 512,          // words per bank
    parameter      INIT_FILE = ""            // sim hex preload; "" = uninitialised
)(
    input  logic                      clk,
    // write port  (VFU drain result / PS load)
    input  logic                      we,
    input  logic [$clog2(DEPTH)-1:0]  waddr,
    input  logic [DW-1:0]             wdata,
    // read port   (pmpu operand / PS readback)  -- RD_LAT = 1 (registered)
    input  logic                      re,
    input  logic [$clog2(DEPTH)-1:0]  raddr,
    output logic [DW-1:0]             rdata
);

    (* ram_style = "block" *) logic [DW-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;         // write port
        if (re) rdata      <= mem[raddr];    // read port (registered output -> BRAM)
    end
    // Note: read-during-write to the same address is not exercised (different banks);
    // SDP collision returns old data, which never reaches the datapath.

    // synthesis translate_off
    initial if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
    // synthesis translate_on

endmodule
