`timescale 1ns/1ps
// 16x16 output-stationary MAC array of `pe` cells (DSP48E2).
//
// Dataflow (broadcast):
//   - act_row[r]  -> every PE in row r      (activation shared along a row)
//   - w_col[c]    -> every PE in column c   (weight shared down a column)
//   - bias_col[c] -> every PE in column c   (per-feature bias, C port, used on init)
//   - init/en/drain/pack2 fan out to all 256 PEs.
//
// Accumulation is in each PE's DSP P register (K streamed over time, K in [1,512]).
// Drain: SHIFT mode shifts P left->right through the PCOUT->PCIN cascade; the
// rightmost column's P is read out on p_drain[] (16 rows/cycle, 16 cycles for the
// tile). Field unpacking / borrow-correction is done downstream (wrapper).
//
// pack2=1 : each PE = 2 MAC (INT8 x two INT4) -> tile is 16 tokens x 32 features
// pack2=0 : each PE = 1 MAC (INT8 x INT8)     -> tile is 16 tokens x 16 features
module pe_array_16x16 (
    input  logic               clk, rst,
    input  logic               init,          // broadcast: tile start (P <- M + bias)
    input  logic               en,            // broadcast: accumulate/shift enable (hold=0)
    input  logic               drain,         // broadcast: SHIFT mode (cascade readout)
    input  logic               pack2,         // broadcast: 1=INT8xINT4 x2, 0=INT8xINT8
    input  logic               hold,
    input  logic signed [7:0]  act_row  [16], // one activation per ROW  (col-broadcast)
    input  logic signed [7:0]  w_col    [16], // one weight per COLUMN   (row-broadcast)
    input  logic signed [47:0] bias_col [16], // one packed bias per COLUMN (C port)
    output logic signed [47:0] p_drain  [16]  // rightmost column P (drain: per row / cycle)
);

    // cascade net: pc[r][c] = pcout of PE(r,c);  PE(r,c).pcin = pc[r][c-1] (left)
    logic signed [47:0] pc [16][16];

    genvar r, c;
    generate
        for (r = 0; r < 16; r++) begin : g_row
            for (c = 0; c < 16; c++) begin : g_col
                pe u_pe (
                    .clk   (clk),
                    .rst   (rst),
                    .drain (drain),
                    .init  (init),
                    .en    (en),
                    .hold  (hold),
                    .pack2 (pack2),
                    .act   (act_row[r]),                    // shared along the row
                    .w     (w_col[c]),                      // shared down the column
                    .bias  (bias_col[c]),                   // per-column bias (C port)
                    .pcin  (c == 0 ? 48'sd0 : pc[r][c-1]),  // left neighbor; leftmost = 0
                    .pcout (pc[r][c])
                );
            end
            assign p_drain[r] = pc[r][15];                  // rightmost column -> readout
        end
    endgenerate

endmodule
