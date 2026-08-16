module fifo #(
    parameter int W = 525, parameter int DEPTH = 8,
    parameter int AF = 2      // almost_full: free slots <= AF
)(
    input  logic clk, rst,
    input  logic wr_en, input logic [W-1:0] wr_data,
    output logic full, almost_full,
    input  logic rd_en, output logic [W-1:0] rd_data, output logic empty
);
    localparam int AW = $clog2(DEPTH);
    logic [W-1:0]  mem [DEPTH];
    logic [AW-1:0] wr_ptr, rd_ptr;
    logic [AW:0]   count;                    

    assign full        = (count == DEPTH);
    assign empty       = (count == 0);
    assign almost_full = (count >= DEPTH - AF);
    assign rd_data     = mem[rd_ptr];      

    wire do_wr = wr_en & ~full;
    wire do_rd = rd_en & ~empty;
    always_ff @(posedge clk) begin
        if (rst) begin wr_ptr<=0; rd_ptr<=0; count<=0; end
        else begin
            if (do_wr) begin mem[wr_ptr]<=wr_data; wr_ptr<=wr_ptr+1'b1; end
            if (do_rd) rd_ptr <= rd_ptr + 1'b1;
            case ({do_wr, do_rd})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;       
            endcase
        end
    end
endmodule
