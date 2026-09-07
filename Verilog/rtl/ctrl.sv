module ctrl(
    input logic clk, rst, 
    output logic busy, done,
    // PS
    input logic start,
    input logic [6:0] seq_len,
    // config
    output logic [9:0] M, N, K,
    // pmpu
    output logic pmpu_start, pack2, bias_en,
    output logic [2:0] act_sel, weight_sel,
    output logic [15:0] baseA, baseW, baseB,
    input logic pmpu_busy, pmpu_done,
    // vfu
    output logic vfu_start, num_ln, transpose, vfu_res_re,
    output logic [2:0] vfu_op,
    output logic [2:0] dest_sel,
    output logic [15:0] baseD, baseLN, baseRES,
    output logic [17:0] req_mult,
    output logic [5:0] req_shamt,
    input logic vfu_busy, vfu_done
);

    localparam [9:0] HIDDEN_DIM = 10'd128;
    localparam [9:0] HEAD_DIM = 10'd64;
    localparam [2:0] A_RESID=3'd0, A_B0=3'd1, A_B1=3'd2, A_B2=3'd3, A_B3=3'd4, A_H=3'd5;
    localparam [2:0] W_QKVO=3'd0, W_FFN0=3'd1, W_FFN1=3'd2, W_B1=3'd3, W_B2=3'd4;
    
    // input latch
    logic [6:0] _seq_len;
    logic [6:0] seq_len_padded;
    assign seq_len_padded = (_seq_len + 15) & ~15;

    // counter
    logic layer, head;
    typedef enum logic [2:0] {
        ST_Q, ST_K, ST_V, ST_SCORES, ST_CTX, ST_O, ST_FFN1, ST_FFN2
    } step_t;
    step_t step;

    // FSM
    typedef enum logic [2:0] {
        S_IDLE, S_CFG, S_RUN, S_STEP, S_DONE
    } state_t;
    state_t state;

    always_ff @(posedge clk or posedge rst) begin // control FSM
        // default
        pmpu_start <= 1'b0; vfu_start <= 1'b0; done <= 1'b0;

        if (rst) begin
            state <= S_IDLE; done <= 1'b0; busy <= 1'b0;
            layer <= 1'b0; head <= 1'b0;
        end
        else begin
            unique case (state)
                S_IDLE: begin
                    if (start) begin
                        layer <= 1'b0; head <= 1'b0; step <= ST_Q;
                        _seq_len <= seq_len; busy <= 1'b1;
                        state <= S_CFG;
                    end
                end
                S_CFG: begin
                    if (~pmpu_busy & ~vfu_busy) begin
                        pmpu_start <= 1'b1; vfu_start <= 1'b1;
                        state <= S_RUN;
                    end           
                end
                S_RUN: begin
                    if (vfu_done) begin
                        state <= S_STEP;
                    end
                end
                S_STEP: begin
                    state <= S_CFG;                             
                    unique case (step)
                        ST_CTX: begin
                            if (head == 1'b0) begin
                                head <= 1'b1; step <= ST_SCORES; 
                            end else begin
                                head <= 1'b0; step <= ST_O;    
                            end
                        end
                        ST_FFN2: begin                       
                            if (layer == 1'b0) begin
                                layer <= 1'b1; step <= ST_Q;   
                            end else begin
                                state <= S_DONE;                
                            end
                        end
                        default: step <= step_t'(step + 1'b1);  
                    endcase
                end
                S_DONE: begin
                    done <= 1'b1; busy <= 1'b0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

    // config decoder
    config_decoder cfg_dec (
        .step(step), .layer(layer), .head(head), .seq_len_padded(seq_len_padded),
        .vfu_op(vfu_op), .M(M), .N(N), .K(K), .num_ln(num_ln), .vfu_res_re(vfu_res_re),
        .pack2(pack2), .bias_en(bias_en), .transpose(transpose),
        .act_sel(act_sel), .weight_sel(weight_sel), .dest_sel(dest_sel),
        .baseA(baseA), .baseW(baseW), .baseB(baseB), .baseD(baseD), .baseRES(baseRES),
        .baseLN(baseLN), .req_mult(req_mult), .req_shamt(req_shamt)     
    );

endmodule