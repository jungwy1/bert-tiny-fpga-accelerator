module config_decoder(
    input logic [2:0] step,
    input logic layer, head,
    input logic [6:0] seq_len_padded,
    output logic [2:0] vfu_op,
    output logic [9:0] M, N, K,
    output logic pack2, bias_en, transpose, num_ln, vfu_res_re,
    output logic [2:0] act_sel, weight_sel, dest_sel,
    output logic [15:0] baseA, baseW, baseB, baseD, baseLN, baseRES,
    output logic [17:0] req_mult,
    output logic [5:0] req_shamt
);

    localparam [2:0] A_RESID=3'd0, A_B0=3'd1, A_B1=3'd2, A_B2=3'd3, A_B3=3'd4, A_H=3'd5;
    localparam [2:0] W_QKVO=3'd0, W_FFN0=3'd1, W_FFN1=3'd2, W_B1=3'd3, W_B2=3'd4;
    localparam [2:0] ST_Q=3'd0, ST_K=3'd1, ST_V=3'd2, ST_SCORES=3'd3, ST_CTX=3'd4, ST_O=3'd5, ST_FFN1=3'd6, ST_FFN2=3'd7;
    localparam [2:0] VFU_RQ = 3'd0, VFU_GELU = 3'd1, VFU_SM = 3'd2, VFU_SM_NL = 3'd3, VFU_LN = 3'd4;

    logic [2:0] MT; 
    assign MT = seq_len_padded[6:4];

    assign transpose = (step == ST_V) ? 1'b1 : 1'b0;
    assign num_ln = (step == ST_FFN2) ? 1'b1 : 1'b0;
    assign vfu_res_re = (step == ST_O) | (step == ST_FFN2);
    assign baseRES = 16'd0;

    always_comb begin // M, N, K decoder
        M = seq_len_padded;
        case (step)
            ST_Q, ST_K, ST_V, ST_O: begin
                N = 10'd128; K = 10'd128;
            end
            ST_SCORES: begin
                N = seq_len_padded; K = 10'd64;
            end
            ST_CTX: begin
                N = 10'd64; K = seq_len_padded;
            end
            ST_FFN1: begin
                N = 10'd512; K = 10'd128;
            end
            ST_FFN2: begin
                N = 10'd128; K = 10'd512;
            end
            default: begin
                N = 10'd0; K = 10'd0;
            end
        endcase
    end
    
    always_comb begin // vfu op decoder
        case (step)
            ST_Q, ST_K, ST_V: begin
                vfu_op = VFU_RQ;
            end
            ST_SCORES: begin
                vfu_op = VFU_SM;
            end
            ST_CTX: begin
                vfu_op = VFU_SM_NL;
            end
            ST_O, ST_FFN2: begin
                vfu_op = VFU_LN;
            end
            ST_FFN1: begin
                vfu_op = VFU_GELU;
            end
            default: begin
                vfu_op = 3'd0;
            end
        endcase
    end

    always_comb begin // pmpu config decoder
        case (step) 
            ST_Q, ST_K, ST_V, ST_O, ST_FFN1, ST_FFN2: begin
                pack2 = 1'b1; bias_en = 1'b1;
            end
            ST_SCORES, ST_CTX: begin
                pack2 = 1'b0; bias_en = 1'b0;
            end
            default: begin
                pack2 = 1'b0; bias_en = 1'b0;
            end
        endcase
    end

    always_comb begin // bank_sel decoder
        case (step) 
            ST_Q: begin
                act_sel = A_RESID; weight_sel = W_QKVO; dest_sel = A_B0;
            end
            ST_K: begin
                act_sel = A_RESID; weight_sel = W_QKVO; dest_sel = A_B1;
            end
            ST_V: begin
                act_sel = A_RESID; weight_sel = W_QKVO; dest_sel = A_B2;
            end
            ST_SCORES: begin
                act_sel = A_B0; weight_sel = W_B1; dest_sel = A_B3;
            end
            ST_CTX: begin
                act_sel = A_B3; weight_sel = W_B2; dest_sel = A_B0;
            end
            ST_O: begin
                act_sel = A_B0; weight_sel = W_QKVO; dest_sel = A_RESID;
            end
            ST_FFN1: begin
                act_sel = A_RESID; weight_sel = layer ? W_FFN1 : W_FFN0; dest_sel = A_H;
            end
            ST_FFN2: begin
                act_sel = A_H; weight_sel = layer ? W_FFN1 : W_FFN0; dest_sel = A_RESID;
            end
            default: begin
                act_sel = 3'd0; weight_sel = 3'd0; dest_sel = 3'd0;
            end
        endcase
    end

    always_comb begin // base address decoder
        case (step)
            ST_Q: begin
                baseA = 16'd0; baseW = layer ? 16'd2048 : 16'd0;
                baseB = layer ? 16'd576 : 16'd0; baseD = 16'd0;
            end
            ST_K: begin
                baseA = 16'd0; baseW = layer ? 16'd2048 + 16'd512 : 16'd512;
                baseB = layer ? 16'd576 + 16'd64 : 16'd64; baseD = 16'd0;
            end
            ST_V: begin
                baseA = 16'd0; baseW = layer ? 16'd2048 + 16'd1024 : 16'd1024;
                baseB = layer ? 16'd576 + 16'd128 : 16'd128; baseD = 16'd0;
            end
            ST_SCORES, ST_CTX: begin
                baseA = head ? MT << 6 : 16'd0; baseW = head ? MT << 6 : 16'd0;
                baseB = 16'd0; baseD = head ? MT << 6 : 16'd0;
            end
            ST_O: begin
                baseA = 16'd0; baseW = layer ? 16'd2048 + 16'd1536 : 16'd1536;
                baseB = layer ? 16'd576 + 16'd192 : 16'd192; baseD = 16'd0;
            end
            ST_FFN1: begin
                baseA = 16'd0; baseW = 16'd0;
                baseB = layer ? 16'd576 + 16'd256 : 16'd256; baseD = 16'd0;
            end
            ST_FFN2: begin
                baseA = 16'd0; baseW = 16'd2048;
                baseB = layer ? 16'd576 + 16'd512 : 16'd512; baseD = 16'd0;
            end
            default: begin
                baseA = 16'd0; baseW = 16'd0; baseB = 16'd0; baseD = 16'd0;
            end
        endcase
    end

    always_comb begin // baseLN decoder
        case (step) 
            ST_O: begin
                baseLN = layer ? 16'd256 : 16'd0;
            end
            ST_FFN2: begin
                baseLN = layer ? 16'd256 + 16'd128 : 16'd128;
            end
            default: begin
                baseLN = 16'd0;
            end
        endcase
    end

    always_comb begin // requant scale decoder
        case (step)
            ST_Q: begin
                req_mult = layer ? 18'd67251 : 18'd90919;
                req_shamt = 6'd20;
            end
            ST_K: begin
                req_mult = layer ? 18'd67940 : 18'd79475;
                req_shamt = 6'd20;
            end
            ST_V: begin
                req_mult = layer ? 18'd122044 : 18'd86499;
                req_shamt = layer ? 6'd21 : 6'd20;
            end
            ST_O: begin
                req_mult = layer ? 18'd68260 : 18'd71895;
                req_shamt = layer ? 6'd19 : 6'd20;
            end
            ST_FFN2: begin
                req_mult = layer ? 18'd124255 : 18'd74655;
                req_shamt = layer ? 6'd20 : 6'd19;
            end
            default: begin
                req_mult = 18'd0; req_shamt = 6'd0;
            end
        endcase
    end

endmodule

