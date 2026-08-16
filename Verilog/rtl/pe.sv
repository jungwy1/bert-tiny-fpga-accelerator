`timescale 1ns/1ps
// Single DSP48E2 MAC cell (in-DSP accumulation) with C-port bias and cascade drain.
//   pack2=1 : INT8 act x two INT4 weights, packed into one multiply (2 MAC/DSP)
//   pack2=0 : INT8 act x one INT8 weight (1 MAC/DSP)
// P is the accumulator: init presets it with bias (P = M + bias), accumulate adds
// products, drain shifts P out through the PCOUT->PCIN cascade (read pcout at the
// end of the chain). Field unpacking / borrow-correction is done downstream.
module pe (
    input  logic               clk, rst,
    input  logic               drain,    // 1: shift P out via PCIN cascade (P <- PCIN)
    input  logic               init,     // 1: tile start, P <- M + bias;  0: P <- P + M
    input  logic               en,       // CEP: gate P update; hold when 0 (must be 1 during drain)
    input  logic               pack2,    // 1: INT8xINT4 pack-2 (2 MAC),  0: INT8xINT8 single (1 MAC)
    input  logic signed [7:0]  act,      // INT8 activation
    input  logic signed [7:0]  w,        // pack2=1: {w_b[3:0], w_a[3:0]} two INT4 weights
                                         // pack2=0: one INT8 weight
    input  logic signed [47:0] bias,     // C port: packed bias, sampled on the init cycle
    input  logic signed [47:0] pcin,     // upstream PE's pcout (drain shift-in)
    output logic signed [47:0] pcout     // = P; to next PE's pcin, read at the chain end
);

    // ALU op (Z-mux): init -> M+C(bias), else P+M; drain -> shift PCIN into P
    localparam logic [8:0] INIT  = 9'b000110101;  // Z=C   : P = M + bias
    localparam logic [8:0] ACCUM = 9'b000100101;  // Z=P   : P = P + M
    localparam logic [8:0] SHIFT = 9'b000010000;  // Z=PCIN: P = PCIN (cascade drain)
    logic [8:0] opmode;
    assign opmode = drain ? SHIFT : (init ? INIT : ACCUM);


    // Pre-adder inputs form the multiplier's A-side (D + A):
    //   pack2=1: D = w_b << 20, A = w_a  -> packed weight (w_b<<20)+w_a, gap 20
    //   pack2=0: D = 0,         A = w    -> single INT8 weight
    logic signed [26:0] D_in, A_in;
    logic signed [17:0] B_in;

    assign D_in   = pack2 ? {{3{w[7]}}, w[7:4], 20'b0} : 27'sd0;
    assign A_in   = pack2 ? {{23{w[3]}}, w[3:0]}       : {{19{w[7]}}, w[7:0]};
    assign B_in   = {{10{act[7]}}, act[7:0]};            // act on the B port (sign-extended)

    // en aligned to the product (1-cycle input-register latency); drives CEP
    logic en_latch;
    always_ff @(posedge clk or posedge rst)
        if (rst) en_latch <= 1'b0;
        else     en_latch <= en;

    DSP48E2 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"),
        .AMULTSEL("AD"),          // multiplier A-side = pre-adder (D+A)
        .BMULTSEL("B"),
        .PREADDINSEL("A"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        // input registers on; product combinational (ADREG/MREG=0); P is the accumulator.
        // C registered (bias); OPMODEREG so init/drain align with the product at the ALU.
        .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1),
        .DREG(1), .ADREG(0), .MREG(0), .PREG(1),
        .CREG(1), .INMODEREG(0), .OPMODEREG(1), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0)
    ) u_dsp (
        .CLK(clk),
        .A(A_in), .D(D_in), .B(B_in), .C(bias),   // C = packed bias (added on init via Z=C)
        .INMODE(5'b00100),        // pre-adder = D + A (add), A2, B2
        .OPMODE(opmode),          // INIT / ACCUM / SHIFT
        .ALUMODE(4'b0000),        // Z + X + Y + CIN (add)
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b1), .CEB1(1'b0), .CEB2(1'b1), .CEC(1'b1),  // CEC: clock C(bias)
        .CED(1'b1), .CEAD(1'b0), .CEM(1'b0), .CEP(en_latch),   // CEP: en gates P (hold 1 during drain)
        .CEALUMODE(1'b0), .CECARRYIN(1'b0), .CECTRL(1'b1), .CEINMODE(1'b0),  // CECTRL: clock OPMODE reg
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0),
        .RSTP(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0), .RSTALUMODE(1'b0),
        .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(pcin),  // PCIN: upstream P (drain shift)
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(),                                     // fabric P unused; read via PCOUT
        .PCOUT(pcout), .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT()
    );

endmodule
