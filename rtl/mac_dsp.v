// 2-way INT8 packed MAC on a directly-instantiated DSP48E2, so the pre-adder
// (packing) and the accumulator (psum) run INSIDE the slice instead of fabric.
//
//   pre-adder :  AD = D + A  =  (b << 18) + c        <- the INT8 packing
//   multiplier:  M  = AD * B =  packed * a           <- 27 x 18
//   ALU/accum :  P  = P + M                           <- psum, in the P register
//
// Only PREG is enabled, so latency is 1 cycle. res1/res2 unpacking (upper vs
// lower product + borrow fix) stays in fabric; it is not a MAC.
module mac_dsp (
    input clk, rst,
    input signed [7:0] a,        // shared activation  -> B port
    input signed [7:0] b, c,     // two packed weights -> pre-adder (D, A)
    output reg signed [31:0] res1, res2
);
    wire signed [26:0] D_in = {b[7], b[7:0], 18'b0};     // b in bits [25:18]
    wire signed [29:0] A_in = {{22{c[7]}}, c[7:0]};      // c sign-extended
    wire signed [17:0] B_in = {{10{a[7]}}, a[7:0]};      // a sign-extended

    wire signed [47:0] P;

    DSP48E2 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"),
        .AMULTSEL("AD"),          // multiplier A-side = pre-adder result (D+A)
        .BMULTSEL("B"),
        .PREADDINSEL("A"),        // pre-adder result feeds the A-side of the multiplier
        .USE_PATTERN_DETECT("NO_PATDET"),
        // pipeline: accumulator register only -> 1-cycle latency, same as PE.v
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(1),
        .CREG(0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0)
    ) u_dsp (
        .CLK(clk),
        .A(A_in), .D(D_in), .B(B_in), .C(48'b0),
        .INMODE(5'b00100),        // pre-adder = D + A (add), A2, B2
        .OPMODE(9'b000100101),    // W=00, Z=010(P), Y=01(M), X=01(M) -> P = P + M
        .ALUMODE(4'b0000),        // Z + X + Y + CIN  (add)
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0),
        .CED(1'b0), .CEAD(1'b0), .CEM(1'b0), .CEP(1'b1),
        .CEALUMODE(1'b0), .CECARRYIN(1'b0), .CECTRL(1'b0), .CEINMODE(1'b0),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0),
        .RSTP(rst), .RSTCTRL(1'b0), .RSTINMODE(1'b0), .RSTALUMODE(1'b0),
        .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(P),
        .PCOUT(), .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT()
    );

    // unpack: upper product (b*a) sits above bit 18, lower product (c*a) below;
    // when the lower product is negative its sign borrows 1 from the upper half.
    wire signed [31:0] _res1 = {{14{P[35]}}, P[35:18]};
    wire signed [31:0] _res2 = {{14{P[17]}}, P[17:0]};
    always @(*) begin
        res1 = _res1 + P[17];
        res2 = _res2;
    end
endmodule
