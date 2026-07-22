// 2-way INT8 packed MAC with the DSP48E2 as a MULTIPLIER ONLY; accumulation is
// done outside, in fabric.
//
//   DSP    : pre-adder packs (b<<18)+c, 27x18 multiply, outputs raw P = M
//            (Z-mux = 0, so NO in-DSP accumulation).
//   fabric : unpack this cycle's two products b*a and c*a (borrow-corrected) and
//            add each into its own wide accumulator.
//
// Because each product lands in a separate accumulator, the packed lower field can
// never overflow -> accumulation depth is unlimited (bounded only by the 32-bit
// accumulator), unlike mac_dsp which is capped at ~8 by the 18-bit packing gap.
//
// Latency 1 cycle (combinational DSP multiply + registered fabric accumulate).
module mac (
    input clk, rst,
    input signed [7:0] a,        // shared activation  -> B port
    input signed [7:0] b, c,     // two packed weights -> pre-adder (D, A)
    output reg signed [31:0] res1,   // Sum(b*a)
    output reg signed [31:0] res2    // Sum(c*a)
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
        .PREADDINSEL("A"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        // no registers inside the DSP: pure combinational packed multiplier
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(0),
        .CREG(0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0)
    ) u_dsp (
        .CLK(clk),
        .A(A_in), .D(D_in), .B(B_in), .C(48'b0),
        .INMODE(5'b00100),        // pre-adder = D + A (add), A2, B2
        .OPMODE(9'b000000101),    // W=00, Z=000(0), Y=01(M), X=01(M) -> P = M
        .ALUMODE(4'b0000),        // Z + X + Y + CIN  (add), Z=0 so just M
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0),
        .CED(1'b0), .CEAD(1'b0), .CEM(1'b0), .CEP(1'b0),
        .CEALUMODE(1'b0), .CECARRYIN(1'b0), .CECTRL(1'b0), .CEINMODE(1'b0),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0),
        .RSTP(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0), .RSTALUMODE(1'b0),
        .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(P),
        .PCOUT(), .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT()
    );

    // unpack THIS cycle's two products (single-product borrow correction):
    // upper product b*a sits above bit 18; when the lower product c*a is negative
    // it borrows 1 from the upper half, which P[17] adds back.
    wire signed [31:0] prod_hi = {{14{P[35]}}, P[35:18]};          // b*a - borrow
    wire signed [31:0] prod_lo = {{14{P[17]}}, P[17:0]};           // c*a

    // accumulate each product in its own fabric register -> deep accumulation.
    // P[17] is THIS cycle's borrow (sign of this cycle's lower product), added
    // back so res1 tracks Sum(b*a) exactly. It is a per-product carry, NOT the
    // accumulated sign res2[31].
    always @(posedge clk) begin
        if (rst) begin
            res1 <= 32'sd0;
            res2 <= 32'sd0;
        end else begin
            res1 <= res1 + prod_hi + P[17];
            res2 <= res2 + prod_lo;
        end
    end
endmodule
