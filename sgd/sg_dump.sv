// Extra simulation top (compiled with -s sg_dump). Adds full-depth dumps of the DUT and the
// golden model into the TB's own wave.vcd, so the verilog-eval TB file stays untouched.
module sg_dump;
  initial begin
    #0;
    $dumpvars(0, tb.top_module1);
    $dumpvars(0, tb.good1);
  end
endmodule
