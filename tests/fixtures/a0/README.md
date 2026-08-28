# A0 API fixtures

These files are declaration/import candidates, not implementation.

The positive fixtures resolve every binding and conceptual public module without
calling declarations. The wrong-arity fixture must fail compilation. The unsupported
fixture must compile and then fail at runtime. The harness repeats compile-only
commands and executable output, and checks the expected negative phase/diagnostic.

These fixtures prove no tensor shapes, dtypes, devices, gradients, performance, or
runtime capability. Those require R0 and downstream numerical tests.
