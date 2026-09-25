# G3-C4 Step 7A private provider-route admission contract

This bounded native leaf admits the already accepted G3-C4 numerical provider
and G3-S sampler provider during private generator construction. It introduces
no exported function, generic resolver, provider registry, numerical invocation,
sampling, token, frame, result, persistence, package, or public generation API.

`ET_G3C4_PROVIDER_ROUTES_PRIVATE` is valid only with the accepted
`ET_G3C4_GENERATOR_PRIVATE` tuple. Without it, all earlier owner, context,
active-call, and generator objects retain their existing boundary.

Construction discovers only `et_g3c4_kernel_provider_v1()` and
`et_g3s_kernel_provider_v1()` through K1. The C4 provider must expose exactly
seven capabilities and admit these deterministic CPU-f32 full-prefix requests,
in order:

1. embedding `[1,4,4,4]` and `[1,4,256,4]`;
2. linear `[1,4,4,4]`, `[1,4,4,8]`, `[1,4,8,4]`, and `[1,4,4,256]`;
3. layer norm `[1,4,4]`, GELU `[1,4,8]`, and residual `[1,4,4]`;
4. head split and merge `[1,4,2,2]`;
5. causal attention `[1,2,2,4,4,2]`.

The G3-S provider must expose exactly two capabilities and admit greedy and
categorical deterministic CPU-f32 requests at `[1,256]`. Capability and
operation names are the exact accepted G3-C4-N and G3-S ABI names.

Both runtimes are staged before context and A2 cache allocation. Any discovery,
route, context, or cache failure destroys every newly staged runtime and leaves
both process-lifetime runtime authorities unpublished. Successful construction
publishes the two runtime pointers together immediately before enrolling the
complete generator context. Later constructors revalidate every required route
and reuse the exact runtime identities. Failure after that point preserves the
accepted identities.

The next numerical leaf may consume these admitted runtimes to implement the
fixed full-prefix forward schedule and explicit sampler transport. This leaf
does not invoke either runtime and supplies no end-to-end generation witness.
