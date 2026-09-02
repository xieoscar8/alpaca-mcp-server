# Hosted AuthKit resource server (not approved for deployment)

The hosted server validates WorkOS AuthKit access tokens directly. WorkOS owns
authorization, login/consent, PKCE, CIMD/DCR and refresh. The MCP server does not
issue reference tokens or host authorization, registration, callback or token
exchange endpoints. Clients must reconnect when migrating from the old proxy.

## Required server configuration

- `ALPACA_MCP_AUTHKIT_ISSUER`: exact HTTPS AuthKit origin, without a trailing slash,
  path, query or fragment. This must equal the issuer in WorkOS metadata and JWTs.
- `ALPACA_MCP_PUBLIC_BASE_URL`: canonical public HTTPS origin, without a trailing
  slash. No real deployment hostname is supplied by the application.
- `ALPACA_MCP_OIDC_AUDIENCE`: exact public MCP endpoint URL. For the planned HTTP
  endpoint this is the configured public base URL plus `/mcp`.
- `ALPACA_MCP_OAUTH_SCOPES`: nonempty, space-separated required OAuth scopes.
  There is no permissive default and no automatic fallback to `openid`.
- `DATABASE_URL`: still required for the durable Safe Trading risk ledger.
- `ALPACA_SAFE_OWNERSHIP_SECRET`: still required and validated as before.

The factory retains its existing `build_managed_oidc_provider` name to minimize
call-site churn, but returns `HardenedAuthKitProvider`, not OIDCProxy.

Upstream OAuth client ID/secret, discovery config URL and the MCP proxy JWT
signing key are no longer used by this provider. It constructs no OAuth storage.
Existing OAuth tables are NOT dropped or migrated. Cleanup requires separate
approval. Risk tables, HMAC ownership, reservations and reconciliation are unchanged.

## Security contract

Only the default trusted FastMCP JWT verifier is used. Its signature, issuer,
audience, scope and strict expiration checks precede the project's finite
NumericDate validation and SDK subject binding. `exp` is mandatory; `nbf` and
`iat` are optional and cannot be more than 60 seconds in the future. Booleans,
strings, null, containers and non-finite numbers are rejected. The additional
gate never revives a token rejected upstream as expired. No maximum lifetime
policy is invented. JWKS/verification failures fail closed.

The issuer is not normalized from token data. Protected-resource metadata emits
the exact configured issuer without the trailing slash that generic URL models
would add. The principal remains `oauth-v2-` plus SHA256 of verified issuer,
NUL, and verified subject. Token refresh must preserve issuer/sub to preserve
the principal; sessions additionally retain the SDK's client/issuer/subject binding.

For `/mcp`, metadata is at `/.well-known/oauth-protected-resource/mcp` and the
401 challenge points there. Its resource and the JWT audience must match the
actual public MCP endpoint; mount mismatch fails during app construction.
`/.well-known/oauth-authorization-server` only forwards WorkOS discovery metadata.
There is no server-side refresh or token storage. Transport settings are unchanged;
this does not claim MCP 2026-07-28 support.

## Remaining deployment blockers

WorkOS Resource Indicators and the actual public endpoint are not configured by
this change. Confirm the exact same resource in WorkOS, metadata and token `aud`.
No Connect Application is assumed necessary: test CIMD first and DCR separately.
Validate the issuer representation across real discovery documents.

WorkOS staging currently advertises standard scopes only. Custom Safe MCP scopes
and the allowed-user/privilege policy remain unproven. Authentication is not
trading authorization: do not remove required scopes or choose weaker scopes
merely to make login succeed. A finalized, explicitly approved authorization
policy and real Claude/WorkOS interoperability remain deployment gates.

Initial real interoperability testing must be authentication and read-only MCP
only, with no Safe write handler invoked and no Alpaca API calls. Test token
refresh, reconnect, server restart, JWKS failure/rotation and risk DB restart
before any deployment approval. Paper-only, exactly three Safe writes and the
disabled close-position tool remain unchanged.
